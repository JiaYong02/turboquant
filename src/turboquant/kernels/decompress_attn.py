"""Triton fused decompress+attention kernel (Phase 10, split-KV).

FlashDecoding-v2 shape: two kernels.

1. ``_attn_split_kernel`` — grid ``(B, H_q, NUM_SPLITS)``. Each program owns
   ``SPLIT_SIZE`` key positions, iterates them in ``BLOCK_N`` tiles, and writes
   its local streaming-softmax state ``(m_local, l_local, acc_local)`` to HBM
   scratch. This is the loop body from the old single-pass kernel, now
   parallelised over KV tiles so the grid saturates SMs at small batch.

2. ``_attn_reduce_kernel`` — grid ``(B, H_q)``. Each program reads the
   ``NUM_SPLITS`` partials for its ``(b, h_q)`` and combines them with the
   standard streaming-softmax merge
   ``m* = max m_i``; ``l* = Σ l_i · exp(m_i - m*)``;
   ``acc* = Σ acc_i · exp(m_i - m*)``; ``out = acc* / l*``.

``Pi_v`` is still applied once at the end in Python, outside the kernel.
Scope: decode (S_q = 1), forward only, ``b_key ∈ {2,3,4}``, ``b_val ∈ {2,3,4}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..bit_packing import packed_dim
from ..integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from ..quantizer_mse_torch import BatchedTurboQuantMSETorch
from ..quantizer_prod_torch import BatchedTurboQuantProdTorch

_QJL_C = math.sqrt(math.pi / 2)

_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=[
        "S_bucket", "D", "GQA_GROUPS",
        "MSE_BITS", "VAL_BITS",
        "N_OUT", "MSE_BITS_HI", "MSE_BITS_LO", "VAL_BITS_HI", "VAL_BITS_LO",
        "SPLIT_SIZE",
    ],
)
@triton.jit
def _attn_split_kernel(
    q_ptr,              # [B, H_q, D]      native dtype (fp16/bf16/fp32)
    s_k_ptr,            # [H_q, D, D]      fp32 (QJL per-Q-head basis)
    pi_k_ptr,           # [H_q, D, D]      fp32 (MSE per-Q-head rotation); ignored when MSE_BITS=0 and N_OUT=0
    qjl_packed_ptr,
    # Single-bucket K packing (used when N_OUT=0)
    mse_packed_ptr,
    # Split-bucket K packing (used when N_OUT>0)
    mse_packed_hi_ptr,
    mse_packed_lo_ptr,
    gamma_ptr,
    norms_k_ptr,
    # Single-bucket V packing (used when N_OUT=0)
    val_idx_packed_ptr,
    # Split-bucket V packing (used when N_OUT>0)
    val_idx_packed_hi_ptr,
    val_idx_packed_lo_ptr,
    norms_v_ptr,
    key_centroids_ptr,
    key_centroids_hi_ptr,
    key_centroids_lo_ptr,
    val_centroids_ptr,
    val_centroids_hi_ptr,
    val_centroids_lo_ptr,
    partials_acc_ptr,  # [B, H_q, NUM_SPLITS, D]  fp32
    partials_m_ptr,    # [B, H_q, NUM_SPLITS]     fp32
    partials_l_ptr,    # [B, H_q, NUM_SPLITS]     fp32
    # strides for q: [B, H_q, D]
    sq_b, sq_h,
    # strides for s_k / pi_k: [H_q, D, D]; last-dim stride assumed contiguous
    ssk_h, ssk_e,
    spi_h, spi_e,
    # strides for partials_acc: [B, H_q, NUM_SPLITS, D]
    spa_b, spa_h, spa_s,
    # strides for partials_m / partials_l: [B, H_q, NUM_SPLITS]
    spm_b, spm_h,
    # strides for packed K (single-bucket): [H_kv, B, S, Dp]
    sqp_h, sqp_b, sqp_s,
    smp_h, smp_b, smp_s,
    # strides for split-bucket K packed: [H_kv, B, S, Dp_hi/lo]
    smph_h, smph_b, smph_s,
    smpl_h, smpl_b, smpl_s,
    # scalar K: [H_kv, B, S]
    sg_h, sg_b,
    snk_h, snk_b,
    # V (single-bucket)
    svp_h, svp_b, svp_s,
    # V (split-bucket)
    svph_h, svph_b, svph_s,
    svpl_h, svpl_b, svpl_s,
    snv_h, snv_b,
    # device scalar: 0-d int32 tensor holding current seq_len; loaded at
    # kernel entry so one captured CUDA graph can serve all S_total values
    # within a bucket.
    s_total_ptr,
    S_bucket,   # autotune key — bucketed S_total
    scale,
    qjl_const,
    NUM_SPLITS,
    GQA_GROUPS: tl.constexpr,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    VAL_BITS: tl.constexpr,
    N_OUT: tl.constexpr,
    MSE_BITS_HI: tl.constexpr,
    MSE_BITS_LO: tl.constexpr,
    VAL_BITS_HI: tl.constexpr,
    VAL_BITS_LO: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    hq_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    hkv_idx = hq_idx // GQA_GROUPS

    S_total = tl.load(s_total_ptr)
    s_base = split_idx * SPLIT_SIZE
    s_end = tl.minimum(s_base + SPLIT_SIZE, S_total)

    d_off = tl.arange(0, D)
    e_off = tl.arange(0, D)
    # Load q once and compute q_qjl = q @ S_k[hq], q_rot = q @ Pi_k[hq] in
    # registers. einsum("bhd,hed->bhe", q, X) — X[hq, e, d] multiplied across d.
    q_off = b_idx * sq_b + hq_idx * sq_h + d_off
    q_d = tl.load(q_ptr + q_off).to(tl.float32)
    sk_tile = tl.load(
        s_k_ptr + hq_idx * ssk_h + e_off[:, None] * ssk_e + d_off[None, :]
    )
    q_qjl = tl.sum(q_d[None, :] * sk_tile, axis=1)
    # Pi_k is needed for the MSE centroid dot in both single-bucket
    # (MSE_BITS>0) and split-bucket (N_OUT>0) modes.
    if (MSE_BITS > 0) or (N_OUT > 0):
        pi_tile = tl.load(
            pi_k_ptr + hq_idx * spi_h + e_off[:, None] * spi_e + d_off[None, :]
        )
        q_rot = tl.sum(q_d[None, :] * pi_tile, axis=1)
    else:
        q_rot = tl.zeros([D], dtype=tl.float32)

    # Per-coord unpack indices
    VPB_QJL: tl.constexpr = 8
    qjl_byte_idx = d_off // VPB_QJL
    qjl_shift = (VPB_QJL - 1 - (d_off % VPB_QJL)).to(tl.int32)

    # Bit-tight unpack: value i occupies bits [i*n_bits, (i+1)*n_bits) in the
    # MSB-first stream. Read a 16-bit window (byte_lo, byte_hi), shift down,
    # and mask. A value may span two adjacent bytes.
    mse_mask_val: tl.constexpr = (1 << MSE_BITS) - 1 if MSE_BITS > 0 else 0
    mse_bit_off = d_off * MSE_BITS
    mse_byte_lo = mse_bit_off // 8
    mse_byte_hi = mse_byte_lo + 1
    mse_shift_down = (16 - MSE_BITS - (mse_bit_off % 8)).to(tl.int32)

    val_mask_val: tl.constexpr = (1 << VAL_BITS) - 1
    val_bit_off = d_off * VAL_BITS
    val_byte_lo = val_bit_off // 8
    val_byte_hi = val_byte_lo + 1
    val_shift_down = (16 - VAL_BITS - (val_bit_off % 8)).to(tl.int32)

    # --- Split-bucket per-coord unpack layout (used when N_OUT > 0) ----------
    # Hi stream addresses coord d directly; lo stream is indexed by (d - N_OUT)
    # and only valid for d >= N_OUT. We clamp the negative indices to 0 on the
    # lo side so the address computation produces an in-range pointer; the
    # corresponding lane is masked out via `d_in_lo` at gather time.
    d_in_hi = d_off < N_OUT
    d_lo_coord = tl.maximum(d_off - N_OUT, 0)

    mse_mask_hi_val: tl.constexpr = (1 << (MSE_BITS_HI - 1)) - 1 if MSE_BITS_HI > 0 else 0
    mse_mask_lo_val: tl.constexpr = (1 << (MSE_BITS_LO - 1)) - 1 if MSE_BITS_LO > 0 else 0
    mse_bit_off_hi = d_off * (MSE_BITS_HI - 1)
    mse_byte_lo_hi = mse_bit_off_hi // 8
    mse_byte_hi_hi = mse_byte_lo_hi + 1
    mse_shift_hi = (16 - (MSE_BITS_HI - 1) - (mse_bit_off_hi % 8)).to(tl.int32)
    mse_bit_off_lo = d_lo_coord * (MSE_BITS_LO - 1)
    mse_byte_lo_lo = mse_bit_off_lo // 8
    mse_byte_hi_lo = mse_byte_lo_lo + 1
    mse_shift_lo = (16 - (MSE_BITS_LO - 1) - (mse_bit_off_lo % 8)).to(tl.int32)

    val_mask_hi_val: tl.constexpr = (1 << VAL_BITS_HI) - 1 if VAL_BITS_HI > 0 else 0
    val_mask_lo_val: tl.constexpr = (1 << VAL_BITS_LO) - 1 if VAL_BITS_LO > 0 else 0
    val_bit_off_hi = d_off * VAL_BITS_HI
    val_byte_lo_hi = val_bit_off_hi // 8
    val_byte_hi_hi = val_byte_lo_hi + 1
    val_shift_hi = (16 - VAL_BITS_HI - (val_bit_off_hi % 8)).to(tl.int32)
    val_bit_off_lo = d_lo_coord * VAL_BITS_LO
    val_byte_lo_lo = val_bit_off_lo // 8
    val_byte_hi_lo = val_byte_lo_lo + 1
    val_shift_lo = (16 - VAL_BITS_LO - (val_bit_off_lo % 8)).to(tl.int32)

    m_run = -float("inf")
    l_run = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    kv_base_qjl = hkv_idx * sqp_h + b_idx * sqp_b
    kv_base_mse = hkv_idx * smp_h + b_idx * smp_b
    kv_base_mseh = hkv_idx * smph_h + b_idx * smph_b
    kv_base_msel = hkv_idx * smpl_h + b_idx * smpl_b
    kv_base_g   = hkv_idx * sg_h  + b_idx * sg_b
    kv_base_nk  = hkv_idx * snk_h + b_idx * snk_b
    kv_base_vp  = hkv_idx * svp_h + b_idx * svp_b
    kv_base_vph = hkv_idx * svph_h + b_idx * svph_b
    kv_base_vpl = hkv_idx * svpl_h + b_idx * svpl_b
    kv_base_nv  = hkv_idx * snv_h + b_idx * snv_b

    # If this split is entirely out of range, bail after writing sentinel state.
    if s_base < S_total:
        for s_start in range(s_base, s_end, BLOCK_N):
            s_off = s_start + tl.arange(0, BLOCK_N)
            s_mask = s_off < s_end  # also implies < S_total

            qjl_ptrs = (
                qjl_packed_ptr + kv_base_qjl
                + s_off[:, None] * sqp_s + qjl_byte_idx[None, :]
            )
            qjl_bytes = tl.load(qjl_ptrs, mask=s_mask[:, None], other=0).to(tl.int32)
            qjl_bits = (qjl_bytes >> qjl_shift[None, :]) & 1
            signs = qjl_bits.to(tl.float32) * 2.0 - 1.0

            if N_OUT > 0:
                # Split-bucket K MSE: unpack from hi and lo streams, blend per-coord.
                mse_base_hi = (
                    mse_packed_hi_ptr + kv_base_mseh + s_off[:, None] * smph_s
                )
                hi_mask = s_mask[:, None] & d_in_hi[None, :]
                mseh_lo = tl.load(
                    mse_base_hi + mse_byte_lo_hi[None, :],
                    mask=hi_mask,
                    other=0,
                ).to(tl.int32)
                mseh_hi = tl.load(
                    mse_base_hi + mse_byte_hi_hi[None, :],
                    mask=hi_mask & (mse_byte_hi_hi[None, :] < smph_s),
                    other=0,
                ).to(tl.int32)
                mseh_word = (mseh_lo << 8) | mseh_hi
                hi_vals = (mseh_word >> mse_shift_hi[None, :]) & mse_mask_hi_val

                mse_base_lo = (
                    mse_packed_lo_ptr + kv_base_msel + s_off[:, None] * smpl_s
                )
                lo_mask = s_mask[:, None] & (~d_in_hi[None, :])
                msel_lo = tl.load(
                    mse_base_lo + mse_byte_lo_lo[None, :],
                    mask=lo_mask,
                    other=0,
                ).to(tl.int32)
                msel_hi = tl.load(
                    mse_base_lo + mse_byte_hi_lo[None, :],
                    mask=lo_mask & (mse_byte_hi_lo[None, :] < smpl_s),
                    other=0,
                ).to(tl.int32)
                msel_word = (msel_lo << 8) | msel_hi
                lo_vals = (msel_word >> mse_shift_lo[None, :]) & mse_mask_lo_val

                hi_cent = tl.load(key_centroids_hi_ptr + hi_vals)
                lo_cent = tl.load(key_centroids_lo_ptr + lo_vals)
                key_cent = tl.where(d_in_hi[None, :], hi_cent, lo_cent)
            elif MSE_BITS > 0:
                mse_base = mse_packed_ptr + kv_base_mse + s_off[:, None] * smp_s
                mse_lo = tl.load(
                    mse_base + mse_byte_lo[None, :],
                    mask=s_mask[:, None],
                    other=0,
                ).to(tl.int32)
                mse_hi = tl.load(
                    mse_base + mse_byte_hi[None, :],
                    mask=s_mask[:, None] & (mse_byte_hi[None, :] < smp_s),
                    other=0,
                ).to(tl.int32)
                mse_word = (mse_lo << 8) | mse_hi
                mse_vals = (mse_word >> mse_shift_down[None, :]) & mse_mask_val
                key_cent = tl.load(key_centroids_ptr + mse_vals)
            else:
                key_cent = tl.zeros([BLOCK_N, D], dtype=tl.float32)

            gamma = tl.load(
                gamma_ptr + kv_base_g + s_off, mask=s_mask, other=0.0
            ).to(tl.float32)
            norms_k = tl.load(
                norms_k_ptr + kv_base_nk + s_off, mask=s_mask, other=0.0
            ).to(tl.float32)

            mse_dot = tl.sum(q_rot[None, :] * key_cent, axis=1)
            qjl_dot = tl.sum(q_qjl[None, :] * signs, axis=1)
            logits = norms_k * (mse_dot + qjl_const * gamma * qjl_dot) * scale
            logits = tl.where(s_mask, logits, -float("inf"))

            m_new = tl.maximum(m_run, tl.max(logits, axis=0))
            alpha = tl.exp(m_run - m_new)
            p = tl.exp(logits - m_new)
            l_run = l_run * alpha + tl.sum(p, axis=0)

            if N_OUT > 0:
                # Split-bucket V: unpack hi and lo streams, blend per-coord.
                val_base_hi = (
                    val_idx_packed_hi_ptr + kv_base_vph + s_off[:, None] * svph_s
                )
                vh_mask = s_mask[:, None] & d_in_hi[None, :]
                vh_lo = tl.load(
                    val_base_hi + val_byte_lo_hi[None, :],
                    mask=vh_mask,
                    other=0,
                ).to(tl.int32)
                vh_hi = tl.load(
                    val_base_hi + val_byte_hi_hi[None, :],
                    mask=vh_mask & (val_byte_hi_hi[None, :] < svph_s),
                    other=0,
                ).to(tl.int32)
                vh_word = (vh_lo << 8) | vh_hi
                v_hi_vals = (vh_word >> val_shift_hi[None, :]) & val_mask_hi_val

                val_base_lo = (
                    val_idx_packed_lo_ptr + kv_base_vpl + s_off[:, None] * svpl_s
                )
                vl_mask = s_mask[:, None] & (~d_in_hi[None, :])
                vl_lo = tl.load(
                    val_base_lo + val_byte_lo_lo[None, :],
                    mask=vl_mask,
                    other=0,
                ).to(tl.int32)
                vl_hi = tl.load(
                    val_base_lo + val_byte_hi_lo[None, :],
                    mask=vl_mask & (val_byte_hi_lo[None, :] < svpl_s),
                    other=0,
                ).to(tl.int32)
                vl_word = (vl_lo << 8) | vl_hi
                v_lo_vals = (vl_word >> val_shift_lo[None, :]) & val_mask_lo_val

                v_hi_cent = tl.load(val_centroids_hi_ptr + v_hi_vals)
                v_lo_cent = tl.load(val_centroids_lo_ptr + v_lo_vals)
                val_cent = tl.where(d_in_hi[None, :], v_hi_cent, v_lo_cent)
            else:
                val_base = val_idx_packed_ptr + kv_base_vp + s_off[:, None] * svp_s
                v_lo = tl.load(
                    val_base + val_byte_lo[None, :],
                    mask=s_mask[:, None],
                    other=0,
                ).to(tl.int32)
                v_hi = tl.load(
                    val_base + val_byte_hi[None, :],
                    mask=s_mask[:, None] & (val_byte_hi[None, :] < svp_s),
                    other=0,
                ).to(tl.int32)
                v_word = (v_lo << 8) | v_hi
                v_vals = (v_word >> val_shift_down[None, :]) & val_mask_val
                val_cent = tl.load(val_centroids_ptr + v_vals)

            norms_v = tl.load(
                norms_v_ptr + kv_base_nv + s_off, mask=s_mask, other=0.0
            ).to(tl.float32)
            weights = p * norms_v
            contrib = tl.sum(weights[:, None] * val_cent, axis=0)

            acc = acc * alpha + contrib
            m_run = m_new

    # Store partials (always, so the reduce reads a valid sentinel for empty splits).
    pm_off = b_idx * spm_b + hq_idx * spm_h + split_idx
    tl.store(partials_m_ptr + pm_off, m_run)
    tl.store(partials_l_ptr + pm_off, l_run)

    pa_off = b_idx * spa_b + hq_idx * spa_h + split_idx * spa_s + d_off
    tl.store(partials_acc_ptr + pa_off, acc)


@triton.jit
def _attn_reduce_kernel(
    partials_acc_ptr,
    partials_m_ptr,
    partials_l_ptr,
    pi_v_ptr,           # [H_q, D, D] fp32 — canonical-V rotation matrix
    out_ptr,
    # strides
    spa_b, spa_h, spa_s,
    spm_b, spm_h,
    spv_h, spv_d,       # pi_v: [H_q, D, D], last-dim stride assumed contiguous
    so_b, so_h,
    NUM_SPLITS,
    D: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,  # >= NUM_SPLITS, power of 2
):
    b_idx = tl.program_id(0)
    hq_idx = tl.program_id(1)

    s_off = tl.arange(0, BLOCK_SPLITS)
    s_mask = s_off < NUM_SPLITS

    pm_off = b_idx * spm_b + hq_idx * spm_h + s_off
    m_i = tl.load(partials_m_ptr + pm_off, mask=s_mask, other=-float("inf"))
    l_i = tl.load(partials_l_ptr + pm_off, mask=s_mask, other=0.0)

    m_star = tl.max(m_i, axis=0)
    scale_i = tl.exp(m_i - m_star)
    l_star = tl.sum(l_i * scale_i, axis=0)

    d_off = tl.arange(0, D)
    e_off = tl.arange(0, D)
    # acc[s, d] = partials_acc[b, hq, s, d]
    pa_off = (
        b_idx * spa_b + hq_idx * spa_h
        + s_off[:, None] * spa_s + d_off[None, :]
    )
    acc_i = tl.load(pa_off + partials_acc_ptr, mask=s_mask[:, None], other=0.0)
    acc_star = tl.sum(acc_i * scale_i[:, None], axis=0)  # [D] in rotated-V space

    # Fold Pi_v: out[e] = sum_d acc_star[d] * pi_v[h_q, d, e] / l_star
    pi_v_tile = tl.load(
        pi_v_ptr + hq_idx * spv_h + d_off[:, None] * spv_d + e_off[None, :]
    )
    out = tl.sum(acc_star[:, None] * pi_v_tile, axis=0) / l_star

    out_off = b_idx * so_b + hq_idx * so_h + e_off
    tl.store(out_ptr + out_off, out)


def _key_centroid_lut(kq: BatchedTurboQuantProdTorch) -> torch.Tensor:
    if kq.mse is None or kq.mse.codebook is None:
        return torch.zeros(1, dtype=torch.float32)
    return kq.mse.codebook._centroids.to(torch.float32).contiguous()


def _val_centroid_lut(vq: BatchedTurboQuantMSETorch) -> torch.Tensor:
    if vq.codebook is None:
        return torch.zeros(1, dtype=torch.float32)
    return vq.codebook._centroids.to(torch.float32).contiguous()


def _key_centroid_lut_split(
    kq: BatchedTurboQuantProdTorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kq.n_out == 0 or kq.mse is None:
        raise ValueError("split centroid LUTs require n_out > 0")
    return (
        kq.mse.codebook_hi._centroids.to(torch.float32).contiguous(),
        kq.mse.codebook_lo._centroids.to(torch.float32).contiguous(),
    )


def _val_centroid_lut_split(
    vq: BatchedTurboQuantMSETorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vq.n_out == 0:
        raise ValueError("split centroid LUTs require n_out > 0")
    return (
        vq.codebook_hi._centroids.to(torch.float32).contiguous(),
        vq.codebook_lo._centroids.to(torch.float32).contiguous(),
    )


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


_DUMMY_PI_K_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def _dummy_pi_k(device: torch.device, H_q: int, D: int) -> torch.Tensor:
    """Return a cached dummy Pi_k tensor for MSE_BITS=0 kernel invocations.

    The kernel only reads from this pointer when MSE_BITS>0 so the contents
    don't matter; what matters is a stable, valid device pointer.
    """
    key = (str(device), H_q, D)
    t = _DUMMY_PI_K_CACHE.get(key)
    if t is None:
        t = torch.empty((H_q, D, D), dtype=torch.float32, device=device)
        _DUMMY_PI_K_CACHE[key] = t
    return t


def _s_bucket(s: int) -> int:
    for b in (256, 1024, 4096, 16384, 65536):
        if s <= b:
            return b
    return _next_pow2(s)


def _pick_split_size(S_total: int, B: int, H_q: int, sm_count: int = 132) -> int:
    """Pick SPLIT_SIZE so grid (B*H_q*NUM_SPLITS) saturates SMs (>= 2x sm_count)."""
    target = max(1, 2 * sm_count // max(1, B * H_q))
    # NUM_SPLITS >= target  ⇒  SPLIT_SIZE <= S_total / target
    if target <= 1 or S_total <= 64:
        return max(64, _next_pow2(S_total))
    split = max(32, _next_pow2(max(32, S_total // target)))
    # Clamp to avoid pathological tiny splits.
    return min(split, max(64, S_total))


@dataclass
class FusedAttnScratch:
    """Long-lived scratch buffers for CUDA-graph decode.

    Shapes must accommodate the bucket's worst case:
      partials_acc: [B, H_q, num_splits_max, D]  fp32
      partials_m:   [B, H_q, num_splits_max]     fp32
      partials_l:   [B, H_q, num_splits_max]     fp32
      out:          [B, H_q, D]                  fp32 (canonical V-space)

    Q-side rotations (q @ S_k, q @ Pi_k) are fused into ``_attn_split_kernel``
    as of Phase 12A. Pi_v is fused into ``_attn_reduce_kernel`` as of 12B, so
    ``out`` already lives in canonical V-space — no post-kernel einsum needed.
    """
    partials_acc: torch.Tensor
    partials_m: torch.Tensor
    partials_l: torch.Tensor
    out: torch.Tensor
    num_splits_max: int
    split_size: int


def fused_decompress_attention_triton(
    q: torch.Tensor,
    key_store: _BatchedKeyStore,
    value_store: _BatchedValueStore,
    key_quantizer: BatchedTurboQuantProdTorch,
    value_quantizer: BatchedTurboQuantMSETorch,
    scale: float | None = None,
    split_size: int | None = None,
    block_n: int | None = None,  # deprecated: BLOCK_N is autotuned now
    pi_k_per_q: torch.Tensor | None = None,
    s_k_per_q: torch.Tensor | None = None,
    pi_v_per_q: torch.Tensor | None = None,
    kv_for_q: torch.Tensor | None = None,
    key_cent_lut: torch.Tensor | None = None,
    val_cent_lut: torch.Tensor | None = None,
    key_cent_lut_hi: torch.Tensor | None = None,
    key_cent_lut_lo: torch.Tensor | None = None,
    val_cent_lut_hi: torch.Tensor | None = None,
    val_cent_lut_lo: torch.Tensor | None = None,
    s_total_tensor: torch.Tensor | None = None,
    scratch: "FusedAttnScratch | None" = None,
) -> torch.Tensor:
    """Fused decompress+attention via Triton (FlashDecoding split-KV), decode-only."""
    if q.dim() != 4 or q.shape[2] != 1:
        raise ValueError(f"expected q shape [B,H_q,1,D], got {tuple(q.shape)}")
    if not q.is_cuda:
        raise RuntimeError("Triton fused kernel requires a CUDA device")

    n_out = key_quantizer.n_out
    if n_out != value_quantizer.n_out:
        raise ValueError(
            f"key.n_out={n_out} must equal value.n_out={value_quantizer.n_out}"
        )
    split_mode = n_out > 0

    if split_mode:
        if key_store.qjl_packed is None or value_store.idx_packed_hi is None:
            raise ValueError("KV store is empty (split-bucket)")
    else:
        if key_store.qjl_packed is None or value_store.idx_packed is None:
            raise ValueError("KV store is empty")

    B, H_q, _, D = q.shape
    H_kv = key_quantizer.num_heads
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    gqa_groups = H_q // H_kv

    S_total = key_store.seq_length()
    if S_total == 0:
        return torch.zeros_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    device = q.device
    in_dtype = q.dtype
    key_bits = key_quantizer.b
    value_bits = value_quantizer.b
    mse_bits = 0 if split_mode else (key_bits - 1)
    if split_mode:
        mse_bits_hi = key_quantizer.bits_hi
        mse_bits_lo = key_quantizer.bits_lo
        val_bits_hi = value_quantizer.bits_hi
        val_bits_lo = value_quantizer.bits_lo
        value_bits = 0  # single-bucket field unused in split mode
    else:
        mse_bits_hi = 0
        mse_bits_lo = 0
        val_bits_hi = 0
        val_bits_lo = 0

    # --- Folded-rotation trick --------------------------------------------
    # Per-Q-head rotation matrices (S_k_per_q, Pi_k_per_q, Pi_v_per_q) are
    # invariant for the lifetime of a cache layer; callers (e.g.
    # TurboQuantCacheLayer._prepare_fused_state) can pass them in to skip the
    # per-call index_select + fp32 cast. Phase 12A folds the q @ S_k and
    # q @ Pi_k mat-vecs into the split kernel itself — q is loaded once in
    # native dtype and rotated in-register.
    q_3d = q.squeeze(2).contiguous()  # [B, H_q, D] in native dtype
    if kv_for_q is None:
        kv_for_q = torch.arange(H_q, device=device) // gqa_groups

    if s_k_per_q is None:
        S_k = key_quantizer.qjl.S.to(device=device, dtype=torch.float32)
        s_k_per_q = S_k.index_select(0, kv_for_q).contiguous()

    if mse_bits > 0 or split_mode:
        if pi_k_per_q is None:
            Pi_k = key_quantizer.mse.rotation.Pi.to(device=device, dtype=torch.float32)
            pi_k_per_q = Pi_k.index_select(0, kv_for_q).contiguous()
    else:
        # Kernel still needs a valid pointer even when MSE_BITS=0; it just
        # won't read from it (the branch returns a zero register tile). Use
        # a per-(device,shape) cached dummy so we don't allocate 64 KB every
        # call on the eager path.
        pi_k_per_q = _dummy_pi_k(device, H_q, D)

    if pi_v_per_q is None:
        Pi_v = value_quantizer.rotation.Pi.to(device=device, dtype=torch.float32)
        pi_v_per_q = Pi_v.index_select(0, kv_for_q).contiguous()

    # --- Gather packed tensors --------------------------------------------
    # gamma / norms are fp16 in storage (Phase A); kernel upcasts per-element
    # via `tl.load(...).to(tl.float32)` so no whole-buffer cast here.
    #
    # NOTE: `mse_packed` / `val_idx_packed` MUST be contiguous in the last
    # dim. The kernel uses `smp_s` / `svp_s` (per-S stride) as the packed_D
    # bound in its hi-byte OOB mask for the 16-bit-window bit-tight unpack;
    # that is correct only when `stride[-1] == 1` and `stride[-2] == packed_D`.
    # A non-contiguous view would silently read into the next token's bytes.
    qjl_packed = key_store.qjl_packed.contiguous()
    gamma = key_store.gamma.contiguous()
    norms_k = key_store.norms.contiguous()
    norms_v = value_store.norms.contiguous()

    _dummy_u8 = torch.zeros(1, dtype=torch.uint8, device=device)
    _dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)

    if split_mode:
        mse_packed = _dummy_u8
        mse_packed_hi = key_store.mse_packed_hi.contiguous()
        mse_packed_lo = key_store.mse_packed_lo.contiguous()
        val_idx_packed = _dummy_u8
        val_idx_packed_hi = value_store.idx_packed_hi.contiguous()
        val_idx_packed_lo = value_store.idx_packed_lo.contiguous()

        if key_cent_lut_hi is None or key_cent_lut_lo is None:
            key_cent_lut_hi, key_cent_lut_lo = _key_centroid_lut_split(key_quantizer)
            key_cent_lut_hi = key_cent_lut_hi.to(device).contiguous()
            key_cent_lut_lo = key_cent_lut_lo.to(device).contiguous()
        if val_cent_lut_hi is None or val_cent_lut_lo is None:
            val_cent_lut_hi, val_cent_lut_lo = _val_centroid_lut_split(value_quantizer)
            val_cent_lut_hi = val_cent_lut_hi.to(device).contiguous()
            val_cent_lut_lo = val_cent_lut_lo.to(device).contiguous()
        key_cent_lut = _dummy_f32
        val_cent_lut = _dummy_f32
    else:
        if key_store.mse_packed is not None:
            mse_packed = key_store.mse_packed.contiguous()
        else:
            mse_packed = _dummy_u8
        mse_packed_hi = _dummy_u8
        mse_packed_lo = _dummy_u8
        val_idx_packed = value_store.idx_packed.contiguous()
        val_idx_packed_hi = _dummy_u8
        val_idx_packed_lo = _dummy_u8

        if key_cent_lut is None:
            key_cent_lut = _key_centroid_lut(key_quantizer).to(device).contiguous()
        if val_cent_lut is None:
            val_cent_lut = _val_centroid_lut(value_quantizer).to(device).contiguous()
        key_cent_lut_hi = _dummy_f32
        key_cent_lut_lo = _dummy_f32
        val_cent_lut_hi = _dummy_f32
        val_cent_lut_lo = _dummy_f32

    # --- Split sizing ------------------------------------------------------
    if scratch is not None:
        split_size = scratch.split_size
        num_splits = scratch.num_splits_max
    else:
        if split_size is None:
            split_size = _pick_split_size(S_total, B, H_q)
        split_size = max(32, int(split_size))
        num_splits = (S_total + split_size - 1) // split_size

    # --- Partial + output scratch -----------------------------------------
    if scratch is not None:
        partials_acc = scratch.partials_acc
        partials_m = scratch.partials_m
        partials_l = scratch.partials_l
        out = scratch.out
    else:
        partials_acc = torch.empty(B, H_q, num_splits, D, dtype=torch.float32, device=device)
        partials_m = torch.empty(B, H_q, num_splits, dtype=torch.float32, device=device)
        partials_l = torch.empty(B, H_q, num_splits, dtype=torch.float32, device=device)
        out = torch.empty(B, H_q, D, dtype=torch.float32, device=device)

    if s_total_tensor is None:
        s_total_tensor = torch.tensor(S_total, dtype=torch.int32, device=device)

    def st(t: torch.Tensor) -> list[int]:
        return list(t.stride())

    sqd = st(q_3d)                            # [B,H_q,D]
    ssk = st(s_k_per_q)                       # [H_q,D,D]
    spi = st(pi_k_per_q)                      # [H_q,D,D]
    spv = st(pi_v_per_q)                      # [H_q,D,D]
    sqjlp = st(qjl_packed)
    smsep = st(mse_packed) if mse_packed.dim() == 4 else [0, 0, 0, 0]
    smseph = st(mse_packed_hi) if mse_packed_hi.dim() == 4 else [0, 0, 0, 0]
    smsepl = st(mse_packed_lo) if mse_packed_lo.dim() == 4 else [0, 0, 0, 0]
    sg = st(gamma)
    snk = st(norms_k)
    svp = st(val_idx_packed) if val_idx_packed.dim() == 4 else [0, 0, 0, 0]
    svph = st(val_idx_packed_hi) if val_idx_packed_hi.dim() == 4 else [0, 0, 0, 0]
    svpl = st(val_idx_packed_lo) if val_idx_packed_lo.dim() == 4 else [0, 0, 0, 0]
    snv = st(norms_v)
    spa = st(partials_acc)                    # [B,H_q,NS,D]
    spm = st(partials_m)                      # [B,H_q,NS]
    sor = st(out)                             # [B,H_q,D] canonical V-space

    S_bucket = _s_bucket(S_total)
    grid_split = (B, H_q, num_splits)
    _attn_split_kernel[grid_split](
        q_3d, s_k_per_q, pi_k_per_q,
        qjl_packed,
        mse_packed, mse_packed_hi, mse_packed_lo,
        gamma, norms_k,
        val_idx_packed, val_idx_packed_hi, val_idx_packed_lo,
        norms_v,
        key_cent_lut, key_cent_lut_hi, key_cent_lut_lo,
        val_cent_lut, val_cent_lut_hi, val_cent_lut_lo,
        partials_acc, partials_m, partials_l,
        sqd[0], sqd[1],
        ssk[0], ssk[1],
        spi[0], spi[1],
        spa[0], spa[1], spa[2],
        spm[0], spm[1],
        sqjlp[0], sqjlp[1], sqjlp[2],
        smsep[0], smsep[1], smsep[2],
        smseph[0], smseph[1], smseph[2],
        smsepl[0], smsepl[1], smsepl[2],
        sg[0], sg[1],
        snk[0], snk[1],
        svp[0], svp[1], svp[2],
        svph[0], svph[1], svph[2],
        svpl[0], svpl[1], svpl[2],
        snv[0], snv[1],
        s_total_tensor,
        S_bucket,
        float(scale),
        float(_QJL_C / D),
        num_splits,
        GQA_GROUPS=gqa_groups,
        D=D,
        MSE_BITS=mse_bits,
        VAL_BITS=value_bits,
        N_OUT=n_out,
        MSE_BITS_HI=mse_bits_hi,
        MSE_BITS_LO=mse_bits_lo,
        VAL_BITS_HI=val_bits_hi,
        VAL_BITS_LO=val_bits_lo,
        SPLIT_SIZE=split_size,
    )

    block_splits = max(2, _next_pow2(num_splits))
    grid_reduce = (B, H_q)
    _attn_reduce_kernel[grid_reduce](
        partials_acc, partials_m, partials_l,
        pi_v_per_q,
        out,
        spa[0], spa[1], spa[2],
        spm[0], spm[1],
        spv[0], spv[1],
        sor[0], sor[1],
        num_splits,
        D=D,
        BLOCK_SPLITS=block_splits,
    )

    return out.unsqueeze(2).to(in_dtype)
