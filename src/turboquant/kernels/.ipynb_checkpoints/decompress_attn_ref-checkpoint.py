"""PyTorch reference for fused decompress+attention (Phase 9a).

Computes single-query (decode) attention directly from the compressed K/V stores
held by ``TurboQuantCacheLayer`` without materialising a dense [B,H,S,D] K/V
tensor. The math is reordered so that the per-(layer,head) rotation matrices
``Pi_k``, ``S_k``, ``Pi_v`` are folded into Q (once per step) and into the
output (once per step), instead of being applied per stored token.

For a key vector reconstructed as

    k_j = norm_j * ( Pi_k @ centroid[mse_idx_j]
                     + (sqrt(pi/2)/D) * gamma_j * S_k @ signs_j )

the dot product with a query Q becomes

    Q . k_j = norm_j * ( <Q_rot, centroid[mse_idx_j]>
                         + c * gamma_j * <Q_qjl, signs_j> )

with Q_rot = Q @ Pi_k.T  and  Q_qjl = Q @ S_k.T  computed once per head.

For values reconstructed as v_j = norm_v_j * Pi_v @ centroid_v[idx_v_j], the
output is

    out = ( sum_j (attn_j * norm_v_j) * centroid_v[idx_v_j] ) @ Pi_v

so accumulation happens in the rotated V-space and Pi_v is applied once at the
end. This matches what the Triton kernel (Phase 9b) will do per program.

The reference uses a streaming-softmax loop over KV tiles to mirror the
FlashDecoding structure that Phase 9b will follow.
"""

from __future__ import annotations

import math

import torch

from ..integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from ..quantizer_mse_torch import BatchedTurboQuantMSETorch
from ..quantizer_prod_torch import BatchedTurboQuantProdTorch

_QJL_C = math.sqrt(math.pi / 2)  # divided by D inside the kernel


def _decode_key_centroids(
    key_quantizer: BatchedTurboQuantProdTorch,
) -> torch.Tensor:
    """Centroid LUT for the (b-1)-bit MSE stage of the key Prod quantizer.

    Returns a 1-D float32 tensor of length 2**(b-1) (or empty for b=1).
    """
    if key_quantizer.mse is None:
        return torch.empty(0, dtype=torch.float32)
    return key_quantizer.mse.codebook._centroids.to(torch.float32)


def _decode_value_centroids(
    value_quantizer: BatchedTurboQuantMSETorch,
) -> torch.Tensor:
    """Centroid LUT for the b-bit MSE value quantizer."""
    return value_quantizer.codebook._centroids.to(torch.float32)


def fused_decompress_attention_ref(
    q: torch.Tensor,
    key_store: _BatchedKeyStore,
    value_store: _BatchedValueStore,
    key_quantizer: BatchedTurboQuantProdTorch,
    value_quantizer: BatchedTurboQuantMSETorch,
    scale: float | None = None,
    block_n: int = 128,
) -> torch.Tensor:
    """Fused decompress+attention reference for a single decode step.

    Args:
        q: Query tensor of shape ``[B, H_q, 1, D]`` in any float dtype.
        key_store: Compressed key storage for one layer (shape ``[H_kv,B,S,...]``).
        value_store: Compressed value storage for one layer.
        key_quantizer: The ``BatchedTurboQuantProdTorch`` used to write
            ``key_store``. Provides ``Pi_k`` (mse.rotation.Pi), ``S_k``
            (qjl.S), and the MSE codebook.
        value_quantizer: The ``BatchedTurboQuantMSETorch`` used to write
            ``value_store``. Provides ``Pi_v`` and the MSE codebook.
        scale: Softmax scale (default ``1/sqrt(D)``).
        block_n: KV-tile size for the streaming softmax.

    Returns:
        Output tensor of shape ``[B, H_q, 1, D]`` in the input dtype.

    Notes:
        - GQA is supported: query head ``hq`` reads from KV head
          ``hq // (H_q // H_kv)``.
        - The function accepts only ``S_q=1`` (decode); prefill must use the
          dense path.
    """
    if q.dim() != 4 or q.shape[2] != 1:
        raise ValueError(
            f"fused_decompress_attention_ref expects q shape [B,H_q,1,D], "
            f"got {tuple(q.shape)}"
        )

    B, H_q, _, D = q.shape
    H_kv = key_quantizer.num_heads
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q={H_q} must be divisible by H_kv={H_kv} for GQA"
        )
    gqa_groups = H_q // H_kv

    if key_store.qjl_packed is None or value_store.idx_packed is None:
        raise ValueError("KV store is empty; nothing to attend over")

    S_total = key_store.seq_length()
    if S_total == 0:
        return torch.zeros_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    in_dtype = q.dtype
    device = q.device

    # ---- 1. Unpack the entire compressed store once ---------------------------
    # Reference impl unpacks up-front; the Triton kernel will load packed tiles
    # from HBM and unpack in registers.
    q_k_dict, norms_k_flat = key_store.to_quantized_dict()
    q_v_dict, norms_v_flat = value_store.to_quantized_dict()

    # Reshape from [H_kv, B*S, D] back to [H_kv, B, S, D] (or [H_kv,B,S] for
    # scalar tensors) so we can index per-(b, hkv, s).
    qjl = q_k_dict["qjl"].reshape(H_kv, B, S_total, D).to(torch.float32)
    gamma = q_k_dict["gamma"].reshape(H_kv, B, S_total).to(torch.float32)
    norms_k = norms_k_flat.reshape(H_kv, B, S_total).to(torch.float32)

    if q_k_dict["mse"] is not None:
        key_mse_idx = q_k_dict["mse"]["idx"].reshape(H_kv, B, S_total, D)
    else:
        key_mse_idx = None

    val_idx = q_v_dict["idx"].reshape(H_kv, B, S_total, D)
    norms_v = norms_v_flat.reshape(H_kv, B, S_total).to(torch.float32)

    # ---- 2. Per-head rotation / projection matrices ---------------------------
    Pi_k = key_quantizer.mse.rotation.Pi.to(device=device, dtype=torch.float32) \
        if key_quantizer.mse is not None else None  # [H_kv, D, D]
    S_k = key_quantizer.qjl.S.to(device=device, dtype=torch.float32)  # [H_kv,D,D]
    Pi_v = value_quantizer.rotation.Pi.to(device=device, dtype=torch.float32)

    key_centroids = _decode_key_centroids(key_quantizer).to(device)
    val_centroids = _decode_value_centroids(value_quantizer).to(device)

    qjl_const = _QJL_C / D

    # ---- 3. Fold rotations into Q (per head) ---------------------------------
    # Q has shape [B, H_q, 1, D]. Each query head hq maps to KV head hkv.
    q_f32 = q.to(torch.float32).squeeze(2)  # [B, H_q, D]

    # Map per-query-head to per-KV-head matrices via index_select.
    kv_for_q = torch.arange(H_q, device=device) // gqa_groups  # [H_q]
    Pi_k_per_q = Pi_k.index_select(0, kv_for_q) if Pi_k is not None else None
    S_k_per_q = S_k.index_select(0, kv_for_q)
    Pi_v_per_q = Pi_v.index_select(0, kv_for_q)

    # Q_rot[b,hq,:] = q[b,hq,:] @ Pi_k[hkv].T  (Pi_k applied to the KEY, so for Q
    # we use Pi_k.T to align the dot product).
    if Pi_k_per_q is not None:
        Q_rot = torch.einsum("bhd,hed->bhe", q_f32, Pi_k_per_q)  # [B, H_q, D]
    else:
        Q_rot = None
    Q_qjl = torch.einsum("bhd,hed->bhe", q_f32, S_k_per_q)  # [B, H_q, D]

    # ---- 4. Streaming softmax over KV tiles -----------------------------------
    out_rot = torch.zeros(B, H_q, D, dtype=torch.float32, device=device)
    m_run = torch.full((B, H_q), float("-inf"), dtype=torch.float32, device=device)
    l_run = torch.zeros(B, H_q, dtype=torch.float32, device=device)

    for s_start in range(0, S_total, block_n):
        s_end = min(s_start + block_n, S_total)
        Sn = s_end - s_start

        # Slices over the S dim for this tile, broadcast to per-query-head via
        # index_select on the KV-head dim.
        # qjl_tile: [H_q, B, Sn, D]
        qjl_tile = qjl[:, :, s_start:s_end, :].index_select(0, kv_for_q)
        gamma_tile = gamma[:, :, s_start:s_end].index_select(0, kv_for_q)
        norms_k_tile = norms_k[:, :, s_start:s_end].index_select(0, kv_for_q)
        val_idx_tile = val_idx[:, :, s_start:s_end, :].index_select(0, kv_for_q)
        norms_v_tile = norms_v[:, :, s_start:s_end].index_select(0, kv_for_q)

        # MSE centroid tile: [H_q, B, Sn, D]
        if key_mse_idx is not None:
            key_mse_tile = key_mse_idx[:, :, s_start:s_end, :].index_select(
                0, kv_for_q
            )
            key_cent_tile = key_centroids[key_mse_tile.long()]  # gather
        else:
            key_cent_tile = None

        # ---- logits = norm_k * ( <Q_rot, centroid> + c*gamma * <Q_qjl, signs> )
        # Q_rot: [B, H_q, D]; key_cent_tile: [H_q, B, Sn, D]
        # Want logits: [B, H_q, Sn]
        if key_cent_tile is not None:
            # einsum over D
            mse_dot = torch.einsum(
                "bhd,hbsd->bhs", Q_rot, key_cent_tile
            )  # [B, H_q, Sn]
        else:
            mse_dot = torch.zeros(B, H_q, Sn, dtype=torch.float32, device=device)

        qjl_dot = torch.einsum("bhd,hbsd->bhs", Q_qjl, qjl_tile)  # [B, H_q, Sn]

        # gamma_tile / norms_k_tile: [H_q, B, Sn] -> [B, H_q, Sn]
        gamma_bhs = gamma_tile.permute(1, 0, 2)
        norms_k_bhs = norms_k_tile.permute(1, 0, 2)

        logits = (
            norms_k_bhs * (mse_dot + qjl_const * gamma_bhs * qjl_dot)
        ) * scale  # [B, H_q, Sn]

        # ---- streaming softmax update -----------------------------------------
        m_new = torch.maximum(m_run, logits.amax(dim=-1))  # [B, H_q]
        alpha = torch.exp(m_run - m_new)                   # rescale prev
        p = torch.exp(logits - m_new.unsqueeze(-1))        # [B, H_q, Sn]

        l_run = l_run * alpha + p.sum(dim=-1)

        # ---- value accumulation in rotated V-space ----------------------------
        # weights = p * norms_v_tile -> [B, H_q, Sn]
        norms_v_bhs = norms_v_tile.permute(1, 0, 2)
        weights = p * norms_v_bhs

        # val_centroid_tile: [H_q, B, Sn, D] via gather
        val_cent_tile = val_centroids[val_idx_tile.long()]
        # contrib = sum_s weights[b,hq,s] * val_cent_tile[hq,b,s,d]
        contrib = torch.einsum("bhs,hbsd->bhd", weights, val_cent_tile)

        out_rot = out_rot * alpha.unsqueeze(-1) + contrib
        m_run = m_new

    # ---- 5. Normalise + apply Pi_v --------------------------------------------
    out_rot = out_rot / l_run.unsqueeze(-1)  # [B, H_q, D]
    out = torch.einsum("bhd,hde->bhe", out_rot, Pi_v_per_q)  # [B, H_q, D]

    return out.unsqueeze(2).to(in_dtype)
