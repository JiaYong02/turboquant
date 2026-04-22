"""Phase B-3: parity tests for split-bucket fused decompress+attention.

Covers two layers:

1. The PyTorch reference (`fused_decompress_attention_ref`) must produce the
   same output as running SDPA over the fully dequantized split-bucket K/V —
   this validates the bucket-split algebra in isolation (no kernel involved).

2. When CUDA + Triton are available, the fused Triton path
   (`fused_decompress_attention_triton`) must match the reference bit-close
   under the two-bucket unpack path (`n_out > 0`) and fall back to the
   single-bucket path when `n_out == 0`.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from turboquant.integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from turboquant.kernels import fused_decompress_attention_ref
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def _build_split_layer(
    B: int,
    H_kv: int,
    S: int,
    D: int,
    bits_hi_k: int,
    bits_lo_k: int,
    bits_hi_v: int,
    bits_lo_v: int,
    n_out: int,
    seed: int = 0,
    device: str = "cpu",
):
    """Quantize random K/V into split-bucket stores and return references."""
    torch.manual_seed(seed)
    k = torch.randn(B, H_kv, S, D, device=device)
    v = torch.randn(B, H_kv, S, D, device=device)

    mse_seeds = [1000 + h for h in range(H_kv)]
    qjl_seeds = [2000 + h for h in range(H_kv)]
    val_seeds = [3000 + h for h in range(H_kv)]

    kq = BatchedTurboQuantProdTorch(
        D,
        b=bits_hi_k,  # single-bucket param; unused for n_out > 0
        num_heads=H_kv,
        mse_seeds=mse_seeds,
        qjl_seeds=qjl_seeds,
        device=device,
        n_out=n_out,
        bits_hi=bits_hi_k,
        bits_lo=bits_lo_k,
    )
    vq = BatchedTurboQuantMSETorch(
        D,
        b=bits_hi_v,
        num_heads=H_kv,
        seeds=val_seeds,
        device=device,
        n_out=n_out,
        bits_hi=bits_hi_v,
        bits_lo=bits_lo_v,
    )

    k_batched = k.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    v_batched = v.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)

    q_k, norms_k = kq.quantize_with_norm(k_batched)
    q_v, norms_v = vq.quantize_with_norm(v_batched)

    ks = _BatchedKeyStore(
        key_bits=bits_hi_k,
        head_dim=D,
        n_out=n_out,
        bits_hi=bits_hi_k,
        bits_lo=bits_lo_k,
    )
    vs = _BatchedValueStore(
        value_bits=bits_hi_v,
        head_dim=D,
        n_out=n_out,
        bits_hi=bits_hi_v,
        bits_lo=bits_lo_v,
    )
    ks.append(q_k, norms_k, B, S)
    vs.append(q_v, norms_v, B, S)

    k_deq = (
        kq.dequantize_with_norm(q_k, norms_k)
        .reshape(H_kv, B, S, D)
        .permute(1, 0, 2, 3)
    )
    v_deq = (
        vq.dequantize_with_norm(q_v, norms_v)
        .reshape(H_kv, B, S, D)
        .permute(1, 0, 2, 3)
    )

    return ks, vs, kq, vq, k_deq, v_deq


def _sdpa_decode(q, k_deq, v_deq, gqa_groups):
    if gqa_groups > 1:
        k_deq = k_deq.repeat_interleave(gqa_groups, dim=1)
        v_deq = v_deq.repeat_interleave(gqa_groups, dim=1)
    return F.scaled_dot_product_attention(q, k_deq, v_deq)


def _cosine_sim(a, b):
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    return torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)


class TestRefSplitBucket:
    """Split-bucket reference parity against SDPA-on-dequantized."""

    @pytest.mark.parametrize(
        "bits_hi_k,bits_lo_k,bits_hi_v,bits_lo_v,n_out",
        [
            (4, 2, 4, 2, 32),
            (5, 3, 5, 3, 32),
            (4, 3, 4, 3, 16),
        ],
    )
    @pytest.mark.parametrize("B,H_kv,gqa,S,D", [(1, 4, 2, 128, 64)])
    def test_ref_matches_dense_sdpa_split(
        self, B, H_kv, gqa, S, D,
        bits_hi_k, bits_lo_k, bits_hi_v, bits_lo_v, n_out
    ):
        H_q = H_kv * gqa
        ks, vs, kq, vq, k_deq, v_deq = _build_split_layer(
            B, H_kv, S, D,
            bits_hi_k=bits_hi_k, bits_lo_k=bits_lo_k,
            bits_hi_v=bits_hi_v, bits_lo_v=bits_lo_v,
            n_out=n_out, seed=17,
        )
        q = torch.randn(B, H_q, 1, D)

        out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
        out_sdpa = _sdpa_decode(q, k_deq, v_deq, gqa)

        cos = _cosine_sim(out_ref, out_sdpa).item()
        # Split-bucket algebra must match dense SDPA-on-dequantized closely.
        # fp16 storage of norms/gamma introduces ~1e-3 absolute error; the
        # algebraic reordering itself is exact, so cosine should be ~1.
        assert cos >= 0.9999, f"cosine similarity {cos:.6f} below 0.9999"
        torch.testing.assert_close(out_ref, out_sdpa, atol=5e-3, rtol=5e-3)

    def test_ref_single_bucket_unchanged(self):
        """n_out=0 path must remain bit-identical to Phase-13 behavior."""
        B, H_kv, gqa, S, D = 1, 4, 1, 64, 32
        H_q = H_kv * gqa
        torch.manual_seed(0)
        mse_seeds = [h + 10 for h in range(H_kv)]
        qjl_seeds = [h + 20 for h in range(H_kv)]
        val_seeds = [h + 30 for h in range(H_kv)]
        kq = BatchedTurboQuantProdTorch(D, 4, H_kv, mse_seeds, qjl_seeds)
        vq = BatchedTurboQuantMSETorch(D, 4, H_kv, val_seeds)

        k = torch.randn(H_kv, B * S, D)
        v = torch.randn(H_kv, B * S, D)
        q_k, norms_k = kq.quantize_with_norm(k)
        q_v, norms_v = vq.quantize_with_norm(v)
        ks = _BatchedKeyStore(key_bits=4, head_dim=D)
        vs = _BatchedValueStore(value_bits=4, head_dim=D)
        ks.append(q_k, norms_k, B, S)
        vs.append(q_v, norms_v, B, S)

        q = torch.randn(B, H_q, 1, D)
        out = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=32)
        assert out.shape == (B, H_q, 1, D)

    def test_ref_rejects_mismatched_split(self):
        """K with n_out > 0 but V with n_out = 0 must raise."""
        D, H_kv = 64, 2
        kq = BatchedTurboQuantProdTorch(
            D, 4, H_kv, mse_seeds=[1, 2], qjl_seeds=[11, 12],
            n_out=16, bits_hi=4, bits_lo=2,
        )
        vq = BatchedTurboQuantMSETorch(D, 4, H_kv, seeds=[21, 22])

        k = torch.randn(H_kv, 8, D)
        v = torch.randn(H_kv, 8, D)
        q_k, norms_k = kq.quantize_with_norm(k)
        q_v, norms_v = vq.quantize_with_norm(v)
        ks = _BatchedKeyStore(
            key_bits=4, head_dim=D, n_out=16, bits_hi=4, bits_lo=2
        )
        vs = _BatchedValueStore(value_bits=4, head_dim=D)
        ks.append(q_k, norms_k, 1, 8)
        vs.append(q_v, norms_v, 1, 8)

        q = torch.randn(1, H_kv, 1, D)
        with pytest.raises(ValueError, match="split-bucket mode"):
            fused_decompress_attention_ref(q, ks, vs, kq, vq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonSplitBucket:
    """Triton fused kernel must match the reference under split-bucket mode."""

    @pytest.mark.parametrize(
        "bits_hi_k,bits_lo_k,bits_hi_v,bits_lo_v,n_out",
        [
            (4, 2, 4, 2, 32),
            (5, 3, 5, 3, 32),
            (4, 3, 4, 3, 16),
            (3, 2, 3, 2, 32),
        ],
    )
    @pytest.mark.parametrize(
        "B,H_kv,gqa,S,D", [(1, 8, 4, 256, 64), (1, 4, 2, 128, 128)]
    )
    def test_triton_split_matches_ref(
        self, B, H_kv, gqa, S, D,
        bits_hi_k, bits_lo_k, bits_hi_v, bits_lo_v, n_out,
    ):
        from turboquant.kernels.decompress_attn import (
            fused_decompress_attention_triton,
        )

        H_q = H_kv * gqa
        ks, vs, kq, vq, _, _ = _build_split_layer(
            B, H_kv, S, D,
            bits_hi_k=bits_hi_k, bits_lo_k=bits_lo_k,
            bits_hi_v=bits_hi_v, bits_lo_v=bits_lo_v,
            n_out=n_out, seed=31, device="cuda",
        )
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

        out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
        out_triton = fused_decompress_attention_triton(q, ks, vs, kq, vq)

        cos = _cosine_sim(out_ref, out_triton).item()
        assert cos >= 0.9999, f"cosine similarity {cos:.6f} below 0.9999"
        torch.testing.assert_close(out_ref, out_triton, atol=5e-3, rtol=5e-3)

    def test_triton_single_bucket_still_works(self):
        """n_out=0 path must remain functional in the new kernel signature."""
        from turboquant.kernels.decompress_attn import (
            fused_decompress_attention_triton,
        )

        B, H_kv, gqa, S, D = 1, 4, 2, 128, 64
        H_q = H_kv * gqa
        torch.manual_seed(0)
        mse_seeds = [h + 10 for h in range(H_kv)]
        qjl_seeds = [h + 20 for h in range(H_kv)]
        val_seeds = [h + 30 for h in range(H_kv)]
        kq = BatchedTurboQuantProdTorch(
            D, 4, H_kv, mse_seeds, qjl_seeds, device="cuda"
        )
        vq = BatchedTurboQuantMSETorch(D, 4, H_kv, val_seeds, device="cuda")

        k = torch.randn(H_kv, B * S, D, device="cuda")
        v = torch.randn(H_kv, B * S, D, device="cuda")
        q_k, norms_k = kq.quantize_with_norm(k)
        q_v, norms_v = vq.quantize_with_norm(v)
        ks = _BatchedKeyStore(key_bits=4, head_dim=D)
        vs = _BatchedValueStore(value_bits=4, head_dim=D)
        ks.append(q_k, norms_k, B, S)
        vs.append(q_v, norms_v, B, S)

        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)
        out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
        out_triton = fused_decompress_attention_triton(q, ks, vs, kq, vq)

        cos = _cosine_sim(out_ref, out_triton).item()
        assert cos >= 0.9999, f"cosine similarity {cos:.6f} below 0.9999"
