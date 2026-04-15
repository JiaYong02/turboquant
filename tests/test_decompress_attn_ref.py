"""Tests for the PyTorch fused decompress+attention reference (Phase 9a).

The reference must match `F.scaled_dot_product_attention` run on the
fully-dequantized K/V (which is what the existing TurboQuantCache returns to
HF's stock attention). The point is to verify the algebraic reordering — Q·Pi
folding and rotated-V accumulation — produces the same numbers.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from turboquant.integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from turboquant.kernels import fused_decompress_attention_ref
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def _build_layer_state(B, H_kv, S, D, key_bits, value_bits, seed=0, device="cpu"):
    """Quantize random K/V into the same packed stores the cache uses."""
    torch.manual_seed(seed)
    k = torch.randn(B, H_kv, S, D, device=device)
    v = torch.randn(B, H_kv, S, D, device=device)

    seed_base = 12345
    mse_seeds = [seed_base + h for h in range(H_kv)]
    qjl_seeds = [seed_base + 100 + h for h in range(H_kv)]
    val_seeds = [seed_base + 500 + h for h in range(H_kv)]

    kq = BatchedTurboQuantProdTorch(D, key_bits, H_kv, mse_seeds, qjl_seeds, device)
    vq = BatchedTurboQuantMSETorch(D, value_bits, H_kv, val_seeds, device)

    k_batched = k.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    v_batched = v.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)

    q_k, norms_k = kq.quantize_with_norm(k_batched)
    q_v, norms_v = vq.quantize_with_norm(v_batched)

    ks = _BatchedKeyStore(key_bits=key_bits, head_dim=D)
    vs = _BatchedValueStore(value_bits=value_bits, head_dim=D)
    ks.append(q_k, norms_k, B, S)
    vs.append(q_v, norms_v, B, S)

    # Reconstruct the dense K/V the cache would return to vanilla SDPA.
    k_deq = kq.dequantize_with_norm(q_k, norms_k).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)
    v_deq = vq.dequantize_with_norm(q_v, norms_v).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)

    return ks, vs, kq, vq, k_deq, v_deq


def _sdpa_decode(q, k_deq, v_deq, gqa_groups):
    """Reference: stock SDPA on dense dequantized K/V (S_q = 1)."""
    H_kv = k_deq.shape[1]
    if gqa_groups > 1:
        k_deq = k_deq.repeat_interleave(gqa_groups, dim=1)
        v_deq = v_deq.repeat_interleave(gqa_groups, dim=1)
    return F.scaled_dot_product_attention(q, k_deq, v_deq)


def _cosine_sim(a, b):
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    return torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)


@pytest.mark.parametrize("key_bits,value_bits", [(4, 4), (3, 3), (2, 2), (4, 3)])
@pytest.mark.parametrize("B,H_kv,gqa,S,D", [(1, 4, 1, 64, 32), (2, 4, 2, 256, 64)])
def test_ref_matches_dense_sdpa(B, H_kv, gqa, S, D, key_bits, value_bits):
    """Folded-rotation algebra must match running SDPA on dequantized K/V."""
    H_q = H_kv * gqa
    ks, vs, kq, vq, k_deq, v_deq = _build_layer_state(
        B, H_kv, S, D, key_bits, value_bits, seed=42
    )

    q = torch.randn(B, H_q, 1, D)

    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
    out_sdpa = _sdpa_decode(q, k_deq, v_deq, gqa)

    cos = _cosine_sim(out_ref, out_sdpa).item()
    assert cos >= 0.9999, f"cosine similarity {cos:.6f} below 0.9999"
    # Tight numerical tolerance — only fp32 rounding differences should remain.
    torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("block_n", [16, 64, 128, 1024])
def test_ref_block_n_invariance(block_n):
    """Streaming softmax must be tile-size invariant."""
    B, H_kv, gqa, S, D = 1, 4, 2, 200, 32
    H_q = H_kv * gqa
    ks, vs, kq, vq, _, _ = _build_layer_state(B, H_kv, S, D, 4, 4, seed=7)
    q = torch.randn(B, H_q, 1, D)

    out_a = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=block_n)
    out_b = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=S)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_ref_rejects_prefill_shape():
    B, H_kv, S, D = 1, 2, 8, 16
    ks, vs, kq, vq, _, _ = _build_layer_state(B, H_kv, S, D, 4, 4)
    q = torch.randn(B, H_kv, 4, D)  # S_q=4 (prefill) — must be rejected
    with pytest.raises(ValueError, match=r"\[B,H_q,1,D\]"):
        fused_decompress_attention_ref(q, ks, vs, kq, vq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ref_cuda():
    B, H_kv, gqa, S, D = 2, 8, 4, 1024, 128
    H_q = H_kv * gqa
    device = "cuda"
    ks, vs, kq, vq, k_deq, v_deq = _build_layer_state(
        B, H_kv, S, D, 4, 4, seed=1, device=device
    )
    q = torch.randn(B, H_q, 1, D, device=device)
    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=128)
    out_sdpa = _sdpa_decode(q, k_deq, v_deq, gqa)
    cos = _cosine_sim(out_ref, out_sdpa).item()
    assert cos >= 0.9999, f"cosine similarity {cos:.6f} below 0.9999"
