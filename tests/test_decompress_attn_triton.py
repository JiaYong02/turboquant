"""Tests for the Triton fused decompress+attention kernel (Phase 9b).

Uses the Phase 9a PyTorch reference as ground truth — same algebra, so any
divergence means a bug in the Triton kernel (bit-unpack, stride math, or
streaming softmax) rather than a math issue.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

triton = pytest.importorskip("triton")

from turboquant.integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore  # noqa: E402
from turboquant.kernels.decompress_attn import (  # noqa: E402
    fused_decompress_attention_triton,
)
from turboquant.kernels.decompress_attn_ref import (  # noqa: E402
    fused_decompress_attention_ref,
)
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch  # noqa: E402
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch  # noqa: E402


def _build_state(B, H_kv, S, D, key_bits, value_bits, seed=0, device="cuda"):
    torch.manual_seed(seed)
    k = torch.randn(B, H_kv, S, D, device=device)
    v = torch.randn(B, H_kv, S, D, device=device)

    mse_seeds = [1000 + h for h in range(H_kv)]
    qjl_seeds = [2000 + h for h in range(H_kv)]
    val_seeds = [3000 + h for h in range(H_kv)]

    kq = BatchedTurboQuantProdTorch(D, key_bits, H_kv, mse_seeds, qjl_seeds, device)
    vq = BatchedTurboQuantMSETorch(D, value_bits, H_kv, val_seeds, device)

    kb = k.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    vb = v.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    q_k, nk = kq.quantize_with_norm(kb)
    q_v, nv = vq.quantize_with_norm(vb)

    ks = _BatchedKeyStore(key_bits=key_bits, head_dim=D)
    vs = _BatchedValueStore(value_bits=value_bits, head_dim=D)
    ks.append(q_k, nk, B, S)
    vs.append(q_v, nv, B, S)
    return ks, vs, kq, vq


def _cos(a, b):
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-12)).item()


@cuda
@pytest.mark.parametrize("key_bits,value_bits", [(4, 4), (3, 3), (2, 2), (4, 3)])
@pytest.mark.parametrize("B,H_kv,gqa,S,D", [(1, 4, 1, 128, 32), (2, 4, 2, 256, 64)])
def test_triton_matches_reference(B, H_kv, gqa, S, D, key_bits, value_bits):
    H_q = H_kv * gqa
    ks, vs, kq, vq = _build_state(B, H_kv, S, D, key_bits, value_bits)
    q = torch.randn(B, H_q, 1, D, device="cuda")

    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
    out_tri = fused_decompress_attention_triton(q, ks, vs, kq, vq, block_n=64)

    cos = _cos(out_ref, out_tri)
    assert cos >= 0.999, f"cosine sim {cos:.6f} < 0.999"
    torch.testing.assert_close(out_tri, out_ref, atol=1e-3, rtol=1e-3)


@cuda
@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_triton_block_n_invariance(block_n):
    B, H_kv, gqa, S, D = 1, 4, 2, 200, 64
    H_q = H_kv * gqa
    ks, vs, kq, vq = _build_state(B, H_kv, S, D, 4, 4, seed=7)
    q = torch.randn(B, H_q, 1, D, device="cuda")
    out_tri = fused_decompress_attention_triton(q, ks, vs, kq, vq, block_n=block_n)
    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=S)
    torch.testing.assert_close(out_tri, out_ref, atol=1e-3, rtol=1e-3)


@cuda
def test_triton_llama_shape():
    """Shape and sizes representative of Llama-3.1-8B per-layer KV cache."""
    B, H_kv, gqa, S, D = 1, 8, 4, 512, 128
    H_q = H_kv * gqa  # 32 query heads, 8 KV heads
    ks, vs, kq, vq = _build_state(B, H_kv, S, D, 4, 4)
    q = torch.randn(B, H_q, 1, D, device="cuda")
    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq)
    out_tri = fused_decompress_attention_triton(q, ks, vs, kq, vq, block_n=64)
    cos = _cos(out_ref, out_tri)
    assert cos >= 0.999, f"cosine sim {cos:.6f} < 0.999 at Llama-ish shape"
