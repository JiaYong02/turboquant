"""Phase 12 parity: fused Q/Pi_v rotations vs PyTorch reference.

The reference (`fused_decompress_attention_ref`) computes the same math but
keeps Q and Pi_v rotations as separate einsums. This sweep exercises the
Triton kernel with rotations folded into the split/reduce kernels across
the shape space we care about at decode time.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
triton = pytest.importorskip("triton")

from turboquant.integrations.hf_cache import (  # noqa: E402
    _BatchedKeyStore,
    _BatchedValueStore,
)
from turboquant.kernels.decompress_attn import (  # noqa: E402
    fused_decompress_attention_triton,
)
from turboquant.kernels.decompress_attn_ref import (  # noqa: E402
    fused_decompress_attention_ref,
)
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch  # noqa: E402
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch  # noqa: E402


def _build(B, H_kv, S, D, key_bits, value_bits, seed=0, device="cuda"):
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


@cuda
@pytest.mark.parametrize(
    "B,H_q,H_kv,S,D,bits",
    [
        (1, 8, 2, 128, 64, 4),
        (1, 8, 4, 1024, 64, 3),
        (1, 32, 8, 1024, 128, 4),
        (1, 32, 8, 4096, 128, 4),
        (1, 32, 8, 1024, 128, 2),
        (2, 8, 4, 256, 64, 4),
    ],
)
def test_rotation_fusion_matches_reference(B, H_q, H_kv, S, D, bits):
    assert H_q % H_kv == 0
    ks, vs, kq, vq = _build(B, H_kv, S, D, bits, bits, seed=B * 100 + H_q + S)
    q = torch.randn(B, H_q, 1, D, device="cuda")
    out_ref = fused_decompress_attention_ref(q, ks, vs, kq, vq, block_n=64)
    out_tri = fused_decompress_attention_triton(q, ks, vs, kq, vq, block_n=64)
    torch.testing.assert_close(out_tri, out_ref, atol=1e-3, rtol=1e-3)


@cuda
def test_rotation_fusion_graph_replay_determinism():
    """Same input → same output across repeated calls (non-graph, stable state)."""
    B, H_q, H_kv, S, D = 1, 32, 8, 1024, 128
    ks, vs, kq, vq = _build(B, H_kv, S, D, 4, 4, seed=17)
    q = torch.randn(B, H_q, 1, D, device="cuda")
    out0 = fused_decompress_attention_triton(q, ks, vs, kq, vq)
    for _ in range(15):
        out_i = fused_decompress_attention_triton(q, ks, vs, kq, vq)
        torch.testing.assert_close(out_i, out0, atol=0.0, rtol=0.0)
