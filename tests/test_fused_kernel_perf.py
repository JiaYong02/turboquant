"""Perf gate for the fused decompress+attention kernel (Phase 11).

Gated by ``TURBOQUANT_PERF=1`` to keep CI cheap. Asserts that the Triton
fused-decode path (with cached per-Q-head rotation state) is within a small
multiple of FP16 SDPA latency at the target decode shape.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
perf = pytest.mark.skipif(
    os.environ.get("TURBOQUANT_PERF") != "1",
    reason="set TURBOQUANT_PERF=1 to run perf gate",
)
graph_gated = pytest.mark.skipif(
    os.environ.get("TURBOQUANT_CUDA_GRAPH") != "1",
    reason="set TURBOQUANT_CUDA_GRAPH=1 to run graph perf gate",
)
triton = pytest.importorskip("triton")

from turboquant.integrations.hf_cache import (  # noqa: E402
    TurboQuantCache,
    _BatchedKeyStore,
    _BatchedValueStore,
)
from turboquant.kernels.decompress_attn import (  # noqa: E402
    fused_decompress_attention_triton,
)
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch  # noqa: E402
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch  # noqa: E402


def _time_ms(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@cuda
@perf
def test_fused_within_budget_vs_sdpa() -> None:
    """Fused decode ≤ 1.2× FP16 SDPA at B=1, H_q=32, H_kv=8, S=1024, D=128, b=4."""
    device = "cuda"
    B, H_q, H_kv, S, D = 1, 32, 8, 1024, 128
    bits = 4
    gqa = H_q // H_kv

    # Build compressed store.
    torch.manual_seed(0)
    k = torch.randn(B, H_kv, S, D, device=device)
    v = torch.randn(B, H_kv, S, D, device=device)

    mse_seeds = [1000 + h for h in range(H_kv)]
    qjl_seeds = [2000 + h for h in range(H_kv)]
    val_seeds = [3000 + h for h in range(H_kv)]
    kq = BatchedTurboQuantProdTorch(D, bits, H_kv, mse_seeds, qjl_seeds, device)
    vq = BatchedTurboQuantMSETorch(D, bits, H_kv, val_seeds, device)

    kb = k.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    vb = v.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    q_k, nk = kq.quantize_with_norm(kb)
    q_v, nv = vq.quantize_with_norm(vb)

    ks = _BatchedKeyStore(key_bits=bits, head_dim=D)
    vs = _BatchedValueStore(value_bits=bits, head_dim=D)
    ks.append(q_k, nk, B, S)
    vs.append(q_v, nv, B, S)

    # Precompute per-Q-head rotations (simulates _prepare_fused_state).
    kv_for_q = torch.arange(H_q, device=device) // gqa
    S_k = kq.qjl.S.to(device=device, dtype=torch.float32).index_select(
        0, kv_for_q
    ).contiguous()
    Pi_k = kq.mse.rotation.Pi.to(device=device, dtype=torch.float32).index_select(
        0, kv_for_q
    ).contiguous()
    Pi_v = vq.rotation.Pi.to(device=device, dtype=torch.float32).index_select(
        0, kv_for_q
    ).contiguous()

    q_dec = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float16)

    # FP16 SDPA baseline — use dense dequant K/V to match shapes.
    k_dense = kq.dequantize_with_norm(q_k, nk).reshape(H_kv, B, S, D).permute(
        1, 0, 2, 3
    )
    v_dense = vq.dequantize_with_norm(q_v, nv).reshape(H_kv, B, S, D).permute(
        1, 0, 2, 3
    )
    k_dense_q = k_dense.repeat_interleave(gqa, dim=1).to(torch.float16)
    v_dense_q = v_dense.repeat_interleave(gqa, dim=1).to(torch.float16)
    scale = 1.0 / math.sqrt(D)

    def run_sdpa() -> None:
        torch.nn.functional.scaled_dot_product_attention(
            q_dec, k_dense_q, v_dense_q, scale=scale
        )

    def run_fused() -> None:
        fused_decompress_attention_triton(
            q_dec, ks, vs, kq, vq,
            scale=scale,
            pi_k_per_q=Pi_k, s_k_per_q=S_k, pi_v_per_q=Pi_v,
            kv_for_q=kv_for_q,
        )

    sdpa_ms = _time_ms(run_sdpa)
    fused_ms = _time_ms(run_fused)

    ratio = fused_ms / sdpa_ms
    print(f"\nFP16 SDPA: {sdpa_ms * 1000:.1f}μs | Fused: {fused_ms * 1000:.1f}μs | "
          f"ratio: {ratio:.2f}×")
    assert ratio <= 1.2, (
        f"fused decode {fused_ms * 1000:.1f}μs exceeds 1.2× FP16 SDPA "
        f"({sdpa_ms * 1000:.1f}μs), ratio={ratio:.2f}"
    )


class _MockConfig:
    def __init__(self, num_layers=1, num_kv_heads=8, num_attention_heads=32, head_dim=128):
        self.num_hidden_layers = num_layers
        self.num_key_value_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim

    def get_text_config(self, decoder: bool = True):
        return self


@cuda
@perf
@graph_gated
def test_graph_within_budget_vs_sdpa() -> None:
    """Fused+CUDA-graph decode ≤ 3.0× FP16 SDPA at B=1, H_q=32, H_kv=8, S=1024, D=128.

    Graph capture absorbs CPU dispatch (Phase 11b delivers ~5× eager speedup).
    Residual vs SDPA is irreducible Triton kernel-launch latency; closing it
    requires persistent-kernel / rotation-fusion work in a follow-up phase.
    This gate detects regressions without blocking on that target.
    """
    device = "cuda"
    B, H_q, H_kv, S, D = 1, 32, 8, 1024, 128
    bits = 4
    gqa = H_q // H_kv

    torch.manual_seed(0)
    k = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)

    config = _MockConfig(num_layers=1, num_kv_heads=H_kv, num_attention_heads=H_q, head_dim=D)
    cache = TurboQuantCache(
        config,
        key_bits=bits,
        value_bits=bits,
        device=device,
        use_fused_kernel=True,
        max_seq_len=2048,
        use_cuda_graph=True,
    )
    layer = cache.layers[0]
    layer.update(k, v)

    scale = 1.0 / math.sqrt(D)
    q_dec = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float16)

    # FP16 SDPA baseline on dense dequant KV (same reference as the eager gate).
    kq = layer._key_quantizer
    vq = layer._value_quantizer
    q_k, nk = layer._key_store.to_quantized_dict()
    q_v, nv = layer._value_store.to_quantized_dict()
    k_dense = kq.dequantize_with_norm(q_k, nk).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)
    v_dense = vq.dequantize_with_norm(q_v, nv).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)
    k_dense_q = k_dense.repeat_interleave(gqa, dim=1).to(torch.float16)
    v_dense_q = v_dense.repeat_interleave(gqa, dim=1).to(torch.float16)

    # Warm up the graph capture so timing excludes one-shot capture cost.
    layer._prepare_fused_state(q_dec.device, H_q)
    with torch.inference_mode():
        _ = layer._fused_graph_forward(q_dec, scale)

    def run_sdpa() -> None:
        torch.nn.functional.scaled_dot_product_attention(
            q_dec, k_dense_q, v_dense_q, scale=scale
        )

    def run_graph() -> None:
        with torch.inference_mode():
            layer._fused_graph_forward(q_dec, scale)

    sdpa_ms = _time_ms(run_sdpa)
    graph_ms = _time_ms(run_graph)

    ratio = graph_ms / sdpa_ms
    print(
        f"\nFP16 SDPA: {sdpa_ms * 1000:.1f}μs | Fused+Graph: {graph_ms * 1000:.1f}μs | "
        f"ratio: {ratio:.2f}×"
    )
    assert ratio <= 3.0, (
        f"graph decode {graph_ms * 1000:.1f}μs exceeds 3.0× FP16 SDPA "
        f"({sdpa_ms * 1000:.1f}μs), ratio={ratio:.2f}"
    )
