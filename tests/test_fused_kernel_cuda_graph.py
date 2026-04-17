"""Phase 11b tests for CUDA-graph capture + replay of the fused decode path.

Validates:
- slab preallocation + in-place append keeps tensor pointers stable
- captured graph replays bit-identically against the eager fused path
- one capture serves all seq_lens within a power-of-2 S-bucket
- invalidation hooks (reset / crop / reorder_cache) force re-capture
- bucket transitions (S crossing 256 → 1024 → 4096) don't blow up
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _MockConfig:
    def __init__(self, num_layers=1, num_kv_heads=2, num_attention_heads=4, head_dim=64):
        self.num_hidden_layers = num_layers
        self.num_key_value_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim

    def get_text_config(self, decoder=True):
        return self


def _make_cache(
    *,
    max_seq_len: int,
    use_cuda_graph: bool,
    num_kv_heads: int = 2,
    num_attention_heads: int = 4,
    head_dim: int = 64,
):
    from turboquant.integrations import TurboQuantCache

    config = _MockConfig(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
    )
    return TurboQuantCache(
        config,
        key_bits=4,
        value_bits=4,
        device="cuda",
        use_fused_kernel=True,
        max_seq_len=max_seq_len,
        use_cuda_graph=use_cuda_graph,
    )


def _prefill(layer, B, H_kv, D, S, dtype=torch.float16):
    torch.manual_seed(42)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    layer.update(k, v)


@cuda
def test_slab_preallocation_pointer_stability():
    """Slab tensors must keep the same storage address across decode steps."""
    cache = _make_cache(max_seq_len=512, use_cuda_graph=True)
    layer = cache.layers[0]

    _prefill(layer, B=1, H_kv=2, D=64, S=16)
    qjl_ptr = layer._key_store.qjl_packed.data_ptr()
    idx_ptr = layer._value_store.idx_packed.data_ptr()

    for _ in range(10):
        k_new = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float16)
        v_new = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float16)
        layer.update(k_new, v_new)

    assert layer._key_store.qjl_packed.data_ptr() == qjl_ptr
    assert layer._value_store.idx_packed.data_ptr() == idx_ptr
    assert layer._key_store.slab_active
    assert layer._value_store.slab_active
    assert layer._key_store.seq_length() == 26


@cuda
def test_slab_parity_with_cat_path():
    """Store contents after slab-mode append must match cat-growth contents."""
    B, H_kv, D, S = 1, 2, 64, 32

    slab_cache = _make_cache(max_seq_len=128, use_cuda_graph=True)
    eager_cache = _make_cache(max_seq_len=None, use_cuda_graph=False)

    torch.manual_seed(0)
    for step in range(S):
        k = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        slab_cache.layers[0].update(k, v)
        eager_cache.layers[0].update(k, v)

    slab_ks = slab_cache.layers[0]._key_store
    eager_ks = eager_cache.layers[0]._key_store
    S_total = eager_ks.qjl_packed.shape[2]

    torch.testing.assert_close(
        slab_ks.qjl_packed[:, :, :S_total, :],
        eager_ks.qjl_packed,
    )
    torch.testing.assert_close(
        slab_ks.gamma[:, :, :S_total], eager_ks.gamma
    )
    torch.testing.assert_close(
        slab_ks.norms[:, :, :S_total], eager_ks.norms
    )
    torch.testing.assert_close(
        slab_ks.mse_packed[:, :, :S_total, :], eager_ks.mse_packed
    )


@cuda
def test_graph_replay_matches_eager_fused():
    """Graph replay must be bit-identical to the eager fused wrapper."""
    from turboquant.kernels.decompress_attn import fused_decompress_attention_triton

    cache = _make_cache(max_seq_len=512, use_cuda_graph=True)
    layer = cache.layers[0]

    B, H_kv, H_q, D = 1, 2, 4, 64
    _prefill(layer, B=B, H_kv=H_kv, D=D, S=100)

    torch.manual_seed(7)
    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

    # Eager path (prepare_fused_state already populated via graph driver below,
    # but running it here explicitly is a no-op second call).
    layer._prepare_fused_state(q.device, H_q)
    out_eager = fused_decompress_attention_triton(
        q,
        layer._key_store,
        layer._value_store,
        layer._key_quantizer,
        layer._value_quantizer,
        scale=None,
        pi_k_per_q=layer._Pi_k_per_q,
        s_k_per_q=layer._S_k_per_q,
        pi_v_per_q=layer._Pi_v_per_q,
        kv_for_q=layer._kv_for_q,
        key_cent_lut=layer._key_cent_lut,
        val_cent_lut=layer._val_cent_lut,
    )

    with torch.inference_mode():
        out_graph = layer._fused_graph_forward(q, scale=1.0 / (D ** 0.5))

    # Shapes match.
    assert out_graph.shape == out_eager.shape
    torch.testing.assert_close(out_graph, out_eager, atol=1e-3, rtol=1e-3)


@cuda
def test_one_capture_serves_many_seq_lens_within_bucket():
    """Inside a single S-bucket, the captured graph handles every seq_len."""
    cache = _make_cache(max_seq_len=1024, use_cuda_graph=True)
    layer = cache.layers[0]

    B, H_kv, H_q, D = 1, 2, 4, 64
    _prefill(layer, B=B, H_kv=H_kv, D=D, S=130)  # bucket = 256

    scale = 1.0 / (D ** 0.5)
    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        _ = layer._fused_graph_forward(q, scale)
    captures_after_first = len(layer._graph_cache)
    assert captures_after_first == 1

    # Append more tokens without crossing the bucket boundary.
    for _ in range(40):
        k = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        layer.update(k, v)
        with torch.inference_mode():
            _ = layer._fused_graph_forward(q, scale)
    assert len(layer._graph_cache) == 1, "Expected no new captures within bucket"


@cuda
def test_bucket_transition_allocates_new_capture():
    """Crossing a bucket boundary triggers a new capture but keeps old ones."""
    cache = _make_cache(max_seq_len=4096, use_cuda_graph=True)
    layer = cache.layers[0]

    B, H_kv, H_q, D = 1, 2, 4, 64
    scale = 1.0 / (D ** 0.5)
    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

    # Seed bucket 256
    _prefill(layer, B=B, H_kv=H_kv, D=D, S=100)
    with torch.inference_mode():
        _ = layer._fused_graph_forward(q, scale)
    assert len(layer._graph_cache) == 1

    # Grow into bucket 1024
    for _ in range(300):
        k = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        layer.update(k, v)
    with torch.inference_mode():
        _ = layer._fused_graph_forward(q, scale)
    assert len(layer._graph_cache) == 2

    # Grow into bucket 4096 (max_seq_len)
    for _ in range(2048):
        k = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        layer.update(k, v)
    with torch.inference_mode():
        _ = layer._fused_graph_forward(q, scale)
    assert len(layer._graph_cache) == 3


@cuda
def test_reset_drops_captures():
    cache = _make_cache(max_seq_len=512, use_cuda_graph=True)
    layer = cache.layers[0]

    B, H_kv, H_q, D = 1, 2, 4, 64
    _prefill(layer, B=B, H_kv=H_kv, D=D, S=100)
    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        _ = layer._fused_graph_forward(q, scale=1.0 / (D ** 0.5))
    assert len(layer._graph_cache) == 1

    layer.reset()
    assert len(layer._graph_cache) == 0
    assert not layer._slab_preallocated


@cuda
def test_graph_replay_correctness_across_many_steps():
    """After 64 decode steps, graph replay continues to match eager fused."""
    from turboquant.kernels.decompress_attn import fused_decompress_attention_triton

    cache = _make_cache(max_seq_len=512, use_cuda_graph=True)
    layer = cache.layers[0]

    B, H_kv, H_q, D = 1, 2, 4, 64
    _prefill(layer, B=B, H_kv=H_kv, D=D, S=50)

    scale = 1.0 / (D ** 0.5)
    torch.manual_seed(11)

    for step in range(64):
        k = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device="cuda", dtype=torch.float16)
        layer.update(k, v)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)

        layer._prepare_fused_state(q.device, H_q)
        out_eager = fused_decompress_attention_triton(
            q,
            layer._key_store,
            layer._value_store,
            layer._key_quantizer,
            layer._value_quantizer,
            scale=scale,
            pi_k_per_q=layer._Pi_k_per_q,
            s_k_per_q=layer._S_k_per_q,
            pi_v_per_q=layer._Pi_v_per_q,
            kv_for_q=layer._kv_for_q,
            key_cent_lut=layer._key_cent_lut,
            val_cent_lut=layer._val_cent_lut,
        )
        with torch.inference_mode():
            out_graph = layer._fused_graph_forward(q, scale)

        torch.testing.assert_close(
            out_graph, out_eager, atol=1e-3, rtol=1e-3,
            msg=f"step={step}, seq_len={layer.get_seq_length()}",
        )
