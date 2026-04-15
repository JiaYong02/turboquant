"""Phase 9c integration test for the fused-kernel attention hook.

Exercises the full decode path: TurboQuantCache.update() seeds the compressed
store, bind_cache() makes the cache visible to the hook, then a call mimicking
HF's attention interface routes through the fused Triton kernel. Reference is
the drop-in (dense dequant + SDPA) path that ships in Phase 8.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _MockConfig:
    def __init__(self, num_layers=2, num_kv_heads=4, num_attention_heads=8, head_dim=64):
        self.num_hidden_layers = num_layers
        self.num_key_value_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim

    def get_text_config(self, decoder=True):
        return self


class _MockAttnModule:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


@cuda
def test_fused_attention_hook_matches_sdpa_reference():
    from turboquant.integrations import TurboQuantCache
    from turboquant.integrations.attention import turboquant_fused_attention_forward

    torch.manual_seed(0)
    device = "cuda"
    B, H_q, H_kv, D, S_prefill = 1, 8, 4, 64, 128

    config = _MockConfig(num_layers=2, num_kv_heads=H_kv, num_attention_heads=H_q, head_dim=D)
    cache = TurboQuantCache(config, key_bits=4, value_bits=4, device=device, use_fused_kernel=True)

    # Prefill: fills compressed store + dense buffers for both layers.
    for layer_idx in range(2):
        k_pref = torch.randn(B, H_kv, S_prefill, D, device=device, dtype=torch.float16)
        v_pref = torch.randn(B, H_kv, S_prefill, D, device=device, dtype=torch.float16)
        cache.layers[layer_idx].update(k_pref, v_pref)

    # Decode step: single new token.
    layer_idx = 0
    layer = cache.layers[layer_idx]
    k_new = torch.randn(B, H_kv, 1, D, device=device, dtype=torch.float16)
    v_new = torch.randn(B, H_kv, 1, D, device=device, dtype=torch.float16)
    layer.update(k_new, v_new)

    q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float16)

    # Reference: stock SDPA against fully dequantized KV.
    k_ref = layer._dequantize_all_keys(B, H_kv, D).to(torch.float16)
    v_ref = layer._dequantize_all_values(B, H_kv, D).to(torch.float16)
    gqa = H_q // H_kv
    k_ref_rep = k_ref.repeat_interleave(gqa, dim=1)
    v_ref_rep = v_ref.repeat_interleave(gqa, dim=1)
    out_ref = F.scaled_dot_product_attention(q, k_ref_rep, v_ref_rep)  # [B,H_q,1,D]

    # Hook path via bind_cache + mock attn module.
    module = _MockAttnModule(layer_idx=layer_idx)
    with cache.bind():
        out_hook, _ = turboquant_fused_attention_forward(
            module, q, k_ref_rep, v_ref_rep, attention_mask=None,
        )
    # Hook returns [B, S_q, H_q, D]; transpose back for comparison.
    out_hook = out_hook.transpose(1, 2)

    cos = F.cosine_similarity(
        out_hook.float().reshape(-1), out_ref.float().reshape(-1), dim=0
    ).item()
    assert cos >= 0.999, f"cosine similarity {cos:.6f} < 0.999"


@cuda
def test_fused_cache_prefill_falls_back_to_dense():
    """Without bind() active, hook must fall back to SDPA (prefill compatible)."""
    from turboquant.integrations import TurboQuantCache
    from turboquant.integrations.attention import turboquant_fused_attention_forward

    torch.manual_seed(1)
    device = "cuda"
    B, H_q, H_kv, D, S = 1, 4, 2, 32, 16
    config = _MockConfig(num_layers=1, num_kv_heads=H_kv, num_attention_heads=H_q, head_dim=D)
    cache = TurboQuantCache(config, key_bits=4, value_bits=4, device=device, use_fused_kernel=True)

    q = torch.randn(B, H_q, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)
    gqa = H_q // H_kv
    k_rep = k.repeat_interleave(gqa, dim=1)
    v_rep = v.repeat_interleave(gqa, dim=1)

    module = _MockAttnModule(layer_idx=0)
    # No bind → fallback. S_q > 1 anyway → fallback.
    out, _ = turboquant_fused_attention_forward(
        module, q, k_rep, v_rep, attention_mask=None, is_causal=False,
    )
    out_ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False).transpose(1, 2)
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)
