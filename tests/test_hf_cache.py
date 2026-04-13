"""Tests for HuggingFace KV cache integration."""

import pytest
import torch

from turboquant.quantizer_prod_torch import TurboQuantProdTorch
from turboquant.integrations.hf_cache import (
    TurboQuantCache,
    TurboQuantCacheLayer,
    _BatchedKeyStore,
    _BatchedValueStore,
)
from turboquant.integrations.memory_tracker import (
    compressed_bytes,
    fp16_equivalent_bytes,
    report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockConfig:
    """Minimal mock of PretrainedConfig for testing."""

    def __init__(
        self,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        hidden_size: int = 256,
        head_dim: int = 64,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

    def get_text_config(self, decoder: bool = False):
        return self


@pytest.fixture
def small_config():
    return MockConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        head_dim=64,
    )


@pytest.fixture
def gqa_config():
    """Config with GQA: 8 attention heads, 2 KV heads."""
    return MockConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=256,
        head_dim=32,
    )


# ---------------------------------------------------------------------------
# Step 1 tests: TurboQuantProdTorch.quantize_with_norm
# ---------------------------------------------------------------------------


class TestProdQuantizeWithNorm:
    def test_single_vector_round_trip(self):
        q = TurboQuantProdTorch(d=64, b=4, seed=42)
        x = torch.randn(64)
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)
        assert x_hat.shape == (64,)
        assert norm.item() == pytest.approx(torch.linalg.norm(x).item(), rel=1e-5)

    def test_batch_round_trip(self):
        q = TurboQuantProdTorch(d=64, b=4, seed=42)
        x = torch.randn(10, 64)
        quantized, norms = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norms)
        assert x_hat.shape == (10, 64)
        assert norms.shape == (10,)
        expected_norms = torch.linalg.norm(x, dim=1)
        torch.testing.assert_close(norms, expected_norms, rtol=1e-5, atol=1e-6)

    def test_zero_vector(self):
        q = TurboQuantProdTorch(d=64, b=4, seed=42)
        x = torch.zeros(64)
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)
        assert x_hat.shape == (64,)
        assert norm.item() == 0.0

    def test_preserves_direction(self):
        q = TurboQuantProdTorch(d=128, b=4, seed=42)
        x = torch.randn(128) * 5.0
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)
        cosine_sim = torch.dot(x, x_hat) / (
            torch.linalg.norm(x) * torch.linalg.norm(x_hat)
        )
        assert cosine_sim.item() > 0.8


# ---------------------------------------------------------------------------
# Cache layer shape tests
# ---------------------------------------------------------------------------


class TestCacheLayerShapes:
    def test_single_token(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 1, 64)
        v = torch.randn(1, 2, 1, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 1, 64)
        assert all_v.shape == (1, 2, 1, 64)
        assert layer.get_seq_length() == 1

    def test_prefill(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 16, 64)
        v = torch.randn(1, 2, 16, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 16, 64)
        assert all_v.shape == (1, 2, 16, 64)
        assert layer.get_seq_length() == 16

    def test_incremental_decode(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        for step in range(5):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 5, 64)
        assert all_v.shape == (1, 2, 5, 64)
        assert layer.get_seq_length() == 5

    def test_prefill_then_decode(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        # Prefill 8 tokens
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        layer.update(k, v)
        # Decode 3 tokens
        for _ in range(3):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 11, 64)
        assert layer.get_seq_length() == 11

    def test_batch_size_2(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(2, 2, 4, 64)
        v = torch.randn(2, 2, 4, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (2, 2, 4, 64)

    def test_b1_pure_qjl(self):
        """b=1 means pure QJL with no MSE stage."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=1, value_bits=1, seed=42
        )
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 4, 64)


# ---------------------------------------------------------------------------
# Full cache tests
# ---------------------------------------------------------------------------


class TestTurboQuantCache:
    def test_creation(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4)
        assert len(cache.layers) == 2
        assert cache.get_seq_length() == 0

    def test_update_through_layers(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4, seed=42)
        # Simulate forward pass: update each layer
        for layer_idx in range(2):
            k = torch.randn(1, 2, 4, 64)
            v = torch.randn(1, 2, 4, 64)
            all_k, all_v = cache.update(k, v, layer_idx)
            assert all_k.shape == (1, 2, 4, 64)
            assert all_v.shape == (1, 2, 4, 64)
        assert cache.get_seq_length() == 4

    def test_incremental_generation(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4, seed=42)
        # Prefill
        for layer_idx in range(2):
            k = torch.randn(1, 2, 8, 64)
            v = torch.randn(1, 2, 8, 64)
            cache.update(k, v, layer_idx)
        # 3 decode steps
        for step in range(3):
            for layer_idx in range(2):
                k = torch.randn(1, 2, 1, 64)
                v = torch.randn(1, 2, 1, 64)
                all_k, all_v = cache.update(k, v, layer_idx)
        assert all_k.shape == (1, 2, 11, 64)
        assert cache.get_seq_length() == 11

    def test_gqa_config(self, gqa_config):
        cache = TurboQuantCache(gqa_config, key_bits=4, value_bits=4)
        for layer_idx in range(2):
            k = torch.randn(1, 2, 4, 32)  # 2 KV heads, dim=32
            v = torch.randn(1, 2, 4, 32)
            all_k, all_v = cache.update(k, v, layer_idx)
            assert all_k.shape == (1, 2, 4, 32)

    def test_per_layer_bits(self, small_config):
        per_layer = {0: {"key_bits": 3, "value_bits": 3}}
        cache = TurboQuantCache(
            small_config, key_bits=4, value_bits=4, per_layer_bits=per_layer
        )
        layer_0 = cache.layers[0]
        layer_1 = cache.layers[1]
        assert layer_0._key_bits == 3
        assert layer_0._value_bits == 3
        assert layer_1._key_bits == 4
        assert layer_1._value_bits == 4


# ---------------------------------------------------------------------------
# Reconstruction quality tests
# ---------------------------------------------------------------------------


class TestReconstructionQuality:
    # Paper Table 1: d * D_mse ≈ 0.36 (b=1), 0.117 (b=2), 0.03 (b=3), 0.009 (b=4)
    # For non-unit vectors through the cache, MSE scales with input variance.
    # Use relative MSE (normalized by input energy) with generous tolerance.

    @pytest.mark.parametrize("bits,max_rel_mse", [(2, 0.25), (3, 0.10), (4, 0.05)])
    def test_value_mse_bounded(self, bits, max_rel_mse):
        """Dequantized values should have bounded relative MSE vs original."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=1, head_dim=64,
            key_bits=bits, value_bits=bits, seed=42,
        )
        n_tokens = 50
        v_orig = torch.randn(1, 1, n_tokens, 64)
        k_dummy = torch.randn(1, 1, n_tokens, 64)
        _, all_v = layer.update(k_dummy, v_orig)

        mse = ((all_v - v_orig) ** 2).mean().item()
        energy = (v_orig ** 2).mean().item()
        rel_mse = mse / energy
        assert rel_mse < max_rel_mse, (
            f"Relative MSE too high at {bits} bits: {rel_mse:.4f} > {max_rel_mse}"
        )

    # Key Prod uses (b-1) MSE bits + 1-bit QJL, so higher reconstruction error than pure MSE
    @pytest.mark.parametrize("bits,max_rel_mse", [(2, 0.70), (3, 0.25), (4, 0.10)])
    def test_key_reconstruction_bounded(self, bits, max_rel_mse):
        """Dequantized keys should have bounded relative reconstruction error."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=1, head_dim=64,
            key_bits=bits, value_bits=bits, seed=42,
        )
        n_tokens = 50
        k_orig = torch.randn(1, 1, n_tokens, 64)
        v_dummy = torch.randn(1, 1, n_tokens, 64)
        all_k, _ = layer.update(k_orig, v_dummy)

        mse = ((all_k - k_orig) ** 2).mean().item()
        energy = (k_orig ** 2).mean().item()
        rel_mse = mse / energy
        assert rel_mse < max_rel_mse, (
            f"Key relative MSE too high at {bits} bits: {rel_mse:.4f} > {max_rel_mse}"
        )


# ---------------------------------------------------------------------------
# M2: Inner-product unbiasedness test (core paper property)
# ---------------------------------------------------------------------------


class TestKeyInnerProductUnbiasedness:
    """TurboQuantProd should preserve unbiased inner products through the cache.

    Paper property: E[<y, x_hat>] = <y, x> for keys quantized with Prod.
    """

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_unbiased_ip_through_cache(self, bits):
        d = 64
        n_seeds = 100
        torch.manual_seed(99)
        k_orig = torch.randn(1, 1, 1, d)
        y = torch.randn(d)
        true_ip = (k_orig[0, 0, 0] @ y).item()

        estimates = []
        for seed in range(n_seeds):
            layer = TurboQuantCacheLayer(
                layer_idx=0, num_kv_heads=1, head_dim=d,
                key_bits=bits, value_bits=bits, seed=seed,
            )
            v_dummy = torch.randn(1, 1, 1, d)
            all_k, _ = layer.update(k_orig, v_dummy)
            estimates.append((all_k[0, 0, 0] @ y).item())

        mean_estimate = sum(estimates) / len(estimates)
        assert abs(mean_estimate - true_ip) < 0.3 * abs(true_ip) + 0.1, (
            f"b={bits}: mean IP={mean_estimate:.4f}, true IP={true_ip:.4f}, "
            f"bias={abs(mean_estimate - true_ip):.4f}"
        )


# ---------------------------------------------------------------------------
# Beam search / reorder tests
# ---------------------------------------------------------------------------


class TestBeamSearch:
    def test_reorder_cache(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4, seed=42)
        # Build cache with batch=2
        for layer_idx in range(2):
            k = torch.randn(2, 2, 4, 64)
            v = torch.randn(2, 2, 4, 64)
            cache.update(k, v, layer_idx)

        # Reorder: swap batch items
        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)

        # After reorder, seq length unchanged
        assert cache.get_seq_length() == 4

        # Dequantize to verify shapes still valid
        for layer_idx in range(2):
            k = torch.randn(2, 2, 1, 64)
            v = torch.randn(2, 2, 1, 64)
            all_k, all_v = cache.update(k, v, layer_idx)
            assert all_k.shape == (2, 2, 5, 64)


# ---------------------------------------------------------------------------
# Crop tests
# ---------------------------------------------------------------------------


class TestCrop:
    def test_crop_layer(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        layer.update(k, v)
        assert layer.get_seq_length() == 10

        layer.crop(5)
        assert layer.get_seq_length() == 5

        # Continue from cropped state
        k = torch.randn(1, 2, 1, 64)
        v = torch.randn(1, 2, 1, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 6, 64)


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_layer(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 5, 64)
        v = torch.randn(1, 2, 5, 64)
        layer.update(k, v)
        assert layer.get_seq_length() == 5

        layer.reset()
        assert layer.get_seq_length() == 0

        # Can update after reset
        k = torch.randn(1, 2, 3, 64)
        v = torch.randn(1, 2, 3, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 3, 64)


# ---------------------------------------------------------------------------
# Memory tracker tests
# ---------------------------------------------------------------------------


class TestMemoryTracker:
    def test_empty_cache(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4)
        r = report(cache)
        assert r["compressed_bytes"] == 0
        assert r["compression_ratio"] == 1.0

    def test_compression_report(self, small_config):
        cache = TurboQuantCache(small_config, key_bits=4, value_bits=4, seed=42)
        for layer_idx in range(2):
            k = torch.randn(1, 2, 32, 64)
            v = torch.randn(1, 2, 32, 64)
            cache.update(k, v, layer_idx)

        r = report(cache)
        assert r["seq_length"] == 32
        assert r["compressed_bytes"] > 0
        assert r["fp16_bytes"] > 0
        assert r["compression_ratio"] > 1.0

    def test_fp16_equivalent(self):
        # 2 layers, 2 heads, dim=64, seq=32, batch=1
        fp16 = fp16_equivalent_bytes(2, 2, 64, 32, 1)
        # K + V = 2 * 1 * 2 * 32 * 64 * 2 bytes * 2 layers
        expected = 2 * 1 * 2 * 32 * 64 * 2 * 2
        assert fp16 == expected


# ---------------------------------------------------------------------------
# Compression ratio tests (verify bit packing achieves theoretical savings)
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    """After bit packing, compression ratios must be substantially better than 1×."""

    @pytest.fixture
    def two_layer_config(self):
        return MockConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=128,
            head_dim=64,
        )

    def _fill_cache(self, config, bits, seq=64):
        cache = TurboQuantCache(config, key_bits=bits, value_bits=bits, seed=42)
        for layer_idx in range(config.num_hidden_layers):
            k = torch.randn(1, config.num_key_value_heads, seq, config.head_dim)
            v = torch.randn(1, config.num_key_value_heads, seq, config.head_dim)
            cache.update(k, v, layer_idx)
        return cache

    @pytest.mark.parametrize("bits,min_ratio", [
        (4, 3.0),   # theoretical 3.05× for head_dim=64; allow 3.0×
        (3, 3.5),   # theoretical 3.76× for head_dim=64 (3-bit wastes 2 bits/byte)
        (2, 5.5),   # theoretical 5.82× for head_dim=64
    ])
    def test_compression_ratio_by_bits(self, two_layer_config, bits, min_ratio):
        cache = self._fill_cache(two_layer_config, bits)
        r = report(cache)
        assert r["compression_ratio"] >= min_ratio, (
            f"b={bits}: got {r['compression_ratio']:.2f}×, "
            f"expected >= {min_ratio}×"
        )

    @pytest.mark.parametrize("bits", [4, 3, 2])
    def test_different_bits_different_sizes(self, two_layer_config, bits):
        """Lower bit widths must use less memory than higher bit widths."""
        cache_high = self._fill_cache(two_layer_config, bits=4)
        cache_low = self._fill_cache(two_layer_config, bits=bits)
        r_high = report(cache_high)
        r_low = report(cache_low)
        assert r_low["compressed_bytes"] <= r_high["compressed_bytes"], (
            f"b={bits} should use <= bytes as b=4"
        )


# ---------------------------------------------------------------------------
# Mask sizes
# ---------------------------------------------------------------------------


class TestMaskSizes:
    def test_initial(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        kv_len, kv_off = layer.get_mask_sizes(torch.tensor([0, 1, 2, 3]))
        assert kv_len == 4  # query_length + 0 past
        assert kv_off == 0

    def test_after_cache(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        layer.update(k, v)

        kv_len, kv_off = layer.get_mask_sizes(torch.tensor([8]))
        assert kv_len == 9  # 1 new + 8 cached
        assert kv_off == 0

    def test_accepts_int_query_length(self):
        """HF's Cache.get_mask_sizes passes a plain int, not a tensor."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        layer.update(k, v)

        kv_len, kv_off = layer.get_mask_sizes(1)
        assert kv_len == 9
        assert kv_off == 0


# ---------------------------------------------------------------------------
# Dtype handling (fp16/bf16 inputs from real models)
# ---------------------------------------------------------------------------


class TestDtypeHandling:
    """Real models run in fp16/bf16; quantizer math must stay fp32 internally
    but round-trip back to the model's dtype at the cache boundary."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_update_preserves_input_dtype(self, dtype):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 8, 64, dtype=dtype)
        v = torch.randn(1, 2, 8, 64, dtype=dtype)
        all_k, all_v = layer.update(k, v)
        assert all_k.dtype == dtype
        assert all_v.dtype == dtype
        assert all_k.shape == (1, 2, 8, 64)

    def test_incremental_fp16(self):
        """Regression: rotation matmul failed on fp16 inputs (Half vs float)."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 8, 64, dtype=torch.float16)
        layer.update(k, v)
        k_new = torch.randn(1, 2, 1, 64, dtype=torch.float16)
        v_new = torch.randn(1, 2, 1, 64, dtype=torch.float16)
        all_k, all_v = layer.update(k_new, v_new)
        assert all_k.dtype == torch.float16
        assert all_k.shape == (1, 2, 9, 64)

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"), reason="FP8 not available in this PyTorch"
    )
    def test_fp8_guard_returns_fp16(self):
        """FP8 input must return FP16 — vanilla PyTorch attention can't consume FP8."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 4, 64).to(torch.float8_e4m3fn)
        v = torch.randn(1, 2, 4, 64).to(torch.float8_e4m3fn)
        all_k, all_v = layer.update(k, v)
        assert all_k.dtype == torch.float16
        assert all_v.dtype == torch.float16
        assert all_k.shape == (1, 2, 4, 64)


# ---------------------------------------------------------------------------
# M3: batch_repeat_interleave and batch_select_indices tests
# ---------------------------------------------------------------------------


class TestBatchRepeatInterleave:
    def test_repeat_doubles_batch(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        layer.update(k, v)

        layer.batch_repeat_interleave(3)

        # Continue with batch=3, verify shapes
        k = torch.randn(3, 2, 1, 64)
        v = torch.randn(3, 2, 1, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (3, 2, 5, 64)
        assert all_v.shape == (3, 2, 5, 64)


class TestBatchSelectIndices:
    def test_select_subset(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(3, 2, 4, 64)
        v = torch.randn(3, 2, 4, 64)
        layer.update(k, v)

        # Select only batch items 0 and 2
        layer.batch_select_indices(torch.tensor([0, 2]))

        # Continue with batch=2
        k = torch.randn(2, 2, 1, 64)
        v = torch.randn(2, 2, 1, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (2, 2, 5, 64)


# ---------------------------------------------------------------------------
# Negative crop test
# ---------------------------------------------------------------------------


class TestNegativeCrop:
    def test_crop_negative(self):
        """Negative crop removes that many tokens from the end."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        layer.update(k, v)

        layer.crop(-3)  # Remove last 3 tokens
        assert layer.get_seq_length() == 7

        k = torch.randn(1, 2, 1, 64)
        v = torch.randn(1, 2, 1, 64)
        all_k, _ = layer.update(k, v)
        assert all_k.shape == (1, 2, 8, 64)


# ---------------------------------------------------------------------------
# Seed determinism test
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    def test_same_seed_same_output(self):
        """Same seed and input should produce identical quantized output."""
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)

        layer1 = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        all_k1, all_v1 = layer1.update(k, v)

        layer2 = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        all_k2, all_v2 = layer2.update(k, v)

        torch.testing.assert_close(all_k1, all_k2)
        torch.testing.assert_close(all_v1, all_v2)

    def test_different_seed_different_output(self):
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)

        layer1 = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        all_k1, _ = layer1.update(k, v)

        layer2 = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=99
        )
        all_k2, _ = layer2.update(k, v)

        assert not torch.allclose(all_k1, all_k2)


# ---------------------------------------------------------------------------
# Different key_bits vs value_bits test
# ---------------------------------------------------------------------------


class TestMixedBitwidths:
    def test_different_key_value_bits(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64,
            key_bits=2, value_bits=4, seed=42,
        )
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        all_k, all_v = layer.update(k, v)
        assert all_k.shape == (1, 2, 8, 64)
        assert all_v.shape == (1, 2, 8, 64)

        # Value (4-bit) should have lower MSE than key (2-bit Prod)
        k_mse = ((all_k - k) ** 2).mean().item()
        v_mse = ((all_v - v) ** 2).mean().item()
        assert v_mse < k_mse, (
            f"Expected value MSE ({v_mse:.4f}) < key MSE ({k_mse:.4f}) "
            "since values use 4 bits vs keys 2 bits"
        )


# ---------------------------------------------------------------------------
# Key/value seed independence test (verifies M1 fix)
# ---------------------------------------------------------------------------


class TestSeedIndependence:
    def test_key_value_different_rotation(self):
        """Key and value quantizers should use different rotation matrices."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=1, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        # Batched quantizers hold stacked [H, D, D] matrices; compare head 0.
        # Key quantizer (Prod) internal MSE rotation, head 0
        key_rot = layer._key_quantizer.mse.rotation.Pi[0]
        # Value quantizer (MSE) rotation, head 0
        val_rot = layer._value_quantizer.rotation.Pi[0]
        assert not torch.allclose(key_rot, val_rot), (
            "Key and value quantizers should use independent rotation matrices"
        )


# ---------------------------------------------------------------------------
# L4: CUDA device tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestCUDADevice:
    def test_cache_layer_on_cuda(self):
        device = "cuda"
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64,
            key_bits=4, value_bits=4, seed=42, device=device,
        )
        k = torch.randn(1, 2, 4, 64, device=device)
        v = torch.randn(1, 2, 4, 64, device=device)
        all_k, all_v = layer.update(k, v)
        assert all_k.device.type == "cuda"
        assert all_v.device.type == "cuda"
        assert all_k.shape == (1, 2, 4, 64)

    def test_incremental_on_cuda(self):
        device = "cuda"
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64,
            key_bits=4, value_bits=4, seed=42, device=device,
        )
        # Prefill
        k = torch.randn(1, 2, 8, 64, device=device)
        v = torch.randn(1, 2, 8, 64, device=device)
        layer.update(k, v)

        # Decode
        for _ in range(3):
            k = torch.randn(1, 2, 1, 64, device=device)
            v = torch.randn(1, 2, 1, 64, device=device)
            all_k, all_v = layer.update(k, v)

        assert all_k.shape == (1, 2, 11, 64)
        assert all_k.device.type == "cuda"

    def test_full_cache_on_cuda(self, small_config):
        device = "cuda"
        cache = TurboQuantCache(
            small_config, key_bits=4, value_bits=4, seed=42, device=device
        )
        for layer_idx in range(2):
            k = torch.randn(1, 2, 4, 64, device=device)
            v = torch.randn(1, 2, 4, 64, device=device)
            all_k, all_v = cache.update(k, v, layer_idx)
            assert all_k.device.type == "cuda"
        assert cache.get_seq_length() == 4


# ---------------------------------------------------------------------------
# Incremental dense cache correctness (Phase 10)
# ---------------------------------------------------------------------------


class TestIncrementalDense:
    """Verify incremental dequant matches full-dequant from packed store.

    The incremental path dequantizes new tokens directly from the raw
    quantized dict (before pack/unpack). The reference path unpacks from the
    compressed store.  Both must yield identical float32 tensors because
    pack/unpack is lossless for valid indices.
    """

    def _run_and_collect(self, layer, steps, B=1, H=2, D=64):
        """Run N updates, return the final dense output."""
        # Prefill
        k0 = torch.randn(B, H, 8, D)
        v0 = torch.randn(B, H, 8, D)
        all_k, all_v = layer.update(k0, v0)

        # Decode steps
        for _ in range(steps):
            k = torch.randn(B, H, 1, D)
            v = torch.randn(B, H, 1, D)
            all_k, all_v = layer.update(k, v)

        return all_k, all_v

    def test_incremental_matches_full_dequant(self):
        """Dense buffer output must match full _dequantize_all_keys / _values."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        N_DECODE = 5
        all_k, all_v = self._run_and_collect(layer, N_DECODE)

        B, H, D = 1, 2, 64
        ref_k = layer._dequantize_all_keys(B, H, D).to(all_k.dtype)
        ref_v = layer._dequantize_all_values(B, H, D).to(all_v.dtype)

        # Incremental path dequantizes in chunks (one per update step) while
        # the reference dequantizes the full sequence in one call.  The same
        # quantized integers produce identical results in exact arithmetic, but
        # floating-point matmul accumulates in different order → tiny fp32
        # rounding differences (empirically ≤ 1e-6 absolute).  Use a tight
        # but non-zero tolerance to verify they are numerically equivalent.
        torch.testing.assert_close(all_k, ref_k, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(all_v, ref_v, rtol=1e-4, atol=1e-5)

    def test_shape_grows_correctly(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        all_k, _ = layer.update(k, v)
        assert all_k.shape == (1, 2, 8, 64)

        for step in range(4):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            all_k, all_v = layer.update(k, v)
            expected_seq = 8 + step + 1
            assert all_k.shape == (1, 2, expected_seq, 64)
            assert all_v.shape == (1, 2, expected_seq, 64)

    def test_crop_resets_dense_buffer(self):
        """Crop must truncate dense buffer to match compressed store."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        layer.update(k, v)

        layer.crop(6)
        assert layer._dense_keys.shape == (1, 2, 6, 64)
        assert layer._dense_values.shape == (1, 2, 6, 64)

        # Can still append after crop
        k = torch.randn(1, 2, 1, 64)
        v = torch.randn(1, 2, 1, 64)
        all_k, _ = layer.update(k, v)
        assert all_k.shape == (1, 2, 7, 64)

    def test_reset_clears_dense_buffer(self):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        layer.update(k, v)
        assert layer._dense_keys is not None

        layer.reset()
        assert layer._dense_keys is None
        assert layer._dense_values is None

    def test_batch_repeat_interleave_dense(self):
        """batch_repeat_interleave must expand dense buffer batch dim."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        layer.update(k, v)
        layer.batch_repeat_interleave(3)
        assert layer._dense_keys.shape[0] == 3
        assert layer._dense_values.shape[0] == 3

    def test_batch_select_indices_dense(self):
        """batch_select_indices must select from dense buffer batch dim."""
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64, key_bits=4, value_bits=4, seed=42
        )
        k = torch.randn(3, 2, 4, 64)
        v = torch.randn(3, 2, 4, 64)
        layer.update(k, v)
        layer.batch_select_indices(torch.tensor([0, 2]))
        assert layer._dense_keys.shape[0] == 2
        assert layer._dense_values.shape[0] == 2

    def test_memory_tracker_still_shows_compression(self):
        """Compressed store must still be populated so memory_tracker works."""
        from turboquant.integrations.memory_tracker import report

        config = MockConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=128,
            head_dim=64,
        )
        cache = TurboQuantCache(config, key_bits=4, value_bits=4, seed=42)
        for i in range(2):
            k = torch.randn(1, 2, 32, 64)
            v = torch.randn(1, 2, 32, 64)
            cache.update(k, v, i)
        r = report(cache)
        assert r["compression_ratio"] >= 3.0, (
            f"Expected >= 3.0× compression, got {r['compression_ratio']:.2f}×"
        )


# ---------------------------------------------------------------------------
# Reconstruction quality tests (Phase 10)
# ---------------------------------------------------------------------------


class TestReconstructionQuality:
    """Verify that quantized KV cache reconstruction is reasonable.

    Uses cosine similarity between original and dequantized vectors.
    Higher bit-widths should yield higher cosine similarity.
    b>=3 is the recommended minimum; b=2 documents the quality floor.
    """

    def _cosine_sim(self, layer, key_bits, n=200, D=64):
        """Average cosine similarity over n random unit vectors."""
        x = torch.randn(1, 2, n, D)
        all_k, _ = layer.update(x, x)  # Use same tensor for K and V simplicity
        # Flatten to [n, D] for both original and reconstructed
        orig = x.reshape(n * 2, D)
        recon = all_k.reshape(n * 2, D)
        cos = torch.nn.functional.cosine_similarity(orig, recon, dim=1)
        return cos.mean().item()

    @pytest.mark.parametrize("bits,min_cos", [
        (4, 0.95),  # 4-bit: very high cosine similarity
        (3, 0.85),  # 3-bit: good but lower
        (2, 0.40),  # 2-bit: quality limit — document as minimum viable
    ])
    def test_cosine_similarity_by_bits(self, bits, min_cos):
        layer = TurboQuantCacheLayer(
            layer_idx=0, num_kv_heads=2, head_dim=64,
            key_bits=bits, value_bits=bits, seed=42,
        )
        cos_sim = self._cosine_sim(layer, bits)
        assert cos_sim >= min_cos, (
            f"b={bits}: cosine similarity {cos_sim:.3f} < {min_cos}. "
            f"{'b=2 is near the quality floor; b>=3 recommended.' if bits == 2 else ''}"
        )

    def test_higher_bits_higher_quality(self):
        """4-bit must reconstruct better than 3-bit, which beats 2-bit."""
        results = {}
        for bits in [2, 3, 4]:
            layer = TurboQuantCacheLayer(
                layer_idx=0, num_kv_heads=2, head_dim=64,
                key_bits=bits, value_bits=bits, seed=42,
            )
            results[bits] = self._cosine_sim(layer, bits)

        assert results[4] > results[3] > results[2], (
            f"Expected b=4 > b=3 > b=2, got {results}"
        )
