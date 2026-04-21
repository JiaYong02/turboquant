"""Phase B-2: parity tests for split-bucket stores in `hf_cache`.

Exercises `_BatchedKeyStore` and `_BatchedValueStore` in split-bucket mode
(``n_out > 0``) via the round-trip through the quantizers. The kernel is
untouched at this phase; the non-fused Python dequant path must work.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from turboquant.integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def _rand(h: int, n: int, d: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(h, n, d, generator=gen, dtype=torch.float32)


class TestKeyStoreSingleBucketParity:
    def test_default_store_unchanged(self) -> None:
        H, D, b = 2, 128, 4
        store = _BatchedKeyStore(key_bits=b, head_dim=D)
        quant = BatchedTurboQuantProdTorch(
            D, b, H, mse_seeds=[1, 2], qjl_seeds=[11, 12]
        )
        x = _rand(H, 8, D, seed=1)
        q, norms = quant.quantize_with_norm(x)
        store.append(q, norms, B=1, S_new=8)
        assert store.mse_packed is not None
        assert not hasattr(store, "mse_packed_hi") or store.mse_packed_hi is None
        q_out, norms_out = store.to_quantized_dict()
        assert set(q_out["mse"].keys()) == {"idx"}


class TestKeyStoreSplitBucket:
    @pytest.mark.parametrize(
        "D,n_out,bits_hi,bits_lo",
        [
            (128, 32, 4, 2),
            (128, 32, 5, 3),
        ],
    )
    def test_roundtrip(
        self, D: int, n_out: int, bits_hi: int, bits_lo: int
    ) -> None:
        H, N, B, S = 2, 8, 1, 8
        store = _BatchedKeyStore(
            key_bits=bits_hi,  # single-bucket fallback, unused when n_out>0
            head_dim=D,
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        quant = BatchedTurboQuantProdTorch(
            D,
            b=bits_hi,
            num_heads=H,
            mse_seeds=[1, 2],
            qjl_seeds=[11, 12],
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        x = _rand(H, N, D, seed=2)
        q, norms = quant.quantize_with_norm(x)
        store.append(q, norms, B=B, S_new=S)

        # Split-bucket tensors allocated, single-bucket field empty.
        assert store.mse_packed is None
        assert store.mse_packed_hi is not None
        assert store.mse_packed_lo is not None
        # Expected packed shapes: ceil(n_out * (bits_hi-1) / 8), etc.
        from turboquant.bit_packing import packed_dim

        assert store.mse_packed_hi.shape == (
            H, B, S, packed_dim(n_out, bits_hi - 1)
        )
        assert store.mse_packed_lo.shape == (
            H, B, S, packed_dim(D - n_out, bits_lo - 1)
        )

        q_out, norms_out = store.to_quantized_dict()
        assert set(q_out["mse"].keys()) == {"idx_hi", "idx_lo"}
        assert torch.equal(q_out["mse"]["idx_hi"].reshape(H, N, n_out),
                           q["mse"]["idx_hi"])
        assert torch.equal(q_out["mse"]["idx_lo"].reshape(H, N, D - n_out),
                           q["mse"]["idx_lo"])

        x_hat = quant.dequantize_with_norm(q_out, norms_out)
        assert x_hat.shape == (H, N, D)
        assert torch.isfinite(x_hat).all()


class TestValueStoreSplitBucket:
    @pytest.mark.parametrize(
        "D,n_out,bits_hi,bits_lo",
        [
            (128, 32, 4, 2),
            (128, 32, 5, 3),
        ],
    )
    def test_roundtrip(
        self, D: int, n_out: int, bits_hi: int, bits_lo: int
    ) -> None:
        H, N, B, S = 2, 8, 1, 8
        store = _BatchedValueStore(
            value_bits=bits_hi,
            head_dim=D,
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        quant = BatchedTurboQuantMSETorch(
            D,
            b=bits_hi,
            num_heads=H,
            seeds=[31, 32],
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        x = _rand(H, N, D, seed=3)
        q, norms = quant.quantize_with_norm(x)
        store.append(q, norms, B=B, S_new=S)

        assert store.idx_packed is None
        assert store.idx_packed_hi is not None
        assert store.idx_packed_lo is not None
        from turboquant.bit_packing import packed_dim

        # V stage uses full bit-widths (no -1 here).
        assert store.idx_packed_hi.shape == (
            H, B, S, packed_dim(n_out, bits_hi)
        )
        assert store.idx_packed_lo.shape == (
            H, B, S, packed_dim(D - n_out, bits_lo)
        )

        q_out, norms_out = store.to_quantized_dict()
        assert set(q_out.keys()) == {"idx_hi", "idx_lo"}
        assert torch.equal(q_out["idx_hi"].reshape(H, N, n_out), q["idx_hi"])
        assert torch.equal(q_out["idx_lo"].reshape(H, N, D - n_out), q["idx_lo"])

        x_hat = quant.dequantize_with_norm(q_out, norms_out)
        assert x_hat.shape == (H, N, D)
        assert torch.isfinite(x_hat).all()


class TestAppendAcrossSteps:
    """Multi-step decode: appends concat along seq dim in split mode too."""

    def test_key_store_growth(self) -> None:
        D, H, n_out, bits_hi, bits_lo = 128, 2, 32, 4, 2
        store = _BatchedKeyStore(
            key_bits=bits_hi, head_dim=D,
            n_out=n_out, bits_hi=bits_hi, bits_lo=bits_lo,
        )
        quant = BatchedTurboQuantProdTorch(
            D, bits_hi, H, mse_seeds=[1, 2], qjl_seeds=[11, 12],
            n_out=n_out, bits_hi=bits_hi, bits_lo=bits_lo,
        )
        for step in range(3):
            x = _rand(H, 4, D, seed=100 + step)
            q, norms = quant.quantize_with_norm(x)
            store.append(q, norms, B=1, S_new=4)
        assert store.seq_length() == 12
        assert store.mse_packed_hi.shape[2] == 12
        assert store.mse_packed_lo.shape[2] == 12
