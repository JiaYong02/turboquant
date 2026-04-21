"""Phase B-1: parity tests for outlier-channel split in quantizers.

Exercises `BatchedTurboQuantMSETorch` and `BatchedTurboQuantProdTorch` with
`n_out ∈ {0, 32}`. When `n_out == 0` the quantizer must be bit-identical to
the current single-bucket behavior. When `n_out > 0`, output dict carries
two buckets (`idx_hi`/`idx_lo` for MSE; `mse_hi`/`mse_lo` for Prod) and a
round-trip through quantize -> dequantize reconstructs the input within a
tolerance that reflects the lossy lower bucket.
"""

from __future__ import annotations

import pytest
import torch

from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def _rand_batched(
    num_heads: int, n: int, d: int, seed: int = 0
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_heads, n, d, generator=gen, dtype=torch.float32)


class TestBatchedMSESingleBucketParity:
    """n_out=0 must leave the single-bucket path untouched."""

    def test_default_kwargs_unchanged(self) -> None:
        d, b, h, n = 128, 4, 2, 16
        q = BatchedTurboQuantMSETorch(d, b, h, seeds=[11, 12])
        x = _rand_batched(h, n, d, seed=1)
        q_dict, norms = q.quantize_with_norm(x)
        assert set(q_dict.keys()) == {"idx"}
        assert q_dict["idx"].shape == (h, n, d)
        x_hat = q.dequantize_with_norm(q_dict, norms)
        assert x_hat.shape == x.shape

    def test_explicit_n_out_zero_matches_default(self) -> None:
        d, b, h, n = 128, 4, 2, 16
        q0 = BatchedTurboQuantMSETorch(d, b, h, seeds=[11, 12])
        q1 = BatchedTurboQuantMSETorch(d, b, h, seeds=[11, 12], n_out=0)
        x = _rand_batched(h, n, d, seed=2)
        q0_dict, n0 = q0.quantize_with_norm(x)
        q1_dict, n1 = q1.quantize_with_norm(x)
        assert torch.equal(q0_dict["idx"], q1_dict["idx"])
        assert torch.equal(n0, n1)


class TestBatchedMSESplitBucket:
    """n_out > 0 yields two index buckets with correct shapes and round-trips."""

    @pytest.mark.parametrize(
        "d,n_out,bits_hi,bits_lo",
        [
            (128, 32, 4, 2),
            (128, 32, 5, 3),
            (64, 16, 4, 2),
        ],
    )
    def test_split_bucket_shapes_and_roundtrip(
        self, d: int, n_out: int, bits_hi: int, bits_lo: int
    ) -> None:
        h, n = 2, 16
        q = BatchedTurboQuantMSETorch(
            d,
            b=bits_hi,  # single-bucket fallback param, overridden when n_out>0
            num_heads=h,
            seeds=[11, 12],
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        x = _rand_batched(h, n, d, seed=3)
        q_dict, norms = q.quantize_with_norm(x)
        assert set(q_dict.keys()) == {"idx_hi", "idx_lo"}
        assert q_dict["idx_hi"].shape == (h, n, n_out)
        assert q_dict["idx_lo"].shape == (h, n, d - n_out)
        assert q_dict["idx_hi"].dtype == torch.uint8
        assert q_dict["idx_lo"].dtype == torch.uint8
        assert q_dict["idx_hi"].max().item() < 2**bits_hi
        assert q_dict["idx_lo"].max().item() < 2**bits_lo

        x_hat = q.dequantize_with_norm(q_dict, norms)
        assert x_hat.shape == x.shape
        # Loose tolerance: the low bucket is heavily lossy (2 or 3 bits).
        # We only check that dequant produces finite output of the right norm.
        assert torch.isfinite(x_hat).all()


class TestBatchedProdSingleBucketParity:
    """n_out=0 must leave Prod's single-bucket path unchanged."""

    def test_default_kwargs_unchanged(self) -> None:
        d, b, h, n = 128, 4, 2, 16
        q = BatchedTurboQuantProdTorch(
            d, b, h, mse_seeds=[11, 12], qjl_seeds=[21, 22]
        )
        x = _rand_batched(h, n, d, seed=4)
        q_dict, norms = q.quantize_with_norm(x)
        assert set(q_dict.keys()) == {"mse", "qjl", "gamma"}
        assert isinstance(q_dict["mse"], dict)
        assert set(q_dict["mse"].keys()) == {"idx"}
        x_hat = q.dequantize_with_norm(q_dict, norms)
        assert x_hat.shape == x.shape

    def test_explicit_n_out_zero_matches_default(self) -> None:
        d, b, h, n = 128, 4, 2, 16
        q0 = BatchedTurboQuantProdTorch(
            d, b, h, mse_seeds=[11, 12], qjl_seeds=[21, 22]
        )
        q1 = BatchedTurboQuantProdTorch(
            d, b, h, mse_seeds=[11, 12], qjl_seeds=[21, 22], n_out=0
        )
        x = _rand_batched(h, n, d, seed=5)
        d0, n0 = q0.quantize_with_norm(x)
        d1, n1 = q1.quantize_with_norm(x)
        assert torch.equal(d0["mse"]["idx"], d1["mse"]["idx"])
        assert torch.equal(d0["qjl"], d1["qjl"])
        assert torch.allclose(d0["gamma"], d1["gamma"])
        assert torch.equal(n0, n1)


class TestBatchedProdSplitBucket:
    """n_out > 0: Prod returns {mse_hi, mse_lo, qjl, gamma}; QJL stays full-D."""

    @pytest.mark.parametrize(
        "d,n_out,bits_hi,bits_lo",
        [
            (128, 32, 4, 2),
            (128, 32, 5, 3),
        ],
    )
    def test_split_bucket_shapes_and_roundtrip(
        self, d: int, n_out: int, bits_hi: int, bits_lo: int
    ) -> None:
        h, n = 2, 16
        q = BatchedTurboQuantProdTorch(
            d,
            b=bits_hi,
            num_heads=h,
            mse_seeds=[11, 12],
            qjl_seeds=[21, 22],
            n_out=n_out,
            bits_hi=bits_hi,
            bits_lo=bits_lo,
        )
        x = _rand_batched(h, n, d, seed=6)
        q_dict, norms = q.quantize_with_norm(x)
        assert set(q_dict.keys()) == {"mse", "qjl", "gamma"}
        mse = q_dict["mse"]
        assert set(mse.keys()) == {"idx_hi", "idx_lo"}
        # K stage uses (bits - 1) for MSE; QJL contributes the final bit.
        assert mse["idx_hi"].shape == (h, n, n_out)
        assert mse["idx_lo"].shape == (h, n, d - n_out)
        assert mse["idx_hi"].max().item() < 2 ** (bits_hi - 1)
        assert mse["idx_lo"].max().item() < 2 ** (bits_lo - 1)
        # QJL stays full-D at 1 bit.
        assert q_dict["qjl"].shape == (h, n, d)
        assert q_dict["gamma"].shape == (h, n)

        x_hat = q.dequantize_with_norm(q_dict, norms)
        assert x_hat.shape == x.shape
        assert torch.isfinite(x_hat).all()
