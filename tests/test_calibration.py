"""Phase B-4: tests for outlier-channel calibration.

The calibration routine picks the top-``n_out`` channels of the rotated
coordinate space (``y = Pi @ x``) by per-channel RMS. Tests here verify:

1. When the *rotated* distribution has a few channels with much larger
   scale than the others, the calibration reliably identifies those
   channels as outliers.
2. ``apply_outlier_permutation`` folds the permutation into the rotation
   so subsequent rotations of new data emit outlier channels in the
   prefix ``[0, n_out)``.
3. End-to-end round-trip with a split-bucket quantizer after calibration
   recovers inputs at expected quality (equal or better than the
   single-bucket baseline at the same average bits).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from turboquant.calibration import apply_outlier_permutation, calibrate_outliers
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def _build_mse(d: int, bits: int, H: int, *, n_out: int = 0,
               bits_hi: int | None = None, bits_lo: int | None = None):
    seeds = [1000 + h for h in range(H)]
    return BatchedTurboQuantMSETorch(
        d, bits, H, seeds,
        n_out=n_out,
        bits_hi=bits_hi,
        bits_lo=bits_lo,
    )


class TestCalibration:
    def test_identifies_scaled_channels_in_rotated_space(self):
        """Scaled coords in rotated space should be selected as outliers."""
        H, D, N = 3, 64, 512
        n_out = 8
        mse = _build_mse(D, 4, H)

        torch.manual_seed(0)
        # Draw y ~ N(0, 1) in rotated space, then boost a fixed subset
        # of channels per head. x = Pi^T @ y — rotating x by Pi recovers
        # the boosted y, so those same channels must show highest RMS.
        target_channels = {
            0: [3, 17, 40, 55, 5, 22, 31, 48],
            1: [1, 10, 20, 30, 40, 50, 60, 7],
            2: [63, 62, 61, 60, 59, 58, 57, 56],
        }
        y = torch.randn(H, N, D)
        for h, chans in target_channels.items():
            y[h, :, chans] *= 10.0
        x = torch.bmm(y, mse.rotation.Pi)  # x = y @ Pi  (since Pi is orth)

        perm = calibrate_outliers(mse, x, n_out=n_out)
        assert perm.shape == (H, D)
        assert perm.dtype == torch.long
        # Each row must be a valid permutation.
        for h in range(H):
            assert sorted(perm[h].tolist()) == list(range(D))
        # First n_out entries must be exactly the boosted channels.
        for h, chans in target_channels.items():
            picked = set(perm[h, :n_out].tolist())
            assert picked == set(chans), (
                f"head {h}: expected outlier set {sorted(chans)}, "
                f"got {sorted(picked)}"
            )

    def test_apply_permutation_places_outliers_in_prefix(self):
        """After apply_outlier_permutation, rotations emit outliers first."""
        H, D, N = 2, 32, 256
        n_out = 4
        mse = _build_mse(D, 4, H)

        torch.manual_seed(1)
        boosted = {0: [5, 12, 20, 28], 1: [0, 9, 17, 25]}
        y = torch.randn(H, N, D)
        for h, chans in boosted.items():
            y[h, :, chans] *= 8.0
        # Build x such that Pi @ x == y, i.e. x = Pi^T y.
        x = torch.bmm(y, mse.rotation.Pi)

        perm = calibrate_outliers(mse, x, n_out=n_out)
        apply_outlier_permutation(mse, perm)

        # Rotate the same x through the *new* Pi. The new rotated output
        # is a permutation of the old y, so the boosted channels must now
        # occupy the prefix [0, n_out).
        y_new = mse.rotation.rotate(x)
        prefix_rms = y_new[..., :n_out].pow(2).mean(dim=1).sqrt().mean(dim=-1)
        suffix_rms = y_new[..., n_out:].pow(2).mean(dim=1).sqrt().mean(dim=-1)
        assert (prefix_rms > suffix_rms * 3.0).all(), (
            f"prefix RMS {prefix_rms.tolist()} must dominate suffix "
            f"{suffix_rms.tolist()} after calibration"
        )

    def test_calibrate_rejects_wrong_shape(self):
        mse = _build_mse(16, 4, 2)
        with pytest.raises(ValueError, match="shape"):
            calibrate_outliers(mse, torch.randn(3, 128, 16), n_out=4)
        with pytest.raises(ValueError, match="shape"):
            calibrate_outliers(mse, torch.randn(2, 128, 8), n_out=4)

    def test_calibrate_rejects_bad_n_out(self):
        mse = _build_mse(16, 4, 2)
        x = torch.randn(2, 64, 16)
        with pytest.raises(ValueError, match="n_out"):
            calibrate_outliers(mse, x, n_out=-1)
        with pytest.raises(ValueError, match="n_out"):
            calibrate_outliers(mse, x, n_out=17)

    def test_calibrate_rejects_empty_batch(self):
        mse = _build_mse(16, 4, 2)
        with pytest.raises(ValueError, match="at least one"):
            calibrate_outliers(mse, torch.randn(2, 0, 16), n_out=4)

    def test_apply_permutation_is_orthogonal_preserving(self):
        """Rotation remains orthogonal after applying a row permutation."""
        mse = _build_mse(32, 4, 2)
        torch.manual_seed(2)
        perm = torch.stack([torch.randperm(32) for _ in range(2)])
        apply_outlier_permutation(mse, perm)
        Pi = mse.rotation.Pi
        # Pi @ Pi^T ≈ I for each head
        eye = torch.eye(32).expand(2, -1, -1)
        torch.testing.assert_close(
            torch.bmm(Pi, Pi.transpose(-1, -2)),
            eye,
            atol=1e-5, rtol=1e-5,
        )


class TestCalibrationWithQuantizer:
    """Calibration is only useful if the split quantizer benefits from it."""

    def test_calibration_improves_over_random_perm(self):
        """Calibrated split beats a random perm at the same avg bit-width."""
        H, D, N = 2, 64, 256
        n_out = 16
        bits_hi, bits_lo = 4, 2

        mse_seeds = [100 + h for h in range(H)]
        qjl_seeds = [200 + h for h in range(H)]

        def build():
            return BatchedTurboQuantProdTorch(
                D, b=3, num_heads=H,
                mse_seeds=mse_seeds, qjl_seeds=qjl_seeds,
                n_out=n_out, bits_hi=bits_hi, bits_lo=bits_lo,
            )

        torch.manual_seed(3)
        # Outlier structure in the rotated space of a reference Pi.
        ref = build()
        y = torch.randn(H, N, D)
        for h in range(H):
            y[h, :, h * 3 : h * 3 + n_out] *= 6.0
        x = torch.bmm(y, ref.mse.rotation.Pi)

        # Baseline: random permutation (no calibration).
        random_prod = build()
        random_perm = torch.stack([torch.randperm(D) for _ in range(H)])
        apply_outlier_permutation(random_prod.mse, random_perm)
        q_rand, nr = random_prod.quantize_with_norm(x)
        x_rand = random_prod.dequantize_with_norm(q_rand, nr)
        cos_rand = F.cosine_similarity(
            x.reshape(H, -1), x_rand.reshape(H, -1), dim=-1
        )

        # Calibrated: derived from the data's actual outlier pattern.
        cal_prod = build()
        perm = calibrate_outliers(cal_prod.mse, x, n_out=n_out)
        apply_outlier_permutation(cal_prod.mse, perm)
        q_cal, nc = cal_prod.quantize_with_norm(x)
        x_cal = cal_prod.dequantize_with_norm(q_cal, nc)
        cos_cal = F.cosine_similarity(
            x.reshape(H, -1), x_cal.reshape(H, -1), dim=-1
        )

        # Calibration should noticeably improve fidelity per head.
        assert (cos_cal > cos_rand).all(), (
            f"calibrated {cos_cal.tolist()} should beat random "
            f"{cos_rand.tolist()}"
        )
        # And meet a basic quality floor.
        assert (cos_cal >= 0.75).all(), f"calibrated cos {cos_cal.tolist()}"
