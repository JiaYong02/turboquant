"""Tests for bit_packing.py utilities."""

import pytest
import torch

from turboquant.bit_packing import (
    pack_bits,
    pack_signs,
    packed_dim,
    unpack_bits,
    unpack_signs,
)


class TestPackedDim:
    @pytest.mark.parametrize("d,n_bits,expected", [
        (128, 1, 16),   # 8 per byte
        (128, 2, 32),   # 4 per byte
        (128, 3, 64),   # 2 per byte (3-bit: 6 bits used, 2 wasted per byte)
        (128, 4, 64),   # 2 per byte
        (7,   1, 1),    # ceil(7/8) = 1
        (9,   2, 3),    # ceil(9/4) = 3
    ])
    def test_packed_dim(self, d, n_bits, expected):
        assert packed_dim(d, n_bits) == expected


class TestRoundTrip:
    """pack → unpack must recover the original values exactly."""

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_1d_round_trip(self, n_bits):
        n_vals = 128
        max_val = (1 << n_bits) - 1
        values = torch.randint(0, max_val + 1, (n_vals,), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        recovered = unpack_bits(packed, n_bits, n_vals)
        torch.testing.assert_close(recovered, values)

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_2d_round_trip(self, n_bits):
        max_val = (1 << n_bits) - 1
        values = torch.randint(0, max_val + 1, (10, 128), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        recovered = unpack_bits(packed, n_bits, 128)
        torch.testing.assert_close(recovered, values)

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_4d_round_trip(self, n_bits):
        """Typical cache tensor shape [H, B, S, D]."""
        max_val = (1 << n_bits) - 1
        H, B, S, D = 8, 1, 32, 128
        values = torch.randint(0, max_val + 1, (H, B, S, D), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        recovered = unpack_bits(packed, n_bits, D)
        torch.testing.assert_close(recovered, values)

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_non_multiple_of_vpb(self, n_bits):
        """D that is not a multiple of values-per-byte must still round-trip."""
        D = 7  # 7 is not divisible by 2 or 4 or 8
        max_val = (1 << n_bits) - 1
        values = torch.randint(0, max_val + 1, (D,), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        recovered = unpack_bits(packed, n_bits, D)
        torch.testing.assert_close(recovered, values)


class TestPackedSize:
    """Packed tensor must have the expected number of bytes."""

    @pytest.mark.parametrize("n_bits,D,expected_packed", [
        (1,  128, 16),
        (2,  128, 32),
        (3,  128, 64),
        (4,  128, 64),
        (1,   64,  8),
        (4,   64, 32),
    ])
    def test_packed_last_dim(self, n_bits, D, expected_packed):
        max_val = (1 << n_bits) - 1
        values = torch.randint(0, max_val + 1, (D,), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        assert packed.shape[-1] == expected_packed, (
            f"n_bits={n_bits}, D={D}: expected {expected_packed} bytes, "
            f"got {packed.shape[-1]}"
        )


class TestSignsRoundTrip:
    """pack_signs → unpack_signs must recover {-1, +1} exactly."""

    def test_1d_signs(self):
        signs = torch.randint(0, 2, (128,)).mul(2).sub(1).to(torch.int8)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 128)
        torch.testing.assert_close(recovered, signs)

    def test_4d_signs(self):
        H, B, S, D = 8, 1, 32, 128
        signs = torch.randint(0, 2, (H, B, S, D)).mul(2).sub(1).to(torch.int8)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, D)
        torch.testing.assert_close(recovered, signs)

    def test_packed_size(self):
        signs = torch.randint(0, 2, (128,)).mul(2).sub(1).to(torch.int8)
        packed = pack_signs(signs)
        assert packed.shape[-1] == 16   # 128 / 8

    def test_all_positive(self):
        signs = torch.ones(64, dtype=torch.int8)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 64)
        torch.testing.assert_close(recovered, signs)

    def test_all_negative(self):
        signs = -torch.ones(64, dtype=torch.int8)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 64)
        torch.testing.assert_close(recovered, signs)


class TestCompressionSavings:
    """Packing must actually reduce tensor size compared to uint8 storage."""

    @pytest.mark.parametrize("n_bits", [1, 2, 3, 4])
    def test_packed_smaller_than_unpacked(self, n_bits):
        D = 128
        max_val = (1 << n_bits) - 1
        values = torch.randint(0, max_val + 1, (D,), dtype=torch.uint8)
        packed = pack_bits(values, n_bits)
        assert packed.nelement() < values.nelement(), (
            f"Packed ({packed.nelement()} bytes) should be < "
            f"unpacked ({values.nelement()} bytes) for n_bits={n_bits}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestCudaDevice:
    """Packing/unpacking must work on CUDA tensors."""

    @pytest.mark.parametrize("n_bits", [1, 2, 4])
    def test_cuda_round_trip(self, n_bits):
        max_val = (1 << n_bits) - 1
        values = torch.randint(
            0, max_val + 1, (128,), dtype=torch.uint8, device="cuda"
        )
        packed = pack_bits(values, n_bits)
        assert packed.device.type == "cuda"
        recovered = unpack_bits(packed, n_bits, 128)
        assert recovered.device.type == "cuda"
        torch.testing.assert_close(recovered, values)

    def test_cuda_signs_round_trip(self):
        signs = (
            torch.randint(0, 2, (128,), device="cuda").mul(2).sub(1).to(torch.int8)
        )
        packed = pack_signs(signs)
        assert packed.device.type == "cuda"
        recovered = unpack_signs(packed, 128)
        torch.testing.assert_close(recovered, signs)
