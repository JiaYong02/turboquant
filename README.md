# TurboQuant

A Python implementation of **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** (Zandieh et al., Google Research / NYU / Google DeepMind, April 2025).

## Overview

TurboQuant compresses high-dimensional floating-point vectors into low-bitwidth integers with provably near-optimal distortion. It solves two quantization problems simultaneously:

- **MSE distortion** — minimize geometric distance between original and reconstructed vectors
- **Inner product distortion** — preserve similarity estimates (crucial for attention mechanisms, vector search)

The key insight: MSE-optimal quantizers introduce a systematic bias (`2/π ≈ 0.636×`) for inner product estimation. TurboQuant corrects this with a two-stage approach.

### Two Algorithms

| Algorithm | Optimizes | Use case |
|---|---|---|
| `TurboQuantMSE` | Mean squared error | General compression, nearest-neighbor indexing |
| `TurboQuantProd` | Inner product (unbiased) | KV cache, attention, similarity search |

### Key Properties

- **Online / data-oblivious** — no calibration data, preprocessing, or codebook training required
- **Accelerator-friendly** — all operations are matrix multiplies + element-wise; fully vectorizable on GPU
- **Near-optimal** — within `√(3π)/2 ≈ 2.7×` of the information-theoretic lower bound
- **Provably unbiased** — `TurboQuantProd` guarantees `E[⟨y, x̃⟩] = ⟨y, x⟩` for any query `y`

## Distortion Bounds

### TurboQuantMSE

| Bit-width | MSE (empirical) | Lower bound | Compression (from FP32) |
|---|---|---|---|
| 1 bit | ≈ 0.36 | 0.25 | 32× |
| 2 bits | ≈ 0.117 | 0.0625 | 16× |
| 3 bits | ≈ 0.03 | 0.0156 | ~10.7× |
| 4 bits | ≈ 0.009 | 0.0039 | 8× |

### TurboQuantProd (inner product distortion, scaled by 1/d)

| Bit-width | D_prod | Unbiased? |
|---|---|---|
| 1 bit | ≈ 1.57/d | Yes |
| 2 bits | ≈ 0.56/d | Yes |
| 3 bits | ≈ 0.18/d | Yes |
| 4 bits | ≈ 0.047/d | Yes |

## How It Works

### Stage 1: Random Rotation (both algorithms)

```
y = Π · x
```

Multiplying by a random orthogonal matrix `Π` uniformly distributes the vector on the unit hypersphere, making each coordinate follow a known Beta distribution — independent of the input. This enables optimal scalar quantization.

### TurboQuantMSE — Lloyd-Max Scalar Quantization

```
idx_j = argmin_k |y_j - c_k|   for each coordinate j
```

Centroids `c_k` are precomputed by solving a 1D k-means problem on the Beta distribution (Lloyd-Max algorithm). Dequantization reverses the lookup and rotation:

```
x̃ = Πᵀ · [c_{idx_j}]_j
```

### TurboQuantProd — Two-Stage with QJL Correction

Uses `(b-1)` bits for the MSE stage, then 1 bit for a **Quantized Johnson-Lindenstrauss (QJL)** correction on the residual:

```
r = x - x̃_mse                    # residual
γ = ‖r‖₂                          # residual norm
qjl = sign(S · r)                 # 1-bit QJL projection

# Dequantize:
x̃ = x̃_mse + √(π/2)/d · γ · Sᵀ · qjl
```

The `√(π/2)` factor ensures unbiasedness. The QJL stage contributes exactly 1 bit per coordinate.

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies**: `numpy >= 1.24`, `scipy >= 1.10`

## Quick Start

```python
import numpy as np
from turboquant import TurboQuantMSE, TurboQuantProd

d = 256   # dimension
b = 4     # bits per coordinate

# --- MSE-optimal quantizer ---
mse_q = TurboQuantMSE(d=d, b=b, seed=42)

x = np.random.randn(d)
x /= np.linalg.norm(x)               # normalize to unit sphere

quantized = mse_q.quantize(x)        # {'idx': array of shape (d,)}
x_hat = mse_q.dequantize(quantized)  # reconstructed vector

print(f"MSE: {np.mean((x - x_hat)**2):.4f}")  # ~0.009 for b=4

# --- Inner-product-optimal quantizer ---
prod_q = TurboQuantProd(d=d, b=b, seed=42)

quantized = prod_q.quantize(x)       # {'mse': ..., 'qjl': ..., 'gamma': ...}
x_hat = prod_q.dequantize(quantized)

y = np.random.randn(d)
print(f"True IP:  {np.dot(y, x):.4f}")
print(f"Est. IP:  {np.dot(y, x_hat):.4f}")   # unbiased estimate
```

## Project Structure

```
turboquant/
├── src/turboquant/
│   ├── rotation.py          # RandomRotation — Haar-distributed QR rotation
│   ├── codebook.py          # LloydMaxCodebook — optimal scalar quantizer
│   ├── qjl.py               # QJL — 1-bit unbiased inner product quantizer
│   ├── quantizer_mse.py     # TurboQuantMSE (Algorithm 1)
│   ├── quantizer_prod.py    # TurboQuantProd (Algorithm 2)
│   └── utils.py             # Beta PDF, Gaussian approx, unit vector generation
├── tests/
│   ├── test_rotation.py
│   ├── test_codebook.py
│   ├── test_qjl.py
│   ├── test_quantizer_mse.py
│   ├── test_quantizer_prod.py
│   └── test_distortion_bounds.py
└── docs/
    ├── notes/turboquant_notes.md         # Algorithm reference
    └── notes/implementation_reference.md # Implementation guide
```

## Running Tests

```bash
pytest tests/ -v
```

## Application: KV Cache Quantization

In transformer attention, queries `q` attend over cached keys `k`:

```
score = ⟨q, k⟩ / √d_head
```

`TurboQuantProd` quantizes keys/values online as they are generated, preserving inner product accuracy. From the paper:

- **3.5 bits**: Matches full-precision performance on LongBench (Llama-3.1-8B)
- **2.5 bits**: Only marginal degradation with 5×+ compression
- **Needle-in-a-haystack**: Perfect recall (0.997) at 4× compression, matching full precision

## Reference

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```
