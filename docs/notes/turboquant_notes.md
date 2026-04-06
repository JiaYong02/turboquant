# TurboQuant Framework

These detailed notes cover the **TurboQuant** framework, covering both **TurboQuant_mse** (MSE-optimal) and **TurboQuant_prod** (inner-product-optimal) algorithms. Based on the paper "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et al., Google Research / NYU / Google DeepMind, April 2025).

---

## Core Problem

Vector quantization compresses high-dimensional floating-point vectors into low-bitwidth integers while minimizing distortion. Two distortion measures matter:

1. **MSE distortion**: `D_mse = E[||x - x_tilde||^2]` -- geometric distance between original and reconstruction
2. **Inner product distortion**: `D_prod = E[|<y,x> - <y,x_tilde>|^2]` -- error in similarity estimation

**Key insight**: MSE-optimal quantizers are **biased** for inner product estimation. At 1-bit, the multiplicative bias is `2/pi ≈ 0.636` (i.e., inner products are "shrunk" by ~36%). This bias diminishes with higher bit-widths but never fully disappears.

TurboQuant solves both problems:
- `TurboQuant_mse` (Algorithm 1): minimizes MSE distortion
- `TurboQuant_prod` (Algorithm 2): provides **unbiased** inner product estimation with low distortion

---

## Key Properties

- **Data-oblivious / Online**: No preprocessing, calibration data, or codebook training needed. Works instantly on any input vector. This is crucial for real-time applications like KV cache quantization during inference.
- **Accelerator-friendly**: All operations are vectorizable (matrix multiplies, element-wise operations). No branching or search loops.
- **Near-optimal**: Within a factor of `sqrt(3*pi)/2 ≈ 2.7` of the information-theoretic lower bound for MSE distortion.
- **Unit-norm assumption**: The algorithm assumes `||x|| = 1`. For non-unit vectors, store the L2 norm separately in full precision and rescale after dequantization.

---

## Algorithm 1: TurboQuant_mse (MSE-Optimal)

**Goal**: Minimize `D_mse = E[||x - x_tilde||^2]`

### Global Setup (done once)

#### Step 1: Generate Random Rotation Matrix Pi

- Generate a random matrix `A` with i.i.d. entries from `N(0, 1)`, shape `(d, d)`
- Compute QR decomposition: `Pi, R = QR(A)`
- `Pi` is a `d x d` orthogonal matrix (random rotation)
- **Store Pi** for reuse across all quantization/dequantization calls

**Why**: Multiplying any unit vector `x` by `Pi` produces a vector uniformly distributed on the unit hypersphere `S^{d-1}`, regardless of the original `x`. This makes the algorithm data-oblivious.

#### Step 2: Precompute Lloyd-Max Codebook

Each coordinate of `Pi * x` follows a **Beta distribution** (Lemma 1 in paper):

```
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

for `x in [-1, 1]`.

In high dimensions (d >> 1), this converges to `N(0, 1/d)` (Gaussian with variance `1/d`).

The optimal scalar quantizer for this distribution is found by solving a **continuous 1-D k-means** problem (Lloyd-Max algorithm):

```
C(f_X, b) = min_{-1 <= c_1 <= ... <= c_{2^b} <= 1} sum_{i=1}^{2^b} integral |x - c_i|^2 * f_X(x) dx
```

where the integral for each centroid `c_i` runs over its Voronoi region (midpoints between consecutive centroids).

**Precomputed codebook examples** (for large d, approximating Gaussian):

| Bit-width b | Number of centroids | Centroid values (scaled) |
|---|---|---|
| 1 | 2 | `{-sqrt(2/(pi*d)), +sqrt(2/(pi*d))}` |
| 2 | 4 | `{-1.51/sqrt(d), -0.453/sqrt(d), +0.453/sqrt(d), +1.51/sqrt(d)}` |
| 3 | 8 | Solve numerically |
| 4 | 16 | Solve numerically |

**Important**: The codebook depends on dimension `d` and bit-width `b`. Precompute once and store.

### Quantization: Quant_mse(x)

```
Input: vector x in R^d (unit norm)
1. y = Pi * x                          // random rotation, shape (d,)
2. For each j in [d]:
     idx_j = argmin_{k in [2^b]} |y_j - c_k|   // nearest centroid index
3. Output: idx (vector of b-bit integers, shape (d,))
```

**Total storage**: `b * d` bits for the index vector.

### Dequantization: DeQuant_mse(idx)

```
Input: idx (vector of indices)
1. For each j in [d]:
     y_tilde_j = c_{idx_j}             // look up centroid
2. x_tilde = Pi^T * y_tilde            // inverse rotation (Pi^T = Pi^{-1})
3. Output: x_tilde (reconstructed vector, shape (d,))
```

### MSE Distortion Bounds

The total MSE equals `d * C(f_X, b)` (sum of d identical scalar quantization errors):

| Bit-width b | D_mse (approx) | Compression ratio |
|---|---|---|
| 1 | 0.36 | 32x (from FP32) |
| 2 | 0.117 | 16x |
| 3 | 0.03 | ~10.7x |
| 4 | 0.009 | 8x |
| General | <= sqrt(3*pi)/2 * (1/4^b) | -- |

**Lower bound** (information-theoretic): `D_mse >= 1/4^b` for any quantizer.

---

## Algorithm 2: TurboQuant_prod (Inner-Product-Optimal)

**Goal**: Provide unbiased inner product estimation with minimal distortion.

### Why MSE-Optimal is Biased for Inner Products

For `b = 1`, MSE-optimal quantization becomes `sign(Pi * x)` with dequantization scaling by `sqrt(2/(pi*d))`. This gives:

```
E[<y, DeQuant_mse(Quant_mse(x))>] = (2/pi) * <y, x>
```

The factor `2/pi ≈ 0.636` is a **multiplicative bias** -- inner products are systematically underestimated. This bias decreases with higher bit-widths but is always present.

### Two-Stage Solution

TurboQuant_prod uses `b` total bits per coordinate:
- **(b-1) bits** for the MSE stage (coarse approximation)
- **1 bit** for the QJL residual correction (bias correction)

### Global Setup

1. Instantiate a `TurboQuant_mse` with bit-width `b - 1` (not `b`!)
2. Generate random projection matrix `S in R^{d x d}` with i.i.d. entries from `N(0, 1)`
3. **Store both Pi (from MSE instance) and S**

### Quantization: Quant_prod(x)

```
Input: vector x in R^d (unit norm)
1. idx = Quant_mse(x)                  // MSE quantize at (b-1) bits
2. x_tilde_mse = DeQuant_mse(idx)      // reconstruct MSE approximation
3. r = x - x_tilde_mse                 // residual vector
4. gamma = ||r||_2                      // L2 norm of residual (scalar)
5. qjl = sign(S * r)                   // QJL: project and take signs
6. Output: (idx, qjl, gamma)
```

**Total storage per vector**:
- `idx`: `(b-1) * d` bits
- `qjl`: `d` bits (one sign bit per coordinate)
- `gamma`: 1 floating-point scalar (e.g., 16 or 32 bits)
- **Total**: `b * d + sizeof(float)` bits

### Dequantization: DeQuant_prod(idx, qjl, gamma)

```
Input: idx, qjl (sign vector in {-1, +1}^d), gamma (scalar)
1. x_tilde_mse = DeQuant_mse(idx)      // MSE reconstruction
2. x_tilde_qjl = (sqrt(pi/2) / d) * gamma * S^T * qjl   // QJL reconstruction
3. x_tilde = x_tilde_mse + x_tilde_qjl // final reconstruction
4. Output: x_tilde
```

### Key Formula: QJL Reconstruction

The QJL dequantization formula is (Definition 1 in paper):

```
Q^{-1}_qjl(z) = sqrt(pi/2) / d * S^T * z
```

When combined with the magnitude scaling `gamma = ||r||_2`:

```
x_tilde_qjl = sqrt(pi/2) / d * gamma * S^T * qjl
```

The constant `sqrt(pi/2) ≈ 1.2533` is a **debiasing factor** derived from the half-normal distribution. It ensures:

```
E[<y, x_tilde_qjl>] = <y, r>
```

which means the QJL stage provides an **unbiased estimate** of the residual's contribution to any inner product.

### Unbiasedness Proof (intuition)

```
E[<y, x_tilde>] = E[<y, x_tilde_mse + x_tilde_qjl>]
                = <y, x_tilde_mse> + E[<y, x_tilde_qjl>]
                = <y, x_tilde_mse> + <y, r>           (QJL is unbiased, Lemma 4)
                = <y, x_tilde_mse> + <y, x - x_tilde_mse>
                = <y, x>
```

### Inner Product Distortion Bounds

| Bit-width b | D_prod (approx) |
|---|---|
| 1 | 1.57 / d |
| 2 | 0.56 / d |
| 3 | 0.18 / d |
| 4 | 0.047 / d |
| General | <= sqrt(3) * pi^2 * ||y||^2 / (2*d) * (1/4^b) |

**Lower bound**: `D_prod >= ||y||^2 / (d * 4^b)` for any quantizer.

The relationship between MSE and inner product distortion is:

```
D_prod <= (pi / (2*d)) * ||y||^2 * D_mse(b-1)
```

where `D_mse(b-1)` is the MSE distortion at bit-width `b-1`.

---

## QJL (Quantized Johnson-Lindenstrauss) Details

The QJL transform (from [62] in paper) is a standalone 1-bit quantizer with strong theoretical guarantees.

### Definition

- **Quantize**: `Q_qjl(x) = sign(S * x)` where `S ~ N(0,1)^{d x d}`
- **Dequantize**: `Q^{-1}_qjl(z) = sqrt(pi/2) / d * S^T * z`

### Properties

1. **Unbiased**: `E[<y, Q^{-1}_qjl(Q_qjl(x))>] = <y, x>` for any `x` on unit sphere
2. **Variance bound**: `Var(<y, Q^{-1}_qjl(Q_qjl(x))>) <= pi / (2*d) * ||y||^2`
3. **1-bit per coordinate**: Each coordinate is compressed to a single sign bit
4. The matrix `S` is shared between quantizer and dequantizer (must be stored or regenerated from same seed)

### Why sqrt(pi/2)?

Each row `s_i` of `S` is drawn from `N(0, I_d)`. The quantity `sign(s_i^T * x)` loses magnitude information. The expected absolute value of a standard normal is `sqrt(2/pi)`, so multiplying by `sqrt(pi/2)` corrects this:

```
E[sqrt(pi/2) * sign(s_i^T * x) * s_i^T * y] = <x, y>  (approximately, for unit x)
```

---

## Practical Implementation Details

### Generating the Random Rotation Matrix Pi

```python
import numpy as np

def generate_rotation_matrix(d, seed=None):
    rng = np.random.RandomState(seed)
    A = rng.randn(d, d)
    Pi, _ = np.linalg.qr(A)
    return Pi  # shape (d, d), orthogonal
```

**Note on Hadamard alternative**: For GPU efficiency, some implementations use a randomized Hadamard transform instead of a full random rotation. This is `O(d log d)` instead of `O(d^2)` for the matrix multiply. The paper uses full random rotation for theoretical guarantees.

### Computing Lloyd-Max Centroids

The Lloyd-Max algorithm iterates:
1. Fix centroids, assign each point to nearest centroid (Voronoi partition)
2. Fix partition, update centroids to be the conditional mean within each region

For our known distribution `f_X`:

```python
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma as gamma_func

def beta_pdf(x, d):
    """PDF of coordinate distribution on unit hypersphere."""
    coeff = gamma_func(d / 2) / (np.sqrt(np.pi) * gamma_func((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)

def compute_codebook(d, b, max_iter=1000, tol=1e-12):
    """Compute optimal Lloyd-Max centroids for bit-width b and dimension d."""
    n_centroids = 2 ** b
    # Initialize centroids uniformly in [-1, 1]
    centroids = np.linspace(-1, 1, n_centroids + 2)[1:-1]

    for _ in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = [-1.0]
        for i in range(n_centroids - 1):
            boundaries.append((centroids[i] + centroids[i + 1]) / 2)
        boundaries.append(1.0)

        # Update centroids to conditional mean within each Voronoi region
        new_centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = quad(lambda x: x * beta_pdf(x, d), lo, hi)
            den, _ = quad(lambda x: beta_pdf(x, d), lo, hi)
            new_centroids[i] = num / den if den > 0 else (lo + hi) / 2

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return centroids
```

### Norm Handling for Non-Unit Vectors

```python
def quantize_with_norm(x, quant_fn):
    norm = np.linalg.norm(x)
    x_unit = x / norm  # normalize to unit sphere
    quantized = quant_fn(x_unit)
    return quantized, norm  # store norm separately

def dequantize_with_norm(quantized, norm, dequant_fn):
    x_unit_approx = dequant_fn(quantized)
    return x_unit_approx * norm  # rescale
```

### Memory Layout

For `n` vectors of dimension `d` at bit-width `b`:

**TurboQuant_mse**:
- Index array: `n * d * b` bits
- Norms (if non-unit): `n * 32` bits (float32)
- Rotation matrix Pi: `d * d * 32` bits (shared, stored once)
- Codebook: `2^b * 32` bits (shared, stored once)

**TurboQuant_prod**:
- MSE indices: `n * d * (b-1)` bits
- QJL signs: `n * d * 1` bits
- Residual norms gamma: `n * 32` bits (float32)
- Input norms (if non-unit): `n * 32` bits (float32)
- Rotation matrix Pi: `d * d * 32` bits (shared)
- Projection matrix S: `d * d * 32` bits (shared)
- Codebook: `2^(b-1) * 32` bits (shared)

### Outlier Channel Strategy (from experiments)

For KV cache quantization, the paper splits channels into **outlier** and **non-outlier** sets, applying different bit-widths:

- **2.5-bit config**: 32 outlier channels at 3 bits + 96 normal channels at 2 bits = `(32*3 + 96*2)/128 = 2.5` effective bits
- **3.5-bit config**: Similar split with higher allocation to outliers

Outlier detection can use per-layer statistics (e.g., channels with largest absolute values).

---

## Distortion Bound Summary

### Upper Bounds (TurboQuant achieves)

| Measure | Bound |
|---|---|
| MSE | `D_mse <= sqrt(3*pi)/2 * (1/4^b) ≈ 2.72 / 4^b` |
| Inner Product | `D_prod <= sqrt(3)*pi^2/(2*d) * ||y||^2 * (1/4^b)` |

### Lower Bounds (no quantizer can beat)

| Measure | Bound |
|---|---|
| MSE | `D_mse >= 1/4^b` |
| Inner Product | `D_prod >= ||y||^2 / (d * 4^b)` |

### Gap

TurboQuant's MSE is within `sqrt(3*pi)/2 ≈ 2.72x` of optimal. For small bit-widths, the gap is tighter:

| b | TurboQuant MSE | Lower bound | Ratio |
|---|---|---|---|
| 1 | 0.36 | 0.25 | 1.44 |
| 2 | 0.117 | 0.0625 | 1.87 |
| 3 | 0.03 | 0.0156 | 1.92 |
| 4 | 0.009 | 0.0039 | 2.31 |

---

## Comparison with Other Methods

| Method | Type | Preprocessing | Bit-width efficiency | Unbiased IP |
|---|---|---|---|---|
| TurboQuant | Online | None (data-oblivious) | Near-optimal | Yes (prod variant) |
| PQ (Product Quantization) | Offline | k-means codebook training | Good but needs calibration | No |
| RabitQ | Online | None | Suboptimal theoretical bounds | Unclear |
| KIVI | Online | None | Scalar quantization, no guarantees | No |
| PolarQuant | Online | Polar transform | Good | Yes |
| QJL (standalone) | Online | None | 1-bit only | Yes |

**Key advantage**: TurboQuant requires essentially **zero indexing time** (Table 2 in paper: 0.0007-0.0021 seconds vs 37-494 seconds for PQ and 597-3957 seconds for RabitQ).

---

## Application: KV Cache Quantization

### How it applies

In transformer attention, queries `q` compute inner products with keys `k` from the KV cache:

```
attention_score = <q, k> / sqrt(d_head)
```

TurboQuant_prod quantizes the keys/values as they are generated (online), and the dequantized versions preserve inner product accuracy.

### Results from paper

- **Needle-in-a-haystack**: Perfect recall (0.997) matching full-precision, even at 4x+ compression
- **LongBench**: At 3.5 bits, average score matches full-precision (50.06 vs 50.06 for Llama-3.1-8B)
- **At 2.5 bits**: Only marginal degradation (49.44 vs 50.06), with 5x+ compression

---

## Experimental Validation Dataset

The paper validates using:
- **DBpedia Entities** encoded with OpenAI embeddings (d=1536 and d=3072)
- **GloVe** embeddings (d=200) for lower-dimensional testing
- 100,000 training vectors + 1,000 query vectors
- Hardware: single NVIDIA A100 GPU
