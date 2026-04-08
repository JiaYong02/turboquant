# TurboQuant Implementation Reference

Step-by-step implementation guide with exact formulas, edge cases, and testing strategies. This document should be sufficient to implement TurboQuant without referring back to the paper.

---

## Module Structure

```
turboquant/
├── core/
│   ├── rotation.py          # Random rotation matrix generation
│   ├── codebook.py          # Lloyd-Max codebook computation
│   ├── qjl.py               # QJL (1-bit quantizer)
│   ├── turboquant_mse.py    # Algorithm 1: MSE-optimal quantizer
│   └── turboquant_prod.py   # Algorithm 2: Inner-product-optimal quantizer
├── utils/
│   ├── norms.py             # Norm handling for non-unit vectors
│   └── math_helpers.py      # Beta PDF, numerical integration
└── tests/
    ├── test_rotation.py
    ├── test_codebook.py
    ├── test_qjl.py
    ├── test_mse.py
    ├── test_prod.py
    └── test_distortion_bounds.py
```

---

## 1. Random Rotation Matrix (rotation.py)

### Interface

```python
class RandomRotation:
    def __init__(self, d: int, seed: int | None = None):
        """Generate and store a d x d random orthogonal matrix."""

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation: y = Pi @ x. Supports batched input (n, d)."""

    def unrotate(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation: x = Pi^T @ y. Supports batched input (n, d)."""
```

### Implementation Details

```python
def _generate(self, d, seed):
    rng = np.random.RandomState(seed)
    A = rng.randn(d, d)
    Pi, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1), not reflection
    # Standard approach: multiply columns by sign of diagonal of R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Pi = Pi * signs[np.newaxis, :]
    return Pi
```

**Why sign correction**: `np.linalg.qr` can produce reflections (det = -1). While this doesn't affect the algorithm's correctness (reflections are also isometries), sign correction ensures a proper uniform random rotation from the Haar measure on O(d).

### Batched Operations

For `n` vectors stacked as `(n, d)`:
```python
def rotate(self, x):
    # x: (d,) or (n, d)
    return x @ self.Pi.T  # equivalent to Pi @ x for each row
```

### Testing Criteria

1. `Pi @ Pi^T ≈ I` (orthogonality)
2. `||Pi @ x|| ≈ ||x||` for any x (norm preservation)
3. Coordinates of `Pi @ x` for random unit x should follow Beta distribution (test with K-S test)
4. `unrotate(rotate(x)) ≈ x` (round-trip)

---

## 2. Codebook Computation (codebook.py)

### Interface

```python
class LloydMaxCodebook:
    def __init__(self, d: int, b: int):
        """Compute optimal centroids for dimension d, bit-width b."""

    @property
    def centroids(self) -> np.ndarray:
        """Sorted array of 2^b centroid values."""

    @property
    def boundaries(self) -> np.ndarray:
        """Sorted array of 2^b + 1 boundary values (including -1 and 1)."""

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Map scalar values to nearest centroid indices."""

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Map indices back to centroid values."""
```

### Beta Distribution PDF

Exact form (Lemma 1):
```python
def beta_pdf(x, d):
    from scipy.special import gammaln
    log_coeff = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    coeff = np.exp(log_coeff)
    return coeff * (1 - x**2) ** ((d - 3) / 2)
```

Use `gammaln` (log-gamma) to avoid overflow for large `d`.

For `d >= 50`, the Gaussian approximation `N(0, 1/d)` is very accurate:
```python
def gaussian_approx_pdf(x, d):
    return np.sqrt(d / (2 * np.pi)) * np.exp(-d * x**2 / 2)
```

### Lloyd-Max Algorithm

```python
def compute_codebook(d, b, max_iter=1000, tol=1e-12):
    n_centroids = 2 ** b
    # Initialize with quantiles of the distribution
    from scipy.integrate import quad

    pdf = lambda x: beta_pdf(x, d)

    # Better initialization: use quantiles
    from scipy.stats import norm
    sigma = 1.0 / np.sqrt(d)
    quantiles = norm.ppf(np.linspace(0.5/n_centroids, 1 - 0.5/n_centroids, n_centroids), scale=sigma)
    centroids = np.clip(quantiles, -0.999, 0.999)

    for iteration in range(max_iter):
        # Step 1: Compute Voronoi boundaries
        boundaries = np.zeros(n_centroids + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_centroids - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        # Step 2: Update centroids (conditional expectation)
        new_centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = (lo + hi) / 2
                continue
            numerator, _ = quad(lambda x: x * pdf(x), lo, hi)
            denominator, _ = quad(lambda x: pdf(x), lo, hi)
            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return np.sort(centroids)
```

### Fast Quantization (nearest centroid lookup)

```python
def quantize_scalar(values, centroids):
    """Assign each value to nearest centroid index. Vectorized."""
    # values: (n,) or (n, d)  -- scalar values
    # centroids: (2^b,) sorted
    # Use searchsorted on boundaries for O(log(2^b)) = O(b) per element
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    indices = np.searchsorted(boundaries, values)
    return indices
```

### Expected Codebook Values

For verification, at large d:

| b | Centroids (times sqrt(d)) |
|---|---|
| 1 | [-0.7979, +0.7979] (i.e., -sqrt(2/pi), +sqrt(2/pi)) |
| 2 | [-1.510, -0.453, +0.453, +1.510] |

### Testing Criteria

1. Centroids are sorted and within [-1, 1]
2. For b=1, centroids should be approximately `±sqrt(2/(pi*d))`
3. MSE cost `d * C(f_X, b)` matches expected values (0.36, 0.117, 0.03, 0.009 for b=1,2,3,4)
4. Codebook is symmetric: `c_i = -c_{2^b - 1 - i}` (due to symmetry of Beta distribution)

---

## 3. QJL Transform (qjl.py)

### Interface

```python
class QJL:
    def __init__(self, d: int, seed: int | None = None):
        """Generate d x d random projection matrix S ~ N(0,1)."""

    def quantize(self, r: np.ndarray) -> np.ndarray:
        """Compute sign(S @ r). Returns {-1, +1}^d."""

    def dequantize(self, signs: np.ndarray, gamma: float) -> np.ndarray:
        """Compute sqrt(pi/2)/d * gamma * S^T @ signs."""
```

### Implementation

```python
class QJL:
    def __init__(self, d, seed=None):
        rng = np.random.RandomState(seed)
        self.S = rng.randn(d, d)
        self.d = d
        self._scale = np.sqrt(np.pi / 2) / d

    def quantize(self, r):
        """r: (d,) or (n, d). Returns sign vector in {-1, +1}."""
        projected = r @ self.S.T  # (n, d) or (d,)
        return np.sign(projected).astype(np.int8)  # zeros become +1 or handle separately

    def dequantize(self, signs, gamma):
        """signs: {-1,+1}^d or (n, d). gamma: scalar or (n,)."""
        # x_qjl = sqrt(pi/2)/d * gamma * S^T @ signs
        if signs.ndim == 1:
            return self._scale * gamma * (self.S.T @ signs.astype(np.float64))
        else:
            # Batched: gamma is (n,)
            return self._scale * gamma[:, np.newaxis] * (signs.astype(np.float64) @ self.S)
```

**Edge case**: `sign(0) = 0` in numpy. Handle by mapping zeros to +1:
```python
signs = np.sign(projected)
signs[signs == 0] = 1
```

### Testing Criteria

1. **Unbiasedness**: Over many random S matrices, `E[<y, dequantize(quantize(x), ||x||)>] ≈ <y, x>` for unit x
2. **Variance bound**: `Var(<y, dequantized>) <= pi/(2*d) * ||y||^2`
3. Output of quantize is always in {-1, +1}

---

## 4. TurboQuant_mse (turboquant_mse.py)

### Interface

```python
class TurboQuantMSE:
    def __init__(self, d: int, b: int, seed: int | None = None):
        """Initialize with dimension, bit-width, and optional seed."""

    def quantize(self, x: np.ndarray) -> dict:
        """
        Quantize vector(s).
        Input: x of shape (d,) or (n, d), assumed unit norm.
        Returns: {'idx': array of shape (d,) or (n,d), dtype uint8/uint16}
        """

    def dequantize(self, quantized: dict) -> np.ndarray:
        """
        Reconstruct vector(s).
        Returns: x_tilde of shape (d,) or (n, d)
        """

    def quantize_with_norm(self, x: np.ndarray) -> dict:
        """For non-unit vectors. Also stores and restores norms."""
```

### Implementation Pseudocode

```python
class TurboQuantMSE:
    def __init__(self, d, b, seed=None):
        self.d = d
        self.b = b
        self.rotation = RandomRotation(d, seed=seed)
        self.codebook = LloydMaxCodebook(d, b)

    def quantize(self, x):
        y = self.rotation.rotate(x)                    # (n, d) or (d,)
        idx = self.codebook.quantize(y)                # nearest centroid indices
        return {'idx': idx}

    def dequantize(self, quantized):
        idx = quantized['idx']
        y_tilde = self.codebook.dequantize(idx)        # centroid lookup
        x_tilde = self.rotation.unrotate(y_tilde)      # inverse rotation
        return x_tilde
```

### Testing Criteria

1. **MSE check**: `mean(||x - dequantize(quantize(x))||^2)` over many random unit vectors should match expected D_mse values
2. **Round-trip shape**: output shape matches input shape
3. **Determinism**: same seed produces same quantization

---

## 5. TurboQuant_prod (turboquant_prod.py)

### Interface

```python
class TurboQuantProd:
    def __init__(self, d: int, b: int, seed: int | None = None):
        """
        Initialize inner-product-optimal quantizer.
        Uses (b-1) bits for MSE stage + 1 bit for QJL stage.
        """

    def quantize(self, x: np.ndarray) -> dict:
        """
        Returns: {
            'idx': MSE indices at (b-1) bits,
            'qjl': sign vector {-1,+1}^d,
            'gamma': ||residual||_2 scalar
        }
        """

    def dequantize(self, quantized: dict) -> np.ndarray:
        """Reconstruct: x_tilde_mse + x_tilde_qjl"""
```

### Implementation

```python
class TurboQuantProd:
    def __init__(self, d, b, seed=None):
        assert b >= 1, "Bit-width must be >= 1"
        self.d = d
        self.b = b
        # MSE stage uses (b-1) bits
        # For b=1, MSE stage uses 0 bits (no MSE quantization, pure QJL)
        if b > 1:
            self.mse = TurboQuantMSE(d, b - 1, seed=seed)
        else:
            self.mse = None
        self.qjl = QJL(d, seed=(seed + 1 if seed is not None else None))

    def quantize(self, x):
        if self.mse is not None:
            mse_q = self.mse.quantize(x)
            x_tilde_mse = self.mse.dequantize(mse_q)
        else:
            mse_q = None
            x_tilde_mse = np.zeros_like(x)

        r = x - x_tilde_mse                           # residual
        gamma = np.linalg.norm(r, axis=-1)             # scalar or (n,)
        # sign() is scale-invariant, so we pass raw r (not normalized)
        qjl_signs = self.qjl.quantize(r)

        return {
            'mse': mse_q,
            'qjl': qjl_signs,
            'gamma': gamma
        }

    def dequantize(self, quantized):
        if quantized['mse'] is not None:
            x_tilde_mse = self.mse.dequantize(quantized['mse'])
        else:
            x_tilde_mse = 0.0

        # QJL dequantization with gamma scaling
        x_tilde_qjl = self.qjl.dequantize(quantized['qjl'], quantized['gamma'])
        return x_tilde_mse + x_tilde_qjl
```

**IMPORTANT**: The paper's QJL definition (Definition 1) applies to vectors on the unit sphere. When applying QJL to the residual `r` which is NOT unit norm, the paper stores `gamma = ||r||_2` separately and applies QJL to the direction of `r`. The dequantization then scales by gamma. Alternatively, you can apply QJL directly to `r` and absorb the norm into the dequantization formula -- the paper uses the latter approach in Algorithm 2 line 7 (`qjl = sign(S * r)`, not `sign(S * r/||r||)`).

Let me clarify the exact formulation from Algorithm 2:

```
# Quantize (Algorithm 2, lines 5-8):
idx = Quant_mse(x)
r = x - DeQuant_mse(idx)
qjl = sign(S * r)              # S applied to raw (non-normalized) residual
output: (idx, qjl, ||r||_2)

# Dequantize (Algorithm 2, lines 9-12):
x_tilde_mse = DeQuant_mse(idx)
x_tilde_qjl = sqrt(pi/2) / d * gamma * S^T * qjl
output: x_tilde_mse + x_tilde_qjl
```

Note that `gamma = ||r||_2` is stored, but the QJL is applied to the raw `r` (not `r/||r||`). The dequantization uses `gamma` as a scaling factor. This works because:
- `sign(S * r) = sign(S * (gamma * r_hat))` = `sign(S * r_hat)` (sign is scale-invariant)
- The dequantization then reconstructs using the stored `gamma`

### Testing Criteria

1. **Unbiasedness**: `E[<y, dequantize(quantize(x))>] ≈ <y, x>` (averaged over randomness)
2. **Inner product distortion**: `E[|<y,x> - <y,x_tilde>|^2] ≈ D_prod` values from table
3. **b=1 special case**: No MSE stage, pure QJL
4. **MSE bias check**: verify that MSE-only IS biased (`E[<y,x_tilde_mse>] != <y,x>`)

---

## 6. Testing Strategy

### Unit Tests

```python
# Test 1: Rotation preserves norms
x = random_unit_vector(d=128)
y = rotation.rotate(x)
assert np.allclose(np.linalg.norm(y), np.linalg.norm(x))

# Test 2: Codebook symmetry
codebook = LloydMaxCodebook(d=128, b=2)
c = codebook.centroids
assert np.allclose(c, -c[::-1], atol=1e-6)

# Test 3: QJL unbiasedness (statistical)
d = 256
n_trials = 10000
x = random_unit_vector(d)
y = random_unit_vector(d)
estimates = []
for _ in range(n_trials):
    qjl = QJL(d)
    signs = qjl.quantize(x)
    x_hat = qjl.dequantize(signs, np.linalg.norm(x))
    estimates.append(np.dot(y, x_hat))
mean_estimate = np.mean(estimates)
true_ip = np.dot(y, x)
assert abs(mean_estimate - true_ip) < 0.01  # should be unbiased

# Test 4: MSE distortion matches theory
d = 512
n_vectors = 1000
b = 2
mse_q = TurboQuantMSE(d, b)
errors = []
for _ in range(n_vectors):
    x = random_unit_vector(d)
    q = mse_q.quantize(x)
    x_hat = mse_q.dequantize(q)
    errors.append(np.sum((x - x_hat)**2))
empirical_mse = np.mean(errors)
expected_mse = 0.117  # for b=2
assert abs(empirical_mse - expected_mse) / expected_mse < 0.1  # within 10%

# Test 5: Prod unbiasedness
d = 256
n_trials = 5000
b = 2
x = random_unit_vector(d)
y = np.random.randn(d)
estimates = []
for _ in range(n_trials):
    prod_q = TurboQuantProd(d, b)
    q = prod_q.quantize(x)
    x_hat = prod_q.dequantize(q)
    estimates.append(np.dot(y, x_hat))
mean_est = np.mean(estimates)
true_ip = np.dot(y, x)
assert abs(mean_est - true_ip) < 0.05  # unbiased

# Test 6: Prod inner product distortion
variance = np.var(estimates)
expected_var = 0.56 / d * np.sum(y**2)  # D_prod for b=2
assert abs(variance - expected_var) / expected_var < 0.2  # within 20%
```

### Integration Tests

1. **Batch quantization**: Quantize 1000 vectors at once, verify shapes and distortion
2. **Different bit-widths**: Test b = 1, 2, 3, 4 and verify distortion matches expected values
3. **Different dimensions**: Test d = 64, 128, 256, 512, 1024, 1536
4. **Non-unit vectors**: Test norm preservation workflow
5. **Seed determinism**: Same seed produces identical results

### Benchmarks to Reproduce from Paper

1. **Distortion vs bit-width** (Figure 3): Plot D_mse and D_prod vs b, compare with upper/lower bounds
2. **Bias of MSE quantizer** (Figure 1): Show that MSE quantizer shifts inner product distribution, while prod is centered at zero
3. **Near-neighbor recall** (Figure 5): Quantize a dataset, compute approximate inner products, measure recall@k

---

## 7. Numerical Precision Notes

1. **Codebook computation**: Use `scipy.integrate.quad` with sufficient precision. The integral bounds near -1 and 1 can cause issues due to `(1-x^2)^((d-3)/2)` vanishing.
2. **Large d**: For d > 100, use the Gaussian approximation for the PDF to avoid numerical issues with the Beta distribution.
3. **Random matrix storage**: For reproducibility, store the seed rather than the matrix. Regenerate from seed when needed.
4. **Sign of zero**: `np.sign(0.0) = 0.0`. Map to +1 (or -1, just be consistent).
5. **Gamma near zero**: When the MSE reconstruction is very accurate, gamma can be tiny. This is fine -- the QJL correction will also be tiny.

---

## 8. GPU/Accelerator Considerations

All operations in TurboQuant are **embarrassingly parallelizable**:

1. **Matrix multiplication** (`Pi @ x`, `S @ r`, `S^T @ qjl`): Standard BLAS/cuBLAS
2. **Nearest centroid lookup**: `searchsorted` or parallel comparison
3. **Sign function**: Element-wise
4. **Norm computation**: Standard reduction

No data-dependent branching, no iterative search, no k-means at quantize time. This is what makes TurboQuant accelerator-friendly.

For PyTorch implementation:
```python
# Rotation
y = torch.matmul(x, Pi.T)

# Quantize (vectorized nearest centroid)
centroids = torch.tensor(codebook.centroids)  # (2^b,)
distances = torch.abs(y.unsqueeze(-1) - centroids)  # (..., d, 2^b)
idx = torch.argmin(distances, dim=-1)  # (..., d)

# Dequantize
y_tilde = centroids[idx]
x_tilde = torch.matmul(y_tilde, Pi)  # Pi^T = Pi inverse for orthogonal Pi

# QJL
qjl = torch.sign(torch.matmul(r, S.T))
x_qjl = (np.sqrt(np.pi/2) / d) * gamma.unsqueeze(-1) * torch.matmul(qjl.float(), S)
```
