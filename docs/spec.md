# TurboQuant Specification

> Based on "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
> (Zandieh et al., Google Research / NYU / Google DeepMind, April 2025)

## Vision

TurboQuant is a data-oblivious vector quantization library that compresses
high-dimensional floating-point vectors into low-bitwidth integers with
near-optimal distortion. Starting from a NumPy reference implementation
(phases 1-6), the project evolves through a PyTorch GPU backend, HuggingFace
KV cache integration, and evaluation against published results, culminating
in a production-ready llama.cpp fork with native GGML quantization types.

**Target hardware**: NVIDIA RTX 5000 Ada (32 GB VRAM)
**Target models**: Llama-3.1-8B, Gemma-7B (7B-8B class)
**Timeline**: No external deadline. Quality over speed.

---

## Phase Overview

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Project scaffolding & core math | Done | pyproject.toml, utils.py, rotation.py |
| 2 | Lloyd-Max codebook | Done | codebook.py with scipy integration |
| 3 | QJL transform | Done | qjl.py, 1-bit unbiased quantizer |
| 4 | TurboQuantMSE (Algorithm 1) | Done | quantizer_mse.py, MSE-optimal pipeline |
| 5 | TurboQuantProd (Algorithm 2) | Done | quantizer_prod.py, unbiased inner product |
| 6 | Distortion bounds & packaging | Done | test_distortion_bounds.py, README, __init__ |
| 7 | PyTorch backend | Done | Port all modules to torch with GPU support |
| 8 | KV cache integration | Done | Architecture-agnostic HuggingFace cache hook |
| 9 | Fused decompress+attention Triton kernel | Done | Single-kernel decode, compressed-KV read |
| 10 | FlashDecoding split-KV | Done | Split-K parallelism, folded-rotation trick |
| 11a | Cached per-Q-head rotations | Done | Eliminate per-call index_select + fp32 cast |
| 11b | CUDA-graph capture | Done | Slab KV, device-scalar S_total, captured decode |
| 12 | Rotation fusion (A/B) + fp16 norms (Phase A) | Done | Fused Q/Pi_v rotations, fp16 gamma/norms. Speed DoD parked as future work |
| 13 | Bit-tight packing | Done | Byte-crossing packer for odd bit widths (3/5/6); unlocks paper-level compression |
| 14 | Outlier-channel split | Done | Fractional avg bit-widths (2.5/3.5) via per-head outlier perm + two buckets |
| 15 | Evaluation suite | Planned | LongBench + perplexity benchmarks |
| 16 | llama.cpp fork | Planned | Native GGML_TYPE_TQ{b} quantization types |
| 17+ | Production hardening | Planned | CI/CD, docs, H100/FP8 |

### Dependency Graph

```
Phase 7 (PyTorch) --> Phase 8 (KV Cache) --> Phase 9 (Fused kernel) --> Phase 10 (Split-KV)
                                                                   \-> Phase 11a (Cached rotations)
                                                                   \-> Phase 11b (CUDA graph)
                                                                   \-> Phase 12 (Rotation fusion) --> Phase 13 (Bit-tight packing)
                                                                                                  \-> Phase 14 (Outlier-channel split)
                                                                                                  \-> Phase 15 (Eval)
                                                                                                  \-> Phase 16 (llama.cpp)
                                                                                                  \-> Phase 17+ (Hardening)
```

---

## Phase 7: PyTorch Backend

### Goal

Port all five core classes to PyTorch while keeping the NumPy implementation
as the reference. Same API, same numerical behavior (within float precision),
with a `device` parameter for CPU/CUDA placement.

### Architecture

Parallel class hierarchy -- users import the variant they need:

```python
from turboquant import TurboQuantMSE           # NumPy (existing)
from turboquant import TurboQuantMSETorch       # PyTorch (new)
```

### New Files

```
src/turboquant/
    backends/
        __init__.py             # Backend enum, detection helpers
        numpy_ops.py            # Thin wrappers around existing NumPy calls
        torch_ops.py            # PyTorch equivalents
    rotation_torch.py           # RandomRotationTorch
    codebook_torch.py           # LloydMaxCodebookTorch
    qjl_torch.py                # QJLTorch
    quantizer_mse_torch.py      # TurboQuantMSETorch
    quantizer_prod_torch.py     # TurboQuantProdTorch
tests/
    test_rotation_torch.py
    test_codebook_torch.py
    test_qjl_torch.py
    test_quantizer_mse_torch.py
    test_quantizer_prod_torch.py
```

### Modified Files

- `src/turboquant/__init__.py` -- export Torch variants
- `pyproject.toml` -- no changes needed (`gpu` extra already declares `torch>=2.0`)

### Key Design Decisions

**Codebook precomputation stays on CPU.** The Lloyd-Max algorithm uses
`scipy.integrate.quad`, which has no PyTorch equivalent. Centroids and
boundaries are computed once via scipy, then transferred to the target
device as `torch.Tensor`. `LloydMaxCodebookTorch` shares the CPU-side
class-level cache with `LloydMaxCodebook` to avoid redundant computation.

**Rotation matrix generated on CPU, moved to device.** `torch.linalg.qr`
exists but the matrix is generated once at init. The sign-correction step
(`signs = torch.sign(torch.diag(R))`) works identically to NumPy.

**QJL projection matrix.** For head_dim=128, S is 128x128 = 64 KB in
float32. Negligible memory per quantizer instance.

**Seed compatibility.** PyTorch and NumPy RNGs produce different values
for the same seed. `TurboQuantMSETorch(d=128, b=4, seed=42)` uses a
different rotation matrix than `TurboQuantMSE(d=128, b=4, seed=42)`.
This is acceptable -- the algorithm's correctness is independent of the
specific rotation (any Haar-distributed rotation works). Within-backend
reproducibility is maintained.

**Float32 default.** PyTorch defaults to float32; NumPy to float64.
Use float32 for all PyTorch ops (matches inference use case). Verify
that float32 precision is sufficient for codebook discrimination at b<=4.

### Op Mapping

| NumPy | PyTorch |
|-------|---------|
| `np.linalg.qr(A)` | `torch.linalg.qr(A)` |
| `np.searchsorted(b, v)` | `torch.searchsorted(b, v)` |
| `np.sign(x)` | `torch.sign(x)` |
| `np.linalg.norm(x, axis=-1)` | `torch.linalg.norm(x, dim=-1)` |
| `x @ M.T` | `x @ M.T` |
| `RandomState(seed).randn(d,d)` | `Generator().manual_seed(seed)` + `torch.randn(d,d, generator=gen)` |

### Testing Strategy

- Mirror every existing NumPy test class with a Torch variant.
- Cross-backend statistical tests: both backends should produce similar
  MSE distributions (not identical values, but statistically equivalent).
- GPU device tests (`@pytest.mark.skipif(not torch.cuda.is_available())`):
  verify device placement, no silent CPU-GPU transfers.
- Benchmark: time comparison NumPy vs PyTorch-CPU vs PyTorch-CUDA for
  batch of 1000 vectors at d=128, b=4.

### Risks

- `torch.searchsorted` edge behavior may differ from NumPy at boundary
  values. Mitigate with explicit boundary condition tests.
- Float32 codebook centroids are closer together at high bit-widths.
  If b=4 shows precision issues, selectively use float64 for codebook
  boundaries only.

### Definition of Done

- All 5 core classes have PyTorch equivalents.
- PyTorch quantizers pass distortion bound tests (within statistical tolerance).
- GPU round-trip: CUDA tensor in -> quantized on CUDA -> dequantized on CUDA.
- No scipy dependency at quantize/dequantize time (only at codebook init).

---

## Phase 8: KV Cache Integration

### Goal

Architecture-agnostic HuggingFace integration that intercepts KV cache
writes, quantizes keys with TurboQuantProd and values with TurboQuantMSE,
and transparently dequantizes on read. Works with any model that uses the
standard HF cache interface (Llama, Gemma, Mistral, etc.).

### New Files

```
src/turboquant/
    integrations/
        __init__.py
        hf_cache.py             # TurboQuantCache, TurboQuantCacheLayer
        memory_tracker.py       # VRAM tracking utilities
tests/
    test_hf_cache.py
```

### Modified Files

- `src/turboquant/__init__.py` -- export `TurboQuantCache`
- `pyproject.toml` -- add `[project.optional-dependencies] hf`

### Architecture

HuggingFace transformers (v4.51+) uses a layered cache architecture.
`DynamicCache` stores raw tensors; `QuantizedCache` demonstrates the
quantized pattern. TurboQuant follows the same contract.

**TurboQuantCacheLayer** (per transformer layer):

```
Stores compressed K/V for one layer.

update(key_states, value_states) -> (all_keys, all_values)
  1. For each new position in seq_len dimension:
     - Keys:   quantize per head with TurboQuantProdTorch (unbiased IP)
     - Values: quantize per head with TurboQuantMSETorch
  2. Dequantize ALL stored positions, return concatenated dense tensors.
```

**TurboQuantCache** (top-level, replaces DynamicCache):

```python
class TurboQuantCache(Cache):
    def __init__(self, config: PretrainedConfig,
                 key_bits: int = 4, value_bits: int = 4,
                 seed: int = 42, device: torch.device = None):
        # Extracts head_dim, num_kv_heads, num_layers from config
        # Creates one TurboQuantCacheLayer per transformer layer
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.integrations.hf_cache import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", ...)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
cache = TurboQuantCache(model.config, key_bits=4, value_bits=4)
outputs = model.generate(**inputs, past_key_values=cache, use_cache=True)
```

### Key Design Decisions

**Per-head quantization.** Keys and values are quantized per attention head
(dim = head_dim, typically 128 for Llama-3.1-8B). Each head gets its own
rotation matrix and QJL matrix, seeded deterministically from
`seed + layer_idx * 1000 + head_idx`.

**GQA support.** Llama-3.1-8B uses GQA with num_kv_heads=8 and
num_attention_heads=32 (4:1 ratio). The cache stores num_kv_heads
compressed vectors per position. The HF model handles GQA repeat/expansion
after the cache returns dequantized tensors. The cache layer is
GQA-agnostic by construction.

**Norm handling.** KV cache vectors are NOT unit-norm. The existing
`quantize_with_norm` / `dequantize_with_norm` methods handle this --
each quantized vector stores its L2 norm as a float32 scalar.

**Full dequantization every step.** The `update()` method returns all K/V
as dense tensors for attention computation. This matches how HF's
`QuantizedCache` works. Incremental attention (computing directly from
compressed K/V) is a Phase 11+ optimization.

**Configurable per-layer bit-widths.** Accept an optional dict mapping
`layer_idx -> {"key_bits": int, "value_bits": int}` for mixed-precision
experiments (e.g., 2.5-bit, 3.5-bit outlier strategies from the paper).

### Memory Analysis

For Llama-3.1-8B (num_kv_heads=8, head_dim=128) at b=4, per token:

| Component | FP16 | TurboQuant b=4 |
|-----------|------|----------------|
| Key per head | 256 bytes | 68 bytes (b*d bits + norm + gamma) |
| Value per head | 256 bytes | 68 bytes (b*d bits + norm) |
| Total per layer (8 heads) | 4096 bytes | 1088 bytes |
| Total per token (32 layers) | 128 KB | 34 KB |
| **Compression ratio** | 1x | **3.76x** |

For 4096 token context: FP16 = 512 MB, TurboQuant b=4 = 136 MB.
Savings: 376 MB of VRAM.

### Dependencies

- Phase 7 (PyTorch backend) must be complete.
- `transformers>=4.51` and `accelerate>=0.20` as optional `[hf]` deps.

### Testing Strategy

- Unit tests with mock tensors (no model): create TurboQuantCacheLayer,
  call `update()` repeatedly, verify output shapes and reconstruction quality.
- Integration test with a small model (TinyLlama-1.1B or 2-layer toy):
  compare generation quality DynamicCache vs TurboQuantCache.
- Memory tracking test: verify VRAM reduction for long sequences (>256 tokens).
- GQA correctness: verify when num_kv_heads != num_attention_heads.

### Risks

- **HF API instability.** The cache API has changed across transformers
  versions. Mitigate: pin `transformers>=4.51`, test the layered cache API.
- **Dequantization latency.** Full cache dequant every step could bottleneck
  throughput for long sequences. This is expected and acceptable for Phase 8.
  Profile and document the tradeoff. Optimization is Phase 11+.
- **Prefill batch shapes.** During prompt encoding, all tokens arrive at once.
  Verify the quantizer handles `(batch, num_kv_heads, seq_len, head_dim)`.

### Definition of Done

- `TurboQuantCache` works as a drop-in for `past_key_values` in HF `generate()`.
- Works with Llama-3.1-8B and Gemma without model-specific code.
- Memory usage is measurably lower than FP16 DynamicCache for seq > 256.
- Generated text is coherent at b=4 (qualitative check).

---


## Phase 9 (revised): Fused Decompress-Attention Triton Kernel

### Goal

Implement the paper's actual speed mechanism: a fused decompress+attention
Triton kernel that reads compressed K/V tiles directly from HBM and
decompresses them in SRAM/registers, so the GPU attention kernel reads
**b/16 × fewer bytes** from HBM compared to FP16. This is what the paper
calls the HBM↔SRAM bandwidth reduction that enables ≥FP16 throughput.

**Status**: Planned (after Phase 8 KV Cache Integration is validated)

### Why the current drop-in HF cache cannot match FP16 speed

Autoregressive decode attention is memory-bandwidth-bound (HBM reads, not
FLOPs). The speed argument is:

```
FP16 decode attention: reads B×H×S×D×2 bytes per layer from HBM
TurboQuant b-bit:      reads B×H×S×D×(b/8) bytes, decompresses in SRAM → b/16× reads
```

The drop-in `TurboQuantCache.update()` currently holds an fp16 dense buffer
(`_dense_keys`/`_dense_values`) in VRAM and returns it to HF's stock attention
kernel. Total HBM traffic = compressed (write path) + fp16 dense (read path),
which is **more** than FP16 baseline — not less. The dense buffer is needed
for functional correctness and to validate math; real speed requires the fused
kernel below.

### Architecture

The kernel fuses two operations into one GPU kernel:

```
Standard (two kernels):
  1. K/V decompress: HBM[compressed] → SRAM → HBM[fp16]   ← extra HBM round-trip
  2. Attention:      HBM[Q] + HBM[fp16 K] + HBM[fp16 V] → HBM[O]

Fused (one kernel):
  1. Load Q tile from HBM to SRAM (unchanged)
  2. Load compressed K/V tile from HBM to SRAM (b/16× smaller)
  3. Decompress K/V tile in SRAM/registers (no HBM write)
  4. Compute attention QK^T + softmax + V in SRAM (FlashAttention-2 tiling)
  5. Write output O tile to HBM (unchanged)
```

Step 3 consists of: index lookup in codebook (shared memory), QJL sign
reconstruction (bitwise), rotation unproject (register-level matmul). All
happen before the compressed tile ever leaves SRAM.

### Implementation Plan

**Phase 9a: Validate math with PyTorch reference kernel**

Before touching Triton, validate that the compressed K/V representation
produces correct attention outputs using a pure-PyTorch reference:

```python
def compressed_attention_reference(q, key_store, value_store, key_quantizer, value_quantizer):
    """Dequantize inside attention loop (reference, not fused)."""
    # Tile over sequence blocks, dequantize one block at a time
    # Verify output matches standard fp16 attention + full dequantize
```

Tests: `TestCompressedAttentionReference` — compare against
`F.scaled_dot_product_attention` with full dequantized K/V. Accept atol=1e-3.

**Phase 9b: Triton fused kernel (FlashAttention-2 base)**

Start from the FlashAttention-2 Triton reference implementation
(`flash_attn_triton.py` in the FA2 repo). Key modifications:

| FA2 location | TurboQuant change |
|---|---|
| `tl.load(K_block_ptr, ...)` | Load compressed K block instead |
| After K load | Dequantize in-register: codebook lookup + QJL sign restore + rotate |
| `tl.load(V_block_ptr, ...)` | Load compressed V block |
| After V load | Dequantize in-register: codebook lookup + norm rescale |
| Block pointer stride | `b/16` fraction of FP16 stride |

Triton primitives to use:
- `tl.load` with `cache_modifier=".cg"` (cache global, skip L2 for streaming)
- Codebook in `tl.constexpr` shared memory (small: 16 centroids × 4 bytes = 64 bytes)
- Sign unpack: `(packed >> bit_offset) & 1` per lane
- In-register rotation: `tl.dot(x, Pi_T)` where `Pi_T` is a register tile

### New Files

```
src/turboquant/
    kernels/
        __init__.py
        decompress_attn.py          # Triton fused kernel (forward pass)
        decompress_attn_ref.py      # Pure-PyTorch reference for validation
        codebook_cache.py           # Shared codebook constant management
tests/
    test_decompress_attn.py         # Reference vs full-dequant, and Triton vs reference
scripts/
    benchmark_fused_kernel.py       # Measure tok/s: FP16 vs drop-in vs fused
```

### Modified Files

- `src/turboquant/integrations/hf_cache.py` — add optional `use_fused_kernel`
  flag to `TurboQuantCache`; when True, skip `_dense_keys`/`_dense_values`
  and invoke the fused kernel instead of returning dense tensors.
- `pyproject.toml` — add `[project.optional-dependencies] triton = ["triton>=2.2"]`

### Dependencies

- `triton>=2.2` (ships with PyTorch 2.1+ on Linux; Windows support via WSL
  or the triton-nightly wheel)
- CUDA-capable GPU (RTX 5000 Ada or better for best results)
- FlashAttention-2 Triton reference (`flash_attn` optional — kernel is
  reimplemented from scratch based on the paper's algorithm)

### Testing Strategy

1. **Reference kernel tests** (CPU-runnable, no Triton required):
   - `test_reference_matches_sdpa`: cosine similarity ≥ 0.999 between
     `compressed_attention_reference()` and `F.scaled_dot_product_attention`
     on random Q, K, V at b=4/3/2.
   - `test_reference_causal_mask`: verify causal masking is applied correctly.

2. **Triton kernel tests** (CUDA required):
   - `test_triton_matches_reference`: Triton output matches PyTorch reference
     within atol=1e-3.
   - `test_triton_all_bitwidths`: b=2/3/4 all produce correct results.

3. **Speed benchmark** (manual, not CI):
   - `scripts/benchmark_fused_kernel.py` measures tok/s for FP16, drop-in,
     and fused-kernel at sequence lengths 256/1024/4096.
   - Target: fused kernel ≥ 0.9× FP16 tok/s at seq=1024, b=4.

### Expected Performance

For RTX 5000 Ada (16 GB/s HBM bandwidth):

| Method | HBM bytes/step (L=32, H=8, D=128, S=1024) | Expected speedup |
|---|---|---|
| FP16 baseline | 2×8×1024×128×2 B × 32 layers | 1.0× |
| Drop-in TQ b=4 | fp16 read + compressed write | ~0.7× (slower) |
| **Fused TQ b=4** | **b/16 × fp16** read only | **~3–4×** |
| **Fused TQ b=3** | **b/16 × fp16** read only | **~4–5×** |

### Risks

- **Triton on Windows**: The stable `triton` wheel does not support Windows
  natively. Mitigation: develop in WSL2 on the RTX 5000 Ada machine; provide
  a Windows fallback that uses the drop-in (dense buffer) path automatically.
- **Rotation matmul in registers**: For head_dim=128, the in-register rotation
  is a 128×128 matmul per tile, which may overflow registers. Mitigation:
  investigate Hadamard transform (O(D log D), fully decomposable into butterfly
  stages) as the paper suggests for large head_dim.
- **Causal mask tiling**: FA2's tiling strategy handles causal masking at tile
  granularity. The decompression step must happen before the mask is applied
  (to avoid operating on zeroed-out tiles). This is a minor correctness detail
  to verify during kernel testing.

### Definition of Done

- `TestCompressedAttentionReference` passes: cosine similarity ≥ 0.999 at b=4.
- `TestTritonKernel` passes on CUDA (if available).
- `scripts/benchmark_fused_kernel.py` shows fused kernel ≥ 0.9× FP16 tok/s
  at seq=1024, b=4 on RTX 5000 Ada.
- `TurboQuantCache(config, use_fused_kernel=True)` runs `model.generate()`
  successfully with Llama-3.1-8B.

---

## Phase 10: FlashDecoding Split-KV (done)

### Goal

Extend the Phase 9 fused kernel with FlashDecoding v2 split-K parallelism so
decode saturates the GPU when `B * H_q` is smaller than the SM count.

### What Landed

- `_attn_split_kernel` — grid `(B, H_q, NUM_SPLITS)`, streaming softmax over
  `SPLIT_SIZE`-wide KV tiles; per-tile centroid gather + QJL sign reconstruct
  inline.
- `_attn_reduce_kernel` — grid `(B, H_q)`, merges `NUM_SPLITS` streaming-softmax
  states into a single canonical-V output via log-sum-exp.
- `_pick_split_size` — auto-tunes `SPLIT_SIZE` so grid covers ≥ 2× SM count at
  the target decode shape (132 SMs on H100).
- Folded-rotation trick — `S_k`, `Pi_k`, `Pi_v` per-Q-head matrices applied
  once via `torch.einsum` before the kernel; kernel consumes pre-rotated `Q_qjl`
  and `Q_rot`.

### Tests

`tests/test_decompress_attn_triton.py` — 15 parity tests across bit-widths
(b ∈ {2, 3, 4, 5}), shapes (B, H_q, H_kv, S, D), and GQA ratios.

### Outcome

Fused kernel becomes correctness-complete and auto-tuned across shapes. Decode
latency at B=1, H_q=32, S=1024, b=4 on H100 sits at ~221 μs eager — dominated
by CPU dispatch, not GPU work. Phase 11 addresses that.

---

## Phase 11a: Cached Per-Q-Head Rotations (done)

### Goal

Hoist the per-Q-head rotation cache (`S_k_per_q`, `Pi_k_per_q`, `Pi_v_per_q`)
out of the per-call hot path — these matrices are invariant for a cache layer's
lifetime.

### What Landed

- `TurboQuantCacheLayer._prepare_fused_state()` computes `kv_for_q`, per-Q-head
  `S_k_per_q`, `Pi_k_per_q`, `Pi_v_per_q`, and centroid LUTs (`_key_cent_lut`,
  `_val_cent_lut`) once on first decode; stores them on the layer.
- `fused_decompress_attention_triton` accepts these as optional kwargs and
  skips the cold `index_select(...).to(fp32).contiguous()` path when provided.

### Outcome

Eager fused path drops from ~450 μs to ~221 μs on the Phase 9 DoD shape. CPU
dispatch still dominates (~180 μs of the 221 μs — 3 einsum→bmm dispatches, 1
`q.to(fp32)` copy, 5 scratch allocations, 2 Triton launches, 1 output einsum).

---

## Phase 11b: CUDA-Graph Capture (done)

### Goal

Absorb the ~180 μs of CPU dispatch overhead that dominates eager decode, by
capturing the fused decode subgraph once per power-of-2 `S` bucket and
replaying on every step.

### What Landed

- **Pre-allocated KV slabs** on `_BatchedKeyStore` and `_BatchedValueStore`:
  `preallocate(max_seq_len, B)` replaces growing `cat` buffers with fixed
  slabs; `append()` writes into `[:, :, seq_len:seq_len+S_new, :]` and
  advances `_seq_len`. Keeps tensor data pointers stable across steps
  (the blocker for naïve graph capture).
- **Device-scalar `S_total`** in `_attn_split_kernel` and `_attn_reduce_kernel`
  — passed as a 0-d int32 tensor and loaded via `tl.load(s_total_ptr)`, so a
  single capture serves every `seq_len` within a bucket.
- **`_fused_graph_forward`** driver on `TurboQuantCacheLayer`:
  warmup-on-miss, capture under a shared
  `torch.cuda.graph_pool_handle()`, replay = 1 D2D copy
  (`q_static.copy_(q)`) + `graph.replay()`.
- **Bucketed capture cache** — one `_CapturedGraph` per
  `_s_bucket(seq_len) ∈ {256, 1024, 4096, 16384, 65536}` (≤ 6 captures per
  layer for `max_seq_len=65536`).
- **Invalidation** on `reset` / `crop` / `reorder_cache` — captures are
  dropped; re-capture on next decode is a one-shot ~200 μs cost.
- **HF attention routing** (`integrations/attention.py`): eligible decode
  steps (`S_q == 1`, fused kernel on, slab preallocated, no autograd, CUDA)
  route through `_fused_graph_forward`; everything else falls back to the
  eager fused path or SDPA.

### Files

| File | Change |
|------|--------|
| `src/turboquant/integrations/hf_cache.py` | Slab KV stores, `_fused_graph_forward`, `max_seq_len` / `use_cuda_graph` options, invalidation hooks |
| `src/turboquant/kernels/decompress_attn.py` | Device-scalar `S_total`, optional `scratch`/`s_total_tensor` passthrough |
| `src/turboquant/integrations/attention.py` | Route eligible decode to graph driver |
| `tests/test_fused_kernel_cuda_graph.py` (new) | Parity vs eager, bucket transitions, invalidation, determinism |
| `tests/test_fused_kernel_perf.py` | Graph perf gate (≤ 3.0× SDPA for regression detection) |
| `scripts/benchmark_fused_kernel.py` | `--cuda-graph` flag |

### Outcome (H100, B=1, H_q=32, H_kv=8, S=1024, D=128, b=4)

| Path | Latency | vs SDPA |
|------|---------|---------|
| FP16 SDPA (cuDNN) | 17.3 μs | 1.00× |
| Fused eager (pre-11b) | ~221 μs | 12.87× |
| **Fused + CUDA graph** | **40.5 μs** | **2.34×** |

5.3× eager speedup. CPU dispatch absorbed as designed. The residual 23 μs is
GPU-side: Triton kernel-launch latency + three rotation dispatches still live
on the PyTorch side of the captured graph. Phase 12 addresses those to close
Phase 9 DoD.

### Definition of Done

- 351 parity tests green (incl. 7 new graph-path tests).
- `TURBOQUANT_PERF=1 TURBOQUANT_CUDA_GRAPH=1` graph gate passes at ≤ 3.0×.
- `model.generate(..., past_key_values=TurboQuantCache(..., use_fused_kernel=True, max_seq_len=4096, use_cuda_graph=True))` runs end-to-end.

---

## Phase 12: Rotation Fusion + fp16 Norms (done, speed DoD parked)

### Goal

Close the 23 μs gap between captured-graph fused decode (40.5 μs) and FP16
SDPA (17.3 μs), meeting the Phase 9 DoD (fused decode ≥ FP16 SDPA tok/s at
S=1024, b=4).

### What Landed (commit `72ea573`)

- **12A (Q-side rotation fusion):** `q @ S_k` and `q @ Pi_k` mat-vecs moved
  into `_attn_split_kernel`. Eliminates the `q.to(fp32)` cast and two
  cuBLAS einsum launches.
- **12B (V-side rotation fusion):** `Pi_v` mat-vec moved into
  `_attn_reduce_kernel`; output written directly in canonical V-space at
  `in_dtype`. Eliminates the trailing einsum and dtype cast.
- **Phase A (fp16 norms):** `gamma` and K/V `norms` narrowed from fp32 to
  fp16 at storage time; kernel upcasts on load. Saves ~6 B/token of
  metadata.
- **Phase 12C (persistent kernel) skipped.** Profile revealed
  `_attn_split_kernel` alone is 23.9 μs (dequant-traffic bound), exceeding
  the 22 μs DoD target even if reduce were free. Residual gap is
  compute-bound (fp32 ALU) vs SDPA tensor cores. Merging reduce would save
  only ~3 μs.

### Outcome

| Path | Latency | vs SDPA |
|------|---------|---------|
| FP16 SDPA (cuDNN) | 17.3 μs | 1.00× |
| Fused + CUDA graph (pre-12) | 40.5 μs | 2.34× |
| **Fused + graph + 12A + 12B** | **37 μs** | **2.14×** |

**Speed DoD (Phase 9) parked as future work.** Closing it requires tensor-
core utilisation inside the dequant path; revisit when integrating into
vLLM/SGLang where a production kernel stack is in play. Memory thesis
(Phase 13 + 14) is the current priority.

## Phase 13: Bit-tight Packing (planned)

### Goal

Close the remaining 16 B/token gap between our measured per-token storage
(150 B at b=4, D=128) and the TurboQuant paper's spec (134 B = **3.82× vs
FP16**). The gap is not a spec bug — `mse_packed` correctly stores `b-1 =
3` bits per coord — but `pack_bits(n_bits=3)` lays values at `vpb = 8 // 3
= 2` values per byte, wasting 25% (2 bits per byte) at odd widths.

This phase also unlocks Phase 14: outlier-channel split at `b+1 = 5` bits
is otherwise packed at `vpb=1` (3 wasted bits/byte, 37.5% overhead), which
would erase Phase 14's compression gains.

### Design

Replace the `vpb`-based byte-aligned packer with a **bit-cursor packer**
that lays values end-to-end across bytes. Widths supported: `n_bits ∈
{1, 2, 3, 4, 5, 6}`. Power-of-two widths stay unchanged (already tight).

**Storage layout:**
- `packed_bytes = ceil(D * n_bits / 8)`
- Value at index `i` occupies global bit offset `i * n_bits`, MSB-first
  within the stream. A single value may span two adjacent bytes.

**Kernel unpack:** precompute per-coord `(byte_lo, byte_hi, shift, mask)`
vectors of length `D`; the inner load path becomes `(byte_lo << 8 |
byte_hi) >> shift) & mask`. Power-of-two widths short-circuit to a single-
byte load when `d_off * n_bits % 8 == 0`.

### Files

| File | Change |
|------|--------|
| `src/turboquant/bit_packing.py` | Rewrite `pack_bits` / `unpack_bits` to bit-cursor layout; signatures unchanged |
| `src/turboquant/kernels/decompress_attn.py` | Replace single-byte MSE / V-idx unpack with two-byte-load spanning logic |
| `tests/test_bit_packing.py` | Extend round-trip sweeps to `n_bits ∈ {3, 5, 6}` at `D ∈ {64, 96, 128, 192}` |
| `tests/test_decompress_attn_triton.py` | Existing 15 parity tests re-run; add `key_bits=5` / `value_bits=5` sweep |
| `scripts/measure_compression.py` (new) | Print per-component bytes/token + ratio |

### Definition of Done

- All parity tests green including new odd-width sweeps.
- `scripts/measure_compression.py --bits 4` emits 134 B/token, 3.82× ratio.
- Graph perf gate (`TURBOQUANT_CUDA_GRAPH=1`) still ≤ 2.4× SDPA.

## Phase 14: Outlier-Channel Split (done)

**Status:** Landed on branch `phase-b-kernel-and-calibration` via the
B-1 through B-5 commits. Measured compression matches theory exactly
(tight packing, no slack):

| Mode | `k_avg` | `v_avg` | B/tok/head | Ratio vs FP16 |
|------|---------|---------|------------|---------------|
| Single b=4 | 4.00 | 4.00 | 134 | 3.82× |
| Single b=3 | 3.00 | 3.00 | 102 | 5.02× |
| Single b=2 | 2.00 | 2.00 |  70 | 7.31× |
| Split (b=4, f=0.25, hi=5, lo=3) | 3.50 | 3.50 | 118 | **4.34×** |
| Split (b=3, f=0.25, hi=4, lo=2) | 2.50 | 2.50 |  86 | **5.95×** |

Reproduce with `python scripts/measure_compression.py`.



### Goal

Reach fractional average bit-widths (2.5, 3.5) by splitting D channels
into **outliers** (high-magnitude, stored at `b+1` bits) and **regulars**
(low-magnitude, stored at `b-1` bits). Matches the paper's Section 4.2 /
LLM.int8 [Dettmers et al.] outlier strategy.

### Design

**Storage model** (average `b` bits, outlier fraction `f`):

- `n_out = round(f * D)` outlier channels → `b_hi = b+1` bits
- `n_reg = D - n_out` regular channels → `b_lo = b-1` bits
- Average: `f*(b+1) + (1-f)*(b-1) = b - 1 + 2f` bits/coord.
  - `f=0.25, b=3` → avg 2.5
  - `f=0.25, b=4` → avg 3.5
  - `f=0.5,  b=b` → avg b (breakeven, useful for validation)

**Per-head static permutation** places outliers in `[0..n_out)` of the
rotated D-axis. Fold permutation into `Pi_effective = Pi[perm]` so the
kernel sees a single rotation and the outlier prefix is natural.

**Offline calibration** (one-time): for each head, compute per-channel RMS
of rotated `Π·x` on a calibration batch; top-`n_out` by magnitude →
outliers. Persisted as `_outlier_perm: LongTensor[H, D]` on the quantizer.

### Files

| File | Change |
|------|--------|
| `src/turboquant/quantizer_prod_torch.py` | `_outlier_perm`, `_bits_hi`, `_bits_lo`, `_n_out`; `quantize_with_norm` splits output into two buckets |
| `src/turboquant/quantizer_mse_torch.py` | Same for V quantizer |
| `src/turboquant/integrations/hf_cache.py` | `_BatchedKeyStore.mse_packed_hi` / `_lo`; `_BatchedValueStore.idx_packed_hi` / `_lo`; `TurboQuantCache(outlier_frac=...)` |
| `src/turboquant/kernels/decompress_attn.py` | `_attn_split_kernel` accepts two buckets + `N_OUT` constexpr; single-bucket fallback when `N_OUT == 0` |
| `src/turboquant/calibration.py` (new) | `calibrate_outliers(quantizer, calib_batch, n_out) -> LongTensor[H,D]` |
| `tests/test_outlier_split_parity.py` (new) | Kernel parity across `(B, H_q, H_kv, S, D, b_hi, b_lo, n_out)` sweep |
| `tests/test_calibration.py` (new) | Calibration picks scaled channels on synthetic data |

### Order of Operations

1. Split-bucket storage with `n_out = 0` = Phase 13 behavior. Commit.
2. Kernel two-region unpack with `N_OUT=0` fallback. Commit after parity.
3. Add calibration, enable `outlier_frac > 0`. Commit.
4. Measure compression + perplexity; update spec. Commit.

### Definition of Done

- `pytest tests/test_outlier_split_parity.py tests/test_calibration.py`
  green with `outlier_frac ∈ {0, 0.25}` across shape sweep.
- `scripts/measure_compression.py --bits-avg 2.5 --outlier-frac 0.25`
  emits ≥ 5.3× ratio.
- Non-regression: graph perf gate still ≤ 2.4× SDPA.
- End-to-end: `model.generate` on Llama-3.1-8B at `(outlier_frac=0.25,
  bits_avg=3.5)` shows WikiText-2 perplexity within 0.1 of `bits=4`
  single-bucket (matches paper Table 2 comparable compression row).

---

## Phase 15: Evaluation Suite

### Goal

Reproduce the paper's Table 2 (LongBench on Llama-3.1-8B) and add
perplexity benchmarks on WikiText-2 and C4. Compare TurboQuant at
b=2,3,4 against FP16 baseline only (no competitor re-implementation).

### New Files

```
benchmarks/
    __init__.py
    config.py               # BenchmarkConfig dataclass
    perplexity.py           # WikiText-2 and C4 perplexity evaluation
    longbench.py            # LongBench suite runner (16 tasks)
    memory.py               # VRAM profiling utilities
    throughput.py            # Tokens/sec measurement
    run_perplexity.py       # CLI: python -m benchmarks.run_perplexity
    run_longbench.py        # CLI: python -m benchmarks.run_longbench
    results/                # JSON output directory
```

### Modified Files

- `pyproject.toml` -- add `[project.optional-dependencies] eval`

### Perplexity Benchmark

- Load model with TurboQuantCache at specified bit-width.
- Evaluate on WikiText-2 test set and C4 validation set.
- Sliding window with stride = context_length / 2.
- Report: perplexity, peak VRAM (MB), throughput (tokens/sec).
- Configurations: FP16, TQ b=2, TQ b=3, TQ b=4.

### LongBench Benchmark

Tasks from the paper's Table 2 (dataset: `THUDM/LongBench`):

| Category | Tasks |
|----------|-------|
| Single-doc QA | NarrativeQA, Qasper, MultiFieldQA-en |
| Multi-doc QA | HotpotQA, 2WikiMQA, Musique |
| Summarization | GovReport, QMSum, MultiNews |
| Few-shot | TREC, TriviaQA, SAMSum |
| Synthetic | PassageCount, PassageRetrieval |
| Code | LCC, RepoBench-P |

Metrics: F1 (QA), ROUGE-L (summarization), accuracy (classification).

### Results Format

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "method": "turboquant",
  "key_bits": 4,
  "value_bits": 4,
  "tasks": {
    "narrativeqa": {"f1": 23.4, "memory_mb": 4200, "tokens_sec": 42.1},
    "qasper": {"f1": 31.2, "memory_mb": 4100, "tokens_sec": 44.8}
  },
  "average": 50.06,
  "peak_vram_mb": 8500,
  "timestamp": "2026-04-08T14:30:00Z"
}
```

### Dependencies

- Phase 8 (KV cache integration) must be complete.
- `datasets>=2.14`, `rouge-score>=0.1` as optional `[eval]` deps.

### Testing Strategy

- Unit tests for metric functions (F1, ROUGE-L, perplexity math).
- Smoke test: perplexity on 1 batch of WikiText-2 with a tiny model.
- Full LongBench is a manual benchmark, not a CI test.

### Risks

- **LongBench evaluation complexity.** Each task has different metrics and
  input formats. Mitigate: use official LongBench evaluation scripts where
  possible rather than reimplementing.
- **Long contexts + RTX 5000 Ada.** Some LongBench tasks require very long
  contexts -- this is exactly what KV cache quantization enables. May need
  to set a max context limit for the FP16 baseline comparison.

### Definition of Done

- Perplexity benchmarks produce reproducible numbers on WikiText-2 and C4.
- LongBench runner covers all 16 tasks from the paper.
- Results format matches paper Table 2 layout (per-task + average).
- Memory and throughput metrics captured alongside accuracy.

---

## Phase 16: llama.cpp Fork

### Goal

Port TurboQuant to C/C++ within a fork of `ggerganov/llama.cpp`, adding
native GGML quantization types for production CPU and GPU inference.

This is a **separate repository** (fork of llama.cpp), not modifications
to the turboquant Python package.

### New GGML Types

Add `GGML_TYPE_TQ2`, `GGML_TYPE_TQ3`, `GGML_TYPE_TQ4`.

Each type defines a block format for a group of quantized values:

```
Block layout (block_size elements at b bits):
  [block_size * b bits]  quantized indices (packed)
  [32 bits]              L2 norm (float32, for non-unit vectors)
  [32 bits]              gamma (float32, Prod only -- residual norm)
```

Block size should be a multiple of 32 for memory alignment.

### Key Design Decisions

**Rotation and QJL matrices.** Regenerated from a deterministic seed at
model load time (seed = `layer_idx * 1000 + head_idx`). No file storage
needed. For 32 layers x 8 KV heads x 128x128 float32 = 16 MB total.

**Codebook.** Precomputed at load time. For d=128 and b=4: 16 centroids +
17 boundaries = 132 bytes. Trivial.

**CUDA kernels.** Rotation via cuBLAS matmul. Scalar quantize: one thread
per element, codebook in shared memory. QJL sign: one thread per element.
Dequantize: fused lookup + matmul kernel.

### Files to Create/Modify

In the llama.cpp fork:

| File | Changes |
|------|---------|
| `ggml/include/ggml.h` | Add GGML_TYPE_TQ{b} enum values |
| `ggml/src/ggml.c` | Block sizes, type traits |
| `ggml/src/ggml-quants.h` | Declare TQ quantize/dequantize functions |
| `ggml/src/ggml-quants.c` | C implementation of TQ ops |
| `ggml/src/ggml-cuda/turboquant.cu` | CUDA kernels for TQ |
| `src/llama-kv-cache.cpp` | Wire TQ types into KV cache allocation |
| `src/llama-model.cpp` | Support TQ cache type in model config |
| `convert_hf_to_gguf.py` | Add TQ metadata to conversion |

### C Implementation of Core Ops

- **Random rotation**: QR of Gaussian matrix via seed-based RNG (PCG or
  xoshiro). Store d x d float32 matrix per layer+head.
- **Lloyd-Max codebook**: Precompute on CPU at load. Binary search for
  nearest centroid (b comparisons per scalar).
- **QJL**: Generate S from seed, compute `sign(S @ r)`, store as bits.
  Dequantize: `sqrt(pi/2)/d * gamma * S^T @ signs`.

### Usage Target

```bash
llama-cli -m model.gguf \
  --cache-type-k tq4 \
  --cache-type-v tq4
```

### Testing Strategy

- Unit tests in C comparing against NumPy reference outputs. Generate test
  vectors in Python, export to binary files, verify C outputs match.
- Integration: compare perplexity against Python Phase 9 results.
- CUDA vs CPU result comparison for numerical consistency.

### Dependencies

- Phase 15 evaluation results for validation targets.
- CMake, CUDA toolkit, C++17 compiler.

### Risks

- **GGML type system is rigid.** Adding types requires touching many files.
  Study existing types (Q4_0, Q8_0) as templates.
- **Rotation matrix memory.** 16 MB total is acceptable but worth tracking.
- **Upstream divergence.** llama.cpp moves fast. Keep TQ code in separate
  files where possible to ease rebasing.

### Definition of Done

- Fork compiles with TQ types on Linux and Windows.
- `llama-cli` runs Llama-3.1-8B with `--cache-type-k tq4 --cache-type-v tq4`.
- Perplexity matches Python implementation within tolerance.
- CUDA backend works on RTX 5000 Ada.

---

## Phase 17+: Production Hardening

### 17a: Performance Optimization

- Profile the dequantization hotpath (likely the d x d matmul for unrotation).
- Investigate randomized Hadamard transform as O(d log d) alternative to
  full rotation (O(d^2)). The paper mentions this as future work.
- Batch per-head quantization into single large matmul operations.
- Explore incremental attention: compute attention scores directly from
  compressed representation without full dequantization.
- Revisit Phase 9 speed DoD (fused decode ≥ FP16 SDPA tok/s) in the context
  of a production kernel stack (vLLM / SGLang) — current bottleneck is
  compute-bound fp32 dequant inside `_attn_split_kernel` vs tensor-core
  SDPA.

### 17b: Documentation

- Jupyter notebook demonstrating KV cache compression on Llama-3.1-8B.
- API reference documentation (Sphinx or MkDocs).
- Performance comparison charts (distortion vs bit-width, throughput vs
  sequence length).

### 17c: CI/CD Pipeline

- GitHub Actions on push: pytest, ruff lint, mypy type check.
- GPU tests on a self-hosted runner or triggered manually.
- Benchmark regression tracking (flag if perplexity degrades > 0.5%).

### 17d: Distribution

- PyPI package publication (once API is stable).
- HuggingFace model cards showing TurboQuant benchmark results.

### 17e: H100 / FP8 Support

Target hardware: NVIDIA H100 (SM 90) with native FP8 tensor cores.

**Scenario A — Weight-only FP8** (bitsandbytes / quanto / torchao)
Model weights are stored as FP8 but activations remain BF16/FP16. `TurboQuantCache`
already handles this correctly — `key_states` arrive as BF16/FP16. No changes needed.

**Scenario B — True FP8 activations** (NVIDIA Transformer Engine)
TE fuses scaling factors into FP8 GEMMs; `key_states` arrive as `float8_e4m3fn`.
Support requires:

- Detect FP8 input dtype in `TurboQuantCacheLayer.update()`.
- Use TE-aware descaling (TE stores per-tensor scale factors outside the tensor,
  passed via `model_inputs`) rather than plain `.float()`.
- Return `float8_e4m3fn` output only if downstream attention is also TE-wrapped;
  otherwise return BF16 (current FP8 guard behaviour).

**Current status**: The FP8 guard (`model_dtype.itemsize >= 2` in `update()`)
prevents silent corruption for Scenario B by promoting output to FP16.
Full Scenario B support requires `transformer-engine` as a dependency and
H100 hardware for validation. Deferred until H100 is available.

**Definition of Done**:

- Scenario A: complete (no changes needed).
- Scenario B: `TurboQuantCacheLayer` round-trips correctly through a
  `te.Linear`-wrapped attention layer on H100.

---

## pyproject.toml Evolution

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]
gpu = ["torch>=2.0"]
hf = ["transformers>=4.51", "accelerate>=0.20"]
eval = ["datasets>=2.14", "rouge-score>=0.1"]
all = ["turboquant[gpu,hf,eval,dev]"]
```

---

## Appendix: Paper Reference Numbers

Expected results for Llama-3.1-8B on LongBench (from paper Table 2):

| Bit-width | Avg Score | Notes |
|-----------|-----------|-------|
| FP16 | ~50 | Full precision baseline |
| TQ 4-bit | ~49-50 | Matches FP16 |
| TQ 3.5-bit | ~48-49 | Slight degradation |
| TQ 3-bit | ~46-48 | Noticeable on long-context tasks |
| TQ 2-bit | ~40-44 | Significant quality loss |

These are approximate targets for Phase 9 validation.
