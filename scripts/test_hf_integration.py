"""Integration test: TurboQuantCache with real HuggingFace models.

Tests Llama-3.1-8B-Instruct and Gemma-7B-it generation quality and
memory savings compared to the FP16 DynamicCache baseline.

Usage:
    # Load token from .env automatically
    python scripts/test_hf_integration.py --model llama --bits 4,3,2

    # Override token explicitly
    HF_TOKEN=hf_xxx python scripts/test_hf_integration.py --model gemma

    # Quick smoke test
    python scripts/test_hf_integration.py --model llama --bits 4 --skip-baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env from project root (two levels up from scripts/)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import torch

MODEL_SHORTCUTS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-7b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",

}

DEFAULT_PROMPT = (
    "The attention mechanism in transformers has revolutionized natural language "
    "processing. Unlike recurrent networks, attention allows models to directly "
    "relate any two positions in a sequence. Explain the key advantages of this "
    "approach and how it enables long-context understanding:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> str:
    return MODEL_SHORTCUTS.get(name, name)


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


def _reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _header(model_id: str, prompt: str, device: str) -> None:
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print()
    print("=" * 66)
    print("  TurboQuant Integration Test")
    print(f"  Model : {model_id}")
    print(f"  Device: {device_name}")
    print(f"  Prompt: {prompt_preview[:60]}")
    print("=" * 66)
    print()


def _table_header() -> None:
    print(
        f"{'Method':<18} {'Bits K/V':>9} {'KV Cache':>10} {'Comp.':>8} "
        f"{'tok/s':>7}  Generated text (first 80 chars)"
    )
    print("-" * 110)


def _table_row(
    method: str,
    key_bits: int | str,
    val_bits: int | str,
    kv_mb: float,
    ratio: float,
    tps: float,
    text: str,
) -> None:
    bits_str = f"{key_bits:>2}/{val_bits:<2}"
    ratio_str = f"{ratio:.2f}x"
    text_preview = text.replace("\n", " ")[:80]
    print(
        f"{method:<18} {bits_str:>9} {kv_mb:>8.1f}MB {ratio_str:>8} "
        f"{tps:>6.1f}  {text_preview}"
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _dynamic_cache_bytes(cache) -> int:
    """Sum bytes stored in a DynamicCache (K + V tensors only, no model weights).

    Handles transformers versions:
    - >=5.x   : DynamicCache.layers[i].keys / .values  (DynamicLayer objects)
    - 4.38–4.x: DynamicCache.key_cache / .value_cache  (lists of tensors)
    """
    total = 0

    # transformers 5.x layered cache (layers[i].keys, layers[i].values)
    if hasattr(cache, "layers") and cache.layers:
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, torch.Tensor):
                total += k.nbytes
            if isinstance(v, torch.Tensor):
                total += v.nbytes
        return total

    # transformers 4.38–4.x (key_cache / value_cache are lists of tensors)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for k, v in zip(cache.key_cache, cache.value_cache):
            if isinstance(k, torch.Tensor):
                total += k.nbytes
            if isinstance(v, torch.Tensor):
                total += v.nbytes
        return total

    return total


def run_baseline(model, tokenizer, inputs, max_new_tokens: int, device: str):
    """Run generation with FP16 DynamicCache."""
    from transformers import DynamicCache

    _reset_vram()
    cache = DynamicCache()
    run_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(run_device)
    attention_mask = inputs["attention_mask"].to(run_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[-1] - input_ids.shape[-1]
    tps = new_tokens / elapsed if elapsed > 0 else 0.0
    # Measure only the KV cache tensors, not model weights or activations.
    kv_bytes = _dynamic_cache_bytes(cache)
    kv_mb = kv_bytes / 1024**2
    text = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)

    return {"text": text, "tps": tps, "kv_mb": kv_mb, "ratio": 1.0}


def run_turboquant(model, tokenizer, inputs, max_new_tokens: int, device: str, bits: int):
    """Run generation with TurboQuantCache."""
    from turboquant.integrations.hf_cache import TurboQuantCache
    from turboquant.integrations.memory_tracker import report as tq_report

    _reset_vram()
    # When device_map="auto", model layers may be split across devices.
    # Use the device of the first parameter as the quantizer device.
    quant_device = next(model.parameters()).device
    cache = TurboQuantCache(model.config, key_bits=bits, value_bits=bits, device=quant_device)
    input_ids = inputs["input_ids"].to(quant_device)
    attention_mask = inputs["attention_mask"].to(quant_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[-1] - input_ids.shape[-1]
    tps = new_tokens / elapsed if elapsed > 0 else 0.0
    mem = tq_report(cache)
    kv_mb = mem["compressed_bytes"] / 1024**2
    ratio = mem["compression_ratio"]
    text = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)

    return {"text": text, "tps": tps, "kv_mb": kv_mb, "ratio": ratio}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test TurboQuantCache with real HuggingFace models."
    )
    parser.add_argument(
        "--model",
        default="llama",
        help="Model shortcut (llama/gemma) or any HF model ID/local path.",
    )
    parser.add_argument(
        "--bits",
        default="4,3,2",
        help="Comma-separated bit-widths to test (e.g. 4,3,2).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        dest="max_new_tokens",
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        dest="skip_baseline",
        help="Skip FP16 DynamicCache baseline run.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--max-gpu-gb",
        type=float,
        default=None,
        dest="max_gpu_gb",
        help=(
            "Cap GPU memory for model weights (GB). Overflow goes to CPU RAM. "
            "Use when model size is close to total VRAM to avoid segfaults. "
            "E.g. --max-gpu-gb 14 for Llama-3.1-8B on a 17 GB GPU."
        ),
    )
    args = parser.parse_args()

    model_id = resolve_model(args.model)
    bit_widths = [int(b.strip()) for b in args.bits.split(",")]
    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    _header(model_id, args.prompt, args.device)

    # Check token
    if hf_token is None:
        print(
            "WARNING: HF_TOKEN not set. Gated models (Llama, Gemma) require a token.\n"
            "  Set HF_TOKEN env var or add it to .env in the project root.\n"
        )

    # Load tokenizer + model
    print(f"Loading {model_id} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # Build max_memory map: cap GPU allocation to leave headroom for activations
    # and KV cache. Overflow layers are placed on CPU RAM automatically.
    # When VRAM is tight (model ≈ total VRAM), device_map="auto" alone still
    # crashes because CUDA OOM during load triggers a segfault before Python
    # can catch it. Explicit max_memory gives accelerate a safe budget.
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Default: reserve 1.5 GB for driver + activations, rest for weights.
        gpu_budget_gb = args.max_gpu_gb if args.max_gpu_gb else max(total_gb - 1.5, 1.0)
        max_memory = {0: f"{gpu_budget_gb:.0f}GiB", "cpu": "48GiB"}
        device_map = "auto"
    else:
        max_memory = None
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"  Dtype      : {torch_dtype}")
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM (model weights): {vram_used:.0f} MB / {vram_total:.0f} MB")
    print()

    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"Prompt tokens  : {prompt_len}")
    print(f"New tokens     : {args.max_new_tokens}")
    print()

    results = []

    # Baseline
    if not args.skip_baseline:
        print("Running FP16 baseline ...", end=" ", flush=True)
        try:
            r = run_baseline(model, tokenizer, inputs, args.max_new_tokens, args.device)
            r["method"] = "FP16 Baseline"
            r["key_bits"] = 16
            r["val_bits"] = 16
            results.append(r)
            print("done")
        except torch.cuda.OutOfMemoryError:
            print("\nERROR: CUDA OOM during baseline. Try --max-new-tokens 50.")
            sys.exit(1)

    # TurboQuant runs
    for bits in bit_widths:
        label = f"TurboQuant b={bits}"
        print(f"Running {label} ...", end=" ", flush=True)
        try:
            r = run_turboquant(
                model, tokenizer, inputs, args.max_new_tokens, args.device, bits
            )
            r["method"] = label
            r["key_bits"] = bits
            r["val_bits"] = bits
            results.append(r)
            print("done")
        except torch.cuda.OutOfMemoryError:
            print(f"\nERROR: CUDA OOM at b={bits}. Try --max-new-tokens 50.")
            continue

    # Print table
    print()
    _table_header()
    for r in results:
        _table_row(
            r["method"],
            r["key_bits"],
            r["val_bits"],
            r["kv_mb"],
            r["ratio"],
            r["tps"],
            r["text"],
        )
    print()

    # Print full generated texts
    print("=" * 80)
    print("Full generated texts:")
    print("=" * 80)
    for r in results:
        print(f"\n[{r['method']}]")
        print(r["text"])

    print()
    print("Done.")


if __name__ == "__main__":
    main()
