"""Check VRAM headroom and available model loading options."""
import torch, os, subprocess
from pathlib import Path

free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total : {total/1e9:.2f} GB")
print(f"VRAM free  : {free/1e9:.2f} GB")
print(f"Driver overhead: {(total-free)/1e9:.2f} GB")
print()

# Check if bitsandbytes is available (for 4-bit/8-bit quantized loading)
try:
    import bitsandbytes
    print("bitsandbytes:", bitsandbytes.__version__, "✓")
except ImportError:
    print("bitsandbytes: NOT installed (can't load in 4/8-bit)")

# Check HF cache for any already-downloaded models
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    models = [d.name for d in hf_cache.iterdir() if d.is_dir() and d.name.startswith("models--")]
    print(f"\nCached HF models ({len(models)}):")
    for m in sorted(models):
        print(f"  {m}")
else:
    print("\nNo HF cache found at", hf_cache)

# Estimate: 8B fp16 model = 16 GB. With 17.2 GB we have ~1.2 GB headroom — too tight.
# 8-bit would be 8 GB, 4-bit would be 4 GB. Either works.
print()
print(f"Llama-3.1-8B fp16  : ~16.0 GB  -> {'OK' if free > 16e9 else 'OOM (need ~16 GB)'}")
print(f"Llama-3.1-8B int8  : ~8.5 GB   -> {'OK' if free > 8.5e9 else 'OOM (need ~8.5 GB)'}")
print(f"Llama-3.1-8B int4  : ~4.5 GB   -> {'OK' if free > 4.5e9 else 'OOM (need ~4.5 GB)'}")
