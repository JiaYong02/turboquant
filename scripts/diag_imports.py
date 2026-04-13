"""Diagnose imports and VRAM state before model load."""
import sys
print("Step 1: import torch")
import torch
print("  CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

print("Step 2: import transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("  OK")

print("Step 3: import turboquant")
from turboquant.integrations.hf_cache import TurboQuantCache
print("  OK")

print("Step 4: load tokenizer")
import os
from pathlib import Path
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
hf_token = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
print("  OK")

print("Step 5: load model (float16, device_map=auto)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=hf_token,
)
model.eval()
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    free2, total2 = torch.cuda.mem_get_info()
    print(f"  Model loaded. VRAM used: {used:.2f} GB, free: {free2/1e9:.2f} GB")

print("Step 6: short generation with TurboQuantCache b=4")
cache = TurboQuantCache(model.config, key_bits=4, value_bits=4, device="cuda")
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=10, do_sample=False)
print("  Generated:", tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:]))
print("All steps OK")
