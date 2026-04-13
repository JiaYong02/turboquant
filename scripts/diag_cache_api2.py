"""Inspect DynamicCache layer structure after a forward pass."""
import torch
from transformers import DynamicCache, AutoTokenizer, AutoModelForCausalLM

# Use tiny gpt2 (no token needed)
print("Loading gpt2 ...")
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

inputs = tok("Hello", return_tensors="pt")
cache = DynamicCache()
with torch.no_grad():
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=5, do_sample=False)

print("cache.layers:", type(cache.layers))
print("num layers:", len(cache.layers))
layer0 = cache.layers[0]
print("layer0 type:", type(layer0))
print("layer0 attrs:", [a for a in dir(layer0) if not a.startswith('__')])
# Check for tensor attrs
for attr in vars(layer0):
    val = getattr(layer0, attr)
    if isinstance(val, torch.Tensor):
        print(f"  tensor {attr}: shape={val.shape}, nbytes={val.nbytes}")
    elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
        print(f"  list[Tensor] {attr}: len={len(val)}, shape={val[0].shape}")
