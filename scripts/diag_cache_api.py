"""Find the actual attribute names on DynamicCache in this transformers version."""
from transformers import DynamicCache
import transformers
print("transformers:", transformers.__version__)
c = DynamicCache()
print("Attributes:", [a for a in dir(c) if not a.startswith('__')])
