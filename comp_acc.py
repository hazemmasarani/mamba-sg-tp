import torch
from transformers import MambaForCausalLM
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

set_seed(42)

# Load model
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").to("cuda:0")
# print(f"Model vocab size is {model.config.vocab_size}")

# Load input
inp = torch.load("input/input_ids_1.pt").to("cuda:0")

set_seed(42)

# Get original output
orig_out = model(inp).logits.cpu()

# Load distributed output
dist_out = torch.load("output/run_1.pt").cpu()

# Compare
batch_size, seq_len = inp.shape
print(f"Comparing for input of batch size {batch_size}, and sequence length {seq_len}.")
if torch.allclose(dist_out, orig_out, rtol=1e-5, atol=1e-5):
    print("Same output.")

# Useful debugging information
diff = (dist_out - orig_out).abs()
print(f"Max absolute difference: {diff.max().item()}")
print(f"Mean absolute difference: {diff.mean().item()}")
with open("results/logits_diff.log", "a") as f:
    f.write(f"{batch_size}, {seq_len}, {diff.max().item()}, {diff.mean().item()}\n")