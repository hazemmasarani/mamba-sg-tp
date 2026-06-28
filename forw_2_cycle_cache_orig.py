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

model = MambaForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b-hf"
).to("cuda:0")

inp = torch.load("input/input_ids_1.pt").to("cuda:0")

with torch.no_grad():

    # First pass
    outputs = model(inp, use_cache=True)

    logits = outputs.logits
    cache = outputs.cache_params   # Mamba cache

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    # Second pass ONLY on new token
    outputs2 = model(
        next_token,
        cache_params=cache,
        use_cache=True
    )

    logits2 = outputs2.logits

    torch.save(logits2, f"results/orig_logits2.log")