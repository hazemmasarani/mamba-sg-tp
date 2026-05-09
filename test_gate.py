import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import os
from model.mamba_gate_modeling import MambaForCausalLM_Gate
from transformers import MambaConfig


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

# Example for in_proj weight splitting
def shard_tensor(tensor, rank, world_size, dim=0):
    chunks = torch.chunk(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()

def read_split_state_dict(path, rank, world_size):
    state_dict = torch.load(path)
    for key in list(state_dict.keys()):
        if 'in_proj' in key:
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=0)  # split along the intermediate dimension
    return state_dict

# -------------------------
# Training / inference function
# -------------------------
def run(rank, world_size, devices):

    # Initialize distributed process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',  # single-node example
        world_size=world_size,
        rank=rank
    )

    # set rank and deviceej
    device = torch.device(devices[rank])

    set_seed(42) 

    config = MambaConfig.from_pretrained("HMasarani/mamba-gate")
    config.device = device
    config.world_size = world_size
    config.rank = rank
    config.local_intermediate_size = config.intermediate_size // world_size

    # Instantiate model on this rank
    model = MambaForCausalLM_Gate(config).to(device)

    state_dict = read_split_state_dict("./model/state_dict_gate.pt", rank, world_size)
    model.load_state_dict(state_dict, strict=False)

    # Dummy input
    # input_ids = torch.ones((1, 128), dtype=torch.long).to(device)
    input_ids = torch.load("./input/input_ids.pt").to(device)
    output = model(input_ids)
    print(output.logits.shape)

    logits = output.logits

    if dist.is_initialized():
        # Gather logits from all ranks
        gathered_logits = [torch.zeros_like(logits) for _ in range(world_size)]
        dist.all_gather(gathered_logits, logits)

        # Compare outputs
        print(f"Rank {rank} shape of gsthered logits: {gathered_logits[0].shape}")
        print(f"size of gathered_logits: {len(gathered_logits)}, rank: {rank}")
        all_equal = True
        for logit in gathered_logits:
            print(f"gathered logit shape: {logit.shape}, rank: {rank}")
            all_equal = all_equal and torch.allclose(logits, logit, atol=1e-4)
        if all_equal:
            print(f"[Rank {rank}] ✅ All ranks outputs match exactly!")
        else:
            print(f"[Rank {rank}] ❌ Outputs differ between ranks!")
        
        # save logits
        torch.save(logits, f"./output/logits_{rank}_{world_size}.pt")

    dist.destroy_process_group()

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":

    set_seed(42)

    # 2️⃣ Add world size, and rank to config for distributed training
    world_size = 2
    devices = ["cuda:0", "cuda:1"]  

    mp.spawn(run, args=(world_size, devices), nprocs=world_size, join=True)
