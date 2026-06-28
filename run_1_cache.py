import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import os
from model.all_to_all_single.mamba_ssm_modeling import MambaForCausalLM_SSM
from model.all_to_all_single.mamba_gate_modeling import MambaForCausalLM_Gate
from transformers import MambaConfig
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

# Example for in_proj weight splitting
def shard_tensor(tensor, rank, world_size, dim=0):
    chunks = torch.tensor_split(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()

def get_shard_sizes(total_size, num_parts):
    base = total_size // num_parts
    remainder = total_size % num_parts

    sizes = []
    for i in range(num_parts):
        if i < remainder:
            sizes.append(base + 1)
        else:
            sizes.append(base)
    return sizes

def build_start_end_dict(splits_size, ranks):
    res = {}
    start = 0

    for i, size in enumerate(splits_size):
        end = start + size - 1
        res[ranks[i]] = [start, end]
        start = end + 1

    return res

def build_reshard_strategy(rank, src_dict, dst_dict):
    """
    Returns:
        {
            dst_rank: [local_start, local_end]
        }

    using [start, end) indexing.
    """

    src_start, src_end = src_dict[rank]

    strategy = {}

    for dst_rank, (dst_start, dst_end) in dst_dict.items():

        overlap_start = max(src_start, dst_start)
        overlap_end   = min(src_end, dst_end)

        if overlap_start < overlap_end:

            local_start = overlap_start - src_start
            local_end   = overlap_end - src_start

            strategy[dst_rank] = [local_start, local_end]

    return strategy

def read_split_state_dict_ssm(path, rank, world_size):
    state_dict = torch.load(path)
    for key in list(state_dict.keys()):
        if "mixer.A_log" in key or "mixer.D" in key or "mixer.conv1d" in key or "mixer.in_proj" in key or "mixer.dt_proj" in key:
            # print(f"Before {key} split : {state_dict[key].shape}")
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=0)  # split along the intermediate dimension
            # print(f"After {key} split : {state_dict[key].shape}")
        elif "mixer.x_proj" in key or "mixer.out_proj" in key:
            # print(f"Before {key} split : {state_dict[key].shape}")
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=-1)  # split along the intermediate dimension
            # print(f"After {key} split : {state_dict[key].shape}")
    return state_dict

def read_split_state_dict_gate(path, rank, world_size):
    state_dict = torch.load(path)
    for key in list(state_dict.keys()):
        if 'in_proj' in key:
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=0)  # split along the intermediate dimension
        elif "mixer.out_proj" in key:
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=-1)  # split along the intermediate dimension
    return state_dict

# -------------------------
# Training / inference function
# -------------------------
def run_ssm(rank, world_size, devices_map, num_cycles):

    # Initialize distributed process group
    device = torch.device(devices_map['ssm'][rank])
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',  # single-node example
        world_size = world_size,
        rank=rank,
        device_id = device
    )

    # Load Configuration
    config = MambaConfig.from_pretrained("HMasarani/mamba-ssm")

    local_group = dist.new_group(ranks=list(devices_map['ssm'].keys()))
    ssm_offset = len(devices_map['gate'])

    set_seed(42)

    config.world_size = len(devices_map['ssm'])
    config.rank = rank
    config.device = device
    config.devices_map = devices_map
    config.ssm_offset = ssm_offset
    config.local_intermediate_size = get_shard_sizes(config.intermediate_size , config.world_size)[rank - ssm_offset]

    # Build reshard Strategy
    gate_shards_size = get_shard_sizes(config.intermediate_size , len(devices_map['gate']))
    ssm_shards_size = get_shard_sizes(config.intermediate_size , len(devices_map['ssm']))
    ssm_s_e_dict = build_start_end_dict(ssm_shards_size, list(devices_map['ssm'].keys()))
    gate_s_e_dict = build_start_end_dict(gate_shards_size, list(devices_map['gate'].keys()))
    reshard_strategy = build_reshard_strategy(rank, ssm_s_e_dict, gate_s_e_dict)
    config.reshard_strategy = reshard_strategy

    # # Instantiate model on this rank
    model = MambaForCausalLM_SSM(config).to(device)
    model.config.local_group = local_group

    state_dict = read_split_state_dict_ssm("./model/state_dict_ssm.pt", rank - ssm_offset, config.world_size)
    model.load_state_dict(state_dict, strict=False)

    print(f"ssm of ranke {rank} has reshard_strategy {reshard_strategy}")

    # Dummy input
    # input_ids = torch.ones((1, 128), dtype=torch.long).to(device)
    input_ids = torch.load("./input/input_ids_1.pt").to(device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, counter=0, return_dict=True, use_cache=True)
        logits = outputs.logits
        cache = outputs.cache_params

        for i in range(1, num_cycles):
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            cache_position = torch.tensor([seq_len + i - 1], device=device)  # absolute position of this new token

            del logits

            outputs = model(
                next_token,
                cache_params=cache,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True,
                counter=i
            )

            logits = outputs.logits
            cache = outputs.cache_params


    if rank == min(devices_map['ssm'].keys()):
        torch.save(logits, f"results/dist_logits2.log")
    
    dist.barrier()
    torch.cuda.synchronize()

    dist.destroy_process_group()

def run_gate(rank, world_size, devices_map, num_cycles):

    # Initialize distributed process group
    device = torch.device(devices_map['gate'][rank])

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',  # single-node example
        world_size = world_size,
        rank=rank,
        device_id = device
    )

    # Load Configuration
    config = MambaConfig.from_pretrained("HMasarani/mamba-gate")

    local_group = dist.new_group(ranks=list(devices_map['gate'].keys()))
    ssm_offset = len(devices_map['gate'])

    # set rank and device

    set_seed(42) 

    config.world_size = len(devices_map['gate'])
    config.rank = rank
    config.ssm_offset = ssm_offset
    config.device = device
    config.devices_map = devices_map
    config.local_intermediate_size = get_shard_sizes(config.intermediate_size , config.world_size)[rank]

    # Build reshard Strategy
    gate_shards_size = get_shard_sizes(config.intermediate_size , len(devices_map['gate']))
    ssm_shards_size = get_shard_sizes(config.intermediate_size , len(devices_map['ssm']))
    ssm_s_e_dict = build_start_end_dict(ssm_shards_size, list(devices_map['ssm'].keys()))
    gate_s_e_dict = build_start_end_dict(gate_shards_size, list(devices_map['gate'].keys()))
    reshard_strategy = build_reshard_strategy(rank, gate_s_e_dict, ssm_s_e_dict)
    config.reshard_strategy = reshard_strategy

    # Instantiate model on this rank
    model = MambaForCausalLM_Gate(config).to(device)
    model.config.local_group = local_group

    state_dict = read_split_state_dict_gate("./model/state_dict_gate.pt", rank, config.world_size)
    model.load_state_dict(state_dict, strict=False)

    print(f"Gate of ranke {rank} has reshard_strategy {reshard_strategy}")

    # # Dummy input
    # # input_ids = torch.ones((1, 128), dtype=torch.long).to(device)
    output = None
    input_ids = torch.load("./input/input_ids_1.pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, counter=0, return_dict=True)
        logits = outputs.logits
        for i in range(1, num_cycles):

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            del outputs

            # Second pass ONLY on new token
            outputs = model(
                next_token,
                return_dict=True,
                counter=i
            )

            logits = outputs.logits

    dist.barrier()
    torch.cuda.synchronize()

    dist.destroy_process_group()

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("-num_iter", type=int, default=10, help="number of iterations")
    args = parser.parse_args()

    input_ids = torch.randint(0, 50280, (args.batch_size, args.seq_len))
    torch.save(input_ids, "./input/input_ids_1.pt")

    world_size = 4

    devices_map = {
        "gate": {0:"cuda:0", 1:"cuda:1"},
        "ssm": { 2:"cuda:2", 3:"cuda:3"},
    }

    processes = []

    ssm_size = len(devices_map["ssm"])
    gate_size = len(devices_map["gate"])

    # Initialize Gate Processes 
    for i in range(gate_size):
        p = mp.Process(target=run_gate, args=(i, world_size, devices_map, args.num_iter))
        p.start()
        processes.append(p)
    
    # Initialize SSM Processes
    for i in range(ssm_size):
        p = mp.Process(target=run_ssm, args=(i + gate_size, world_size, devices_map, args.num_iter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
