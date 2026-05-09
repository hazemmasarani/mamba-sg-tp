import torch
from model.mamba_gate_modeling import MambaForCausalLM_Gate


model = MambaForCausalLM_Gate.from_pretrained("HMasarani/mamba-gate")

state_dict = model.state_dict()

torch.save(state_dict, "./model/state_dict_gate.pt")
print("State dict saved successfully!")
for key in state_dict.keys():
    print(f"key: {key}, shape: {state_dict[key].shape}")