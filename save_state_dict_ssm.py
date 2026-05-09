import torch
from model.mamba_ssm_modeling import MambaForCausalLM_SSM


model = MambaForCausalLM_SSM.from_pretrained("HMasarani/mamba-ssm")

state_dict = model.state_dict()

torch.save(state_dict, "./model/state_dict_ssm.pt")
print("State dict saved successfully!")
for key in state_dict.keys():
    print(f"key: {key}, shape: {state_dict[key].shape}")
    