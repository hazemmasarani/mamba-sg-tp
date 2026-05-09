import torch

input_ids = torch.randint(0, 50280, (1, 128))
torch.save(input_ids, "./input/input_ids.pt")