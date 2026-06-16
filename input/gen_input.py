import torch

input_ids = torch.randint(0, 50280, (2, 128))
torch.save(input_ids, "./input/input_ids_1.pt")