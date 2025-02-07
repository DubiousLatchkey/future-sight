# Check if training occurs

import torch

# Load two saved checkpoints
checkpoint1 = torch.load("checkpoints/model_epoch_900.pth")
checkpoint2 = torch.load("checkpoints/model_epoch_1000.pth")

# Compare weights layer by layer
for name, param in checkpoint1.items():
    if name in checkpoint2:
        diff = torch.sum(torch.abs(checkpoint1[name] - checkpoint2[name])).item()
        print(f"Layer: {name}, Weight Difference: {diff}")

for name, param in checkpoint2.items():
    if param.requires_grad and param.grad is not None:
        grad_norm = torch.norm(param.grad).item()
        print(f"Layer: {name}, Gradient Norm: {grad_norm}")