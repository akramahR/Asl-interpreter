import torch
import numpy as np

# Load the PyTorch model
model = torch.load('flow_charades.pt')
# Assume the model is already in eval mode
model.eval()

# Extract the weights
weights_dict = {}
for name, param in model.named_parameters():
    weights_dict[name] = param.data.numpy()

# Save the weights as a NumPy file
np.save('flow_charades_weights.npy', weights_dict)
