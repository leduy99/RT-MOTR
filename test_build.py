import torch

# Load the model state dict from the .pth file
checkpoint = torch.load('exp_g2mot/latest.pth')
print(checkpoint.keys())

# Access the state dict
state_dict = checkpoint['model_state_dict']  # Change 'model_state_dict' to whatever the current key is

# Change the key from 'model_state_dict' to 'model'
checkpoint['model'] = state_dict

# Optionally, you can remove the old key if it's no longer needed
del checkpoint['model_state_dict']

# Save the modified checkpoint back to a .pth file
torch.save(checkpoint, 'modified_model.pth')