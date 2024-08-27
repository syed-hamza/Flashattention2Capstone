
import torch
import torchprofile
# Import the necessary module and class 
from gpt import GPTLanguageModel # Assuming GPTLanguageModel is in gpt.py

# Define the variables
batch_size = 64
block_size = 256
max_iters = 200
eval_interval = 10
learning_rate = 3e-4
vocab_size = 50257
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Define your model here
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
model.to(device)

# Dummy input for profiling
input_ids = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

# Calculate FLOPs
flops = torchprofile.profile_macs(model, args=(input_ids,))
print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")