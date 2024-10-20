import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPTLanguageModel
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No handlers found:.*")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Define constants
batch_size = 4
n_layer = 6
dropout = 0.2
head_dim = 64  # Fixed head dimension
n_head = 6     # You can adjust this if needed

# Calculate embedding dimension
n_embd = head_dim * n_head

# Set vocabulary size (can be arbitrary for dummy data)
vocab_size = 50257  # Common value used in GPT models (e.g., OpenAI's GPT-2)

# Function to get batches of dummy data
def get_dummy_batch(batch_size, seq_length):
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length), dtype=torch.long, device=device)
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length), dtype=torch.long, device=device)
    return x, y

output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

def calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd):
    # FLOPs calculation based on standard Transformer computations
    d_head = n_embd // n_head
    flop = 0

    # Self-attention FLOPs
    # Q, K, V projections
    flop += 3 * n_embd * seq_length * n_embd
    # Attention scores and probabilities
    flop += n_head * seq_length * (seq_length * d_head + seq_length)
    # Attention output projection
    flop += n_embd * seq_length * n_embd

    # Feed-forward FLOPs
    flop += 2 * seq_length * n_embd * (4 * n_embd)  # Assuming 4*n_embd in FFN

    total_flops = flop * batch_size * n_layer
    return total_flops

def benchmark_model(model, seq_length, n_head, n_embd, n_layer, warmup=3, iterations=10):
    model.eval()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        X, Y = get_dummy_batch(batch_size, seq_length)
        with torch.no_grad():
            _ = model(X, Y)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_time = 0
    for _ in range(iterations):
        X, Y = get_dummy_batch(batch_size, seq_length)
        start_event.record()
        with torch.no_grad():
            _ = model(X, Y)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    
    avg_time = total_time / iterations  # In milliseconds
    flops = calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd)
    tflops = flops / (avg_time * 1e-3) / 1e12  # Convert to TFLOPs/s
    return tflops

# Sequence lengths from 512 to 8192, doubling each time
seq_lengths = [512 * (2 ** i) for i in range(5)]  # [512, 1024, 2048, 4096, 8192]
flops_per_seq_length = []

for seq_length in seq_lengths:
    block_size = seq_length

    # Ensure n_embd is divisible by n_head
    assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
    
    print(f"Benchmarking for sequence length: {seq_length}")

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    ).to(device)
    
    # Benchmark model
    try:
        tflops = benchmark_model(model, seq_length, n_head, n_embd, n_layer)
        flops_per_seq_length.append((seq_length, tflops))
        print(f"Sequence Length: {seq_length}, Performance: {tflops:.2f} TFLOPs/s")
    except RuntimeError as e:
        print(f"Error benchmarking sequence length {seq_length}: {e}")
        print("Skipping to next sequence length.")
        continue
    
    del model
    torch.cuda.empty_cache()

# Save results to CSV
csv_path = os.path.join(output_dir, 'benchmark_results_seq_length.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sequence Length', 'TFLOPs/s'])
    for seq_length, flops in flops_per_seq_length:
        writer.writerow([seq_length, flops])

print(f"Benchmark results saved to {csv_path}")
