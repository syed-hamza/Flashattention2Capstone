import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from gptfa2 import GPTLanguageModel  # Importing the updated GPT model with FlashAttention
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No handlers found:.*")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Define constants
n_layer = 6
dropout = 0.0  # Set dropout to 0 for benchmarking
head_dim = 64  # Fixed head dimension
n_head = 8     # Adjusted to ensure n_embd is divisible by n_head

# Calculate embedding dimension
n_embd = head_dim * n_head

# Set vocabulary size (arbitrary for dummy data)
vocab_size = 50257  # Common value used in GPT models

# Output directory for benchmark results
output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

# Function to generate dummy data
def get_dummy_batch(batch_size, seq_length):
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        dtype=torch.long,
        device=device
    )
    y = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        dtype=torch.long,
        device=device
    )
    return x, y

# FLOPs calculation function
def calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd):
    # Adjusted FLOPs calculation for FlashAttention
    # FlashAttention computes attention more efficiently
    # but we use standard Transformer FLOPs for comparison
    d_head = n_embd // n_head
    flop = 0

    # Self-attention FLOPs
    # Q, K, V projections
    flop += 3 * n_embd * seq_length * n_embd
    # Attention (FlashAttention is more efficient, but we approximate)
    flop += n_head * seq_length * seq_length * d_head
    # Output projection
    flop += n_embd * seq_length * n_embd

    # Feed-forward FLOPs
    flop += 2 * seq_length * n_embd * (4 * n_embd)  # Assuming 4*n_embd in FFN

    total_flops = flop * batch_size * n_layer
    return total_flops

# Benchmarking function
def benchmark_model(model, seq_length, n_head, n_embd, n_layer,
                    batch_size, warmup=3, iterations=10):
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
    
    avg_time = total_time / iterations  # in milliseconds
    flops = calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd)
    tflops = flops / (avg_time * 1e-3) / 1e12  # Convert to TFLOPs/s
    return tflops

# Sequence lengths to benchmark
seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
base_batch_size = 32

# List to store results
flops_per_seq_length = []

for seq_length in seq_lengths:
    # Adjust batch size to prevent out-of-memory errors
    batch_size = max(1, int(base_batch_size * (512 / seq_length)))
    block_size = seq_length

    # Ensure n_embd is divisible by n_head
    assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

    print(f"Benchmarking for sequence length: {seq_length} with batch size: {batch_size}")

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
        tflops = benchmark_model(
            model, seq_length, n_head, n_embd, n_layer, batch_size
        )
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
