import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPTLanguageModel
import os
from datasets import load_dataset
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No handlers found:.*")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Define constants
batch_size = 32
max_iters = 1
eval_interval = 10
learning_rate = 3e-4
eval_iters = 1
n_layer = 6
dropout = 0.2
head_dim = 64  # Fixed head dimension
n_head = 6     # You can adjust this if needed

# Calculate embedding dimension
n_embd = head_dim * n_head

# Load dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')

# Tokenizer
chars = sorted(list(set(''.join(dataset['text']))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode/decode functions
def encode(s):
    return [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the dataset
def encode_batch(batch):
    encoded_texts = [encode(text) for text in batch['text'] if text]
    return {'input_ids': encoded_texts}

encoded_dataset = dataset.map(encode_batch, batched=True, remove_columns=['text'], desc="Encoding dataset")

# Concatenate all data
all_data = []
for seq in encoded_dataset['input_ids']:
    all_data.extend(seq)
data = torch.tensor(all_data, dtype=torch.long)

# Split into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Function to get batches
def get_batch(data_source, batch_size, block_size):
    seq_len = block_size + 1  # +1 for the target
    total_len = data_source.size(0)
    if total_len - seq_len <= 0:
        raise ValueError("Data source is too small.")
    indices = torch.randint(0, total_len - seq_len, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in indices])
    y = torch.stack([data_source[i+1:i+seq_len] for i in indices])
    x = x.to(device)
    y = y.to(device)
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

def benchmark_model(model, data_iter, n_head, n_embd, n_layer, seq_length, warmup=3, iterations=10):
    model.eval()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        X, Y = next(data_iter)
        with torch.no_grad():
            _ = model(X, Y)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_time = 0
    for _ in range(iterations):
        X, Y = next(data_iter)
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

# Sequence lengths from 512 to 16384 (16k), doubling each time
seq_lengths = [512 * (2 ** i) for i in range(5)]  # [512, 1024, 2048, 4096, 8192, 16384]
flops_per_seq_length = []

for seq_length in seq_lengths:
    block_size = seq_length

    # Ensure n_embd is divisible by n_head
    assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
    
    # Check if block_size exceeds the length of the dataset
    if block_size + 1 >= len(train_data):
        print(f"Sequence length {seq_length} is too long for the dataset. Skipping.")
        continue

    print(f"Benchmarking for sequence length: {seq_length}")

    # Data iterator
    def data_iter():
        while True:
            yield get_batch(train_data, batch_size, block_size)
    data_loader = data_iter()
    
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
        tflops = benchmark_model(model, data_loader, n_head, n_embd, n_layer, seq_length)
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
