import time
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPTLanguageModel
from benchmarker import Benchmarker
import os
from datasets import load_dataset
import warnings
import torch.cuda

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No handlers found:.*")

# Hyperparameters
batch_size = 128
max_iters = 1
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1
n_embd = 384
n_layer = 6
dropout = 0.2
n_head = 64  # Fixed number of heads

torch.manual_seed(1337)

# Load dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-2-v1", split='train')

# Tokenizer
chars = sorted(list(set(''.join(dataset['text']))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Function to encode text and handle unknown characters
def encode(s):
    return [stoi[c] for c in s if c in stoi]

decode = lambda l: ''.join([itos[i] for i in l])

# Encode the dataset
def encode_batch(batch):
    encoded_texts = [encode(text) for text in batch['text'] if text]
    return {'input_ids': encoded_texts}

encoded_dataset = dataset.map(encode_batch, batched=True, remove_columns=['text'])

# Create data loader
def data_loader(split, batch_size, block_size):
    split_data = encoded_dataset.train_test_split(test_size=0.1)
    data = split_data['train'] if split == 'train' else split_data['test']

    def collate_fn(examples):
        examples = [ex for ex in examples if len(ex['input_ids']) <= block_size]

        inputs = torch.full((len(examples), block_size), fill_value=0, dtype=torch.long)
        labels = torch.full((len(examples), block_size), fill_value=0, dtype=torch.long)

        for i, ex in enumerate(examples):
            length = len(ex['input_ids'])
            inputs[i, :length] = torch.tensor(ex['input_ids'][:block_size])
            labels[i, :length-1] = torch.tensor(ex['input_ids'][1:block_size])

        return inputs.to(device), labels.to(device)

    return torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

def calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd):
    d_head = n_embd // n_head
    d_ff = n_embd * 4  # For feed-forward layer

    # FLOPs for self-attention
    attention_flops = 2 * batch_size * seq_length * n_embd * n_embd * 3  # Q, K, V projections
    attention_flops += 2 * batch_size * n_head * seq_length * seq_length * d_head  # attention

    # FLOPs for feed-forward layers
    ff_flops = 2 * batch_size * seq_length * n_embd * d_ff

    total_flops = (attention_flops + ff_flops) * n_layer
    return total_flops

def benchmark_model(model, data_loader, n_head, n_embd, n_layer, seq_length, warmup=3, iterations=10):
    model.eval()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        X, Y = next(iter(data_loader))
        with torch.no_grad():
            _ = model(X, Y)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    total_time = 0
    for _ in range(iterations):
        X, Y = next(iter(data_loader))
        start_event.record()
        with torch.no_grad():
            _ = model(X, Y)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    
    avg_time = total_time / iterations
    flops = calculate_flops(batch_size, seq_length, n_layer, n_head, n_embd)
    tflops = flops / (avg_time / 1000) / 1e12  # Convert to TFLOPs/s
    return tflops

seq_lengths = [128, 256, 512, 1024, 2048]  # Add or modify sequence lengths as needed
flops_per_seq_length = []

for seq_length in seq_lengths:
    train_loader = data_loader('train', batch_size, seq_length)
    
    model = GPTLanguageModel(vocab_size, n_embd, seq_length, n_head, n_layer, dropout, device)
    model = model.to(device)
    
    tflops = benchmark_model(model, train_loader, n_head, n_embd, n_layer, seq_length)
    flops_per_seq_length.append(tflops)
    print(f"Sequence Length: {seq_length}, Performance: {tflops:.2f} TFLOPs/s")
    
    del model
    torch.cuda.empty_cache()

# Save results to CSV
csv_path = os.path.join(output_dir, 'benchmark_results_seq_length.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sequence Length', 'TFLOPs/s'])
    for seq_length, flops in zip(seq_lengths, flops_per_seq_length):
        writer.writerow([seq_length, flops])

print(f"Benchmark results saved to {csv_path}")