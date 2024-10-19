import torch
import torch.nn as nn
from torch.nn import functional as F
# from gpt import GPTLanguageModel
from gpt import GPTLanguageModel
from benchmarker import Benchmarker
import os
from datasets import load_dataset

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 200
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Load dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')

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
def data_loader(split, batch_size):
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

train_loader = data_loader('train', batch_size)
val_loader = data_loader('val', batch_size)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(loader):
            if k >= eval_iters:
                break
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create output directory
output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

# Create a Benchmarker instance
benchmarker = Benchmarker(os.path.join(output_dir, 'gpt_benchmark.csv'))

# Start benchmarking
benchmarker.start()

try:
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        for xb, yb in train_loader:
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update the current loss in the benchmarker
            benchmarker.update_loss(loss.item())

    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

finally:
    # Stop benchmarking
    benchmarker.stop()

    # Plot benchmark results
    benchmarker.plot_benchmark(output_dir)

print(f"Benchmarking data and plot saved in {output_dir}")
