import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

    
class FA2Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B,T, C = x.shape # Batch, Time, Channels
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # q = q.unsqueeze(0)
        # k = k.unsqueeze(0)
        # v = v.unsqueeze(0) 

        q = q.view(B, T, 1, self.head_size)
        k = k.view(B, T, 1, self.head_size)
        v = v.view(B, T, 1, self.head_size)

        # print("x:",x.shape)
        # print("Query:",q.shape)
        # print("Key:",k.shape)
        # print("Value:",v.shape)
        out = flash_attn_func(q, k, v)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout,block_size):
        super().__init__()
        self.heads = nn.ModuleList([FA2Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        out = out.view(B, T, -1)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout,block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout,block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def forward(self, idx, targets=None):
    #     B, T = idx.shape

    #     tok_emb = self.token_embedding_table(idx)
    #     pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
    #     x = tok_emb + pos_emb
    #     x = self.blocks(x)
    #     x = self.ln_f(x)
    #     logits = self.lm_head(x)

    #     if targets is None:
    #         loss = None
    #     else:
    #         B, T, C = logits.shape
    #         logits = logits.view(B*T, C)
    #         targets = targets.view(B*T)
    #         loss = F.cross_entropy(logits, targets)

    #     return logits, loss

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Convert idx to LongTensor if it's not already
        idx = idx.to(torch.long)

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device, dtype=torch.long))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # loss = F.cross_entropy(logits, targets)
            loss = F.cross_entropy(logits.float(), targets.long())

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx