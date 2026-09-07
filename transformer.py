# building a transformer model from scratch in PyTorch, based on Andrej Karpathy's "miniGPT" implementation
# trained on the TinyStories dataset on Hugging Face
# using a BPE tokenizer with a vocabulary size of 1,024 tokens

import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from pathlib import Path
import pyarrow.parquet as pq

# hyperparameters tuned for the larger TinyStories dataset
batch_size = 64 # number of sequences processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 1500 # maximum training iterations
learning_rate = 3e-4 # learning rate for the optimizer (alpha: controls step size)
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # defaults training to Apple GPU
eval_iters = 50 # keep validation checks useful without adding much runtime
n_embd = 256 # embedding dimension
n_head = 4 # number of heads in MHA
n_layer = 4 # number of transformer blocks (MHA + FFN)
dropout = 0.1 # regularization to reduce overfitting
weight_decay = 0.01 # regularization applied by AdamW optimizer
patience = 3 # stops training if validation loss doesn't improve for this many evaluations

# set seed for reproducibility
torch.manual_seed(2026)

# Load a manageable local subset of data
max_train_stories = 100_000
max_validation_stories = 10_000
data_path = Path(__file__).resolve().parent / 'hf_stories'
stories = pq.read_table(
    data_path,
    columns=['text'],
).slice(0, max_train_stories + max_validation_stories).column('text').to_pylist()

# splits data
train_texts = stories[:max_train_stories]
validation_texts = stories[max_train_stories:]

# cleans data
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

train_texts = [clean_text(text) for text in train_texts]
validation_texts = [clean_text(text) for text in validation_texts]
cleaned_train_text = "\n\n".join(train_texts)
cleaned_validation_text = "\n\n".join(validation_texts)

# create embeddings and tokenizer

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

## 1. Initialize BPE model and byte-level handling
## merges most common byte pairs into single tokens = compact representation of text

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()

## 2. Configure trainer for exactly 1,024 tokens
trainer = trainers.BpeTrainer(
    vocab_size = 1024,
    special_tokens = ["<|endoftext|>"],
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
)

## 3. Train directly on the TinyStories training examples
tokenizer.train_from_iterator(train_texts, trainer = trainer)

## 4. Define encode and decode functions matching Karpathy's interface
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Encode the official TinyStories train and validation splits separately.
train_data = torch.tensor(encode(cleaned_train_text), dtype=torch.long)
test_data = torch.tensor(encode(cleaned_validation_text), dtype=torch.long)

vocab_size = tokenizer.get_vocab_size()

# generate small batches of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# periodically pauses training to produce a low-variance estimate of model performance 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(weight, dim=-1)
        out = self.dropout(attn_weights) @ self.value(x)
        if return_attn:
            return out, attn_weights
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        if return_attn:
            head_outputs = [h(x, return_attn=True) for h in self.heads]
            out = torch.cat([result[0] for result in head_outputs], dim=-1)
            attn_weights = torch.stack([result[1] for result in head_outputs], dim=1)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        if return_attn:
            return out, attn_weights
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # multiplying by 4 as seen in the original transformer paper, to increase model capacity
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_attn=False):
        if return_attn:
            attention_output, attn_weights = self.sa(self.ln1(x), return_attn=True)
            x = x + attention_output
        else:
            x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0., std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0., std = 0.02)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

# running the model
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(
    model.parameters(), lr = learning_rate, weight_decay = weight_decay
)

# keep the best validation checkpoint and stop after repeated regressions.
best_test_loss = float('inf')
best_model_state = None
evaluations_without_improvement = 0
for iter in range(max_iters):
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
        if losses['test'] < best_test_loss:
            best_test_loss = losses['test']
            best_model_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
            if evaluations_without_improvement >= patience:
                print('Early stopping: validation loss stopped improving.')
                break

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    # CHANGED: prevent unstable updates from making validation loss deteriorate quickly.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

if best_model_state is not None:
    model.load_state_dict({name: value.to(device) for name, value in best_model_state.items()})

context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))
#open('example.txt', 'w').write(decode(m.generate(context, max_new_tokens = 10000)[0].tolist()))