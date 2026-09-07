# TinyStories Transformer From Scratch

A compact, decoder-only Transformer language model written in PyTorch. This project follows the educational progression in [Andrej Karpathy's *Let's reproduce GPT-2* walkthrough](https://www.youtube.com/watch?v=kCc8FmEb1nY), then deliberately changes two important choices:

- **Data:** it trains on Hugging Face's [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) corpus rather than the Shakespeare text used in the walkthrough.
- **Tokenization:** it learns a **byte-level BPE** vocabulary instead of treating each character as one token.

The point is not to reproduce a production GPT model. It is a readable implementation for understanding the mechanics shared by modern Transformer language models: subword tokenization, causal self-attention, multi-head attention, residual pathways, normalization, next-token likelihood, and autoregressive sampling.

## Repository contents

| Path | Role |
| --- | --- |
| `transformer.py` | Complete data preparation, BPE training, model definition, training loop, validation/early stopping, and text generation. |
| `hf_stories` | Local Apache Parquet snapshot of TinyStories. It has one `text` column, 529,930 rows, and is about 237 MiB. The script reads its first 110,000 rows. |
| `Attention.pdf` | The supplied copy of Vaswani et al., *Attention Is All You Need* (2017), with reading annotations. It is the conceptual reference for the architecture discussion below. |
| `example.txt` | A saved sample of generated TinyStories-style text. Its imperfect grammar and word fragments are useful evidence of the model's small scale and limited training budget. |

> **GitHub note:** GitHub rejects ordinary files over 100 MiB, so `hf_stories` is excluded in this repository's `.gitignore`. Keep it local or fetch it as part of setup. If it was committed previously, run `git rm --cached hf_stories` from the repository root before pushing; this preserves your local file.

## Quick start

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch tokenizers pyarrow
```

PyTorch uses Apple Metal Performance Shaders (`mps`) automatically when it is available; otherwise this implementation uses CPU. It does not currently select CUDA explicitly.

### 2. Put the dataset beside the script

The training script expects this layout:

```text
.
├── transformer.py
└── hf_stories       # Apache Parquet file with a text column
```

Obtain a TinyStories Parquet export from the [dataset page](https://huggingface.co/datasets/roneneldan/TinyStories) and save it as `hf_stories`. The local artifact used while developing this project contains 529,930 stories; the code takes the first 100,000 for training and the next 10,000 for validation. This is a contiguous local split, **not** TinyStories' official train/validation split.

### 3. Train and sample

```bash
python transformer.py
```

The program prints its parameter count, periodic train/validation losses, and a 500-token continuation sampled from an all-zero one-token prompt. It keeps the best validation weights in memory, restores them after training, and then generates. It does **not** write a model checkpoint or tokenizer file to disk. To save a longer sample manually, uncomment the final `open('example.txt', ...)` line.

## Data and BPE pipeline

The code intentionally keeps the preprocessing visible:

1. Read the `text` column from `hf_stories` with PyArrow.
2. Use 100,000 stories for training and 10,000 following stories for validation.
3. Collapse each story's internal whitespace to single spaces, then join stories with blank-line separators.
4. Train a Hugging Face `tokenizers` BPE model on the cleaned **training stories** only.
5. Configure byte-level pre-tokenization and decoding, include the full byte alphabet, and learn a vocabulary of 1,024 entries, including `<|endoftext|>`.
6. Encode the concatenated train and validation text into `torch.long` token streams.
7. Draw random contiguous windows of 256 input tokens; targets are the same windows shifted one token to the right.

`<|endoftext|>` is reserved in the BPE vocabulary, but the current script does not insert it between stories; blank-line separators are the only document boundaries in the encoded stream.

### Why BPE instead of character tokens?

Karpathy's character tokenizer makes the training objective transparent: one character is one vocabulary item. It also creates long sequences and cannot represent common words or morphemes as reusable units. Byte-pair encoding starts with byte-level symbols and repeatedly merges frequent adjacent pairs. The resulting vocabulary learns useful subword chunks while retaining byte-level coverage for arbitrary text.

Here, `vocab_size = 1024` is deliberately small for an instructional run. BPE normally shortens a sentence relative to character tokenization, so the 256-token context can express more text. At the same time, a 1,024-token vocabulary and 1,500 training updates limit sample quality; `example.txt` should be read as a qualitative sample rather than a benchmark result.

## Model at a glance

```text
BPE token IDs
    │
    ├── token embedding
    ├── learned position embedding
    └── element-wise sum
              │
              ▼
  4 × pre-norm Transformer block
      LayerNorm → masked multi-head self-attention → residual add
      LayerNorm → position-wise MLP → residual add
              │
              ▼
     final LayerNorm → vocabulary projection → next-token logits
```

| Component | Project setting | What it does |
| --- | ---: | --- |
| Context window | 256 tokens | Maximum context the model can attend to; generation crops longer histories to this window. |
| Vocabulary | 1,024 BPE tokens | Maps learned byte-level subwords to IDs and back. |
| Embedding width (`d_model`) | 256 | Hidden-vector width for tokens, positions, attention, and the MLP. |
| Transformer blocks | 4 | Repeated attention-plus-MLP computation. |
| Attention heads | 4 | Parallel 64-dimensional attention subspaces (`256 / 4`). |
| MLP width | 1,024 | A `4 × d_model` expansion, then ReLU, projection back to 256, and dropout. |
| Dropout | 0.1 | Applied to attention probabilities and projected attention/MLP outputs. |
| Optimizer | AdamW | Learning rate `3e-4`, weight decay `0.01`. |
| Training budget | 1,500 updates | Batch size 64; evaluate every 50 updates. |
| Stability controls | Gradient clipping + early stopping | Clips global gradient norm to 1.0; restores best validation state after 3 non-improving evaluations. |

### The learning objective

For every position, the model returns a distribution over the next BPE token. Training minimizes cross-entropy over all batch positions:

$$
\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\log p_\theta\left(x_{b,t+1}\mid x_{b,\leq t}\right).
$$

During generation, the model keeps only the latest 256 tokens, takes the final-position logits, applies softmax, samples one token with `torch.multinomial`, appends it, and repeats. This is an autoregressive decoder: output token $t+1$ conditions only on earlier generated tokens.

## Connecting the code to *Attention Is All You Need*

`Attention.pdf` is the supplied primary-paper reference. The model implements the language-model side of the original Transformer ideas, with intentional simplifications.

### Scaled dot-product causal self-attention

Each `Head` learns separate linear projections for queries, keys, and values:

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V.
$$

It computes attention scores and scales them by the square root of the head dimension:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

`M` is a lower-triangular causal mask. Future positions receive $-\infty$ before softmax, so their probabilities become zero. Without this mask, training could leak the token the model is supposed to predict.

### Multi-head attention

`MultiHeadAttention` runs four `Head` modules in parallel, concatenates their outputs, and projects the result back to the 256-dimensional model space. Separate heads can learn different compatibility patterns; a single attention distribution would force all relationships through one averaged view.

### Position-wise MLP, residual paths, and layer normalization

Attention mixes information **between positions**. The `FeedForward` module then applies the same two-layer MLP independently at every position:

$$
\operatorname{FFN}(x) = \operatorname{ReLU}(xW_1 + b_1)W_2 + b_2.
$$

Each block adds attention and MLP outputs back to the residual stream. This implementation is **pre-norm**:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The original paper described post-norm residual sublayers, `LayerNorm(x + Sublayer(x))`. Pre-norm is a common later design because it improves optimization stability in deep Transformer stacks.

### Position information

Self-attention alone is permutation-invariant: it has no notion of first, second, or last token. The paper uses fixed sinusoidal encodings and reports comparable results with learned positional embeddings. This project chooses the learned option:

```python
x = token_embedding(idx) + position_embedding(arange(T))
```

That gives every location from 0 through 255 a trainable vector, but it also fixes the trained positional range to `block_size`.

## What this is—and is not

This is a **decoder-only GPT-style language model**, not the full encoder-decoder translation Transformer in the 2017 paper.

- It has masked self-attention, but no encoder and no encoder-decoder cross-attention.
- It predicts the next token in one text stream; it does not take a source sequence and translate it to a target sequence.
- It uses learned position embeddings rather than the paper's default sinusoidal encodings.
- It uses pre-norm blocks and separate input/output embedding matrices; the original paper's architecture used post-norm and shared some embedding/softmax weights.
- Full causal attention costs $O(T^2)$ in sequence length. That is manageable for the 256-token context here, but becomes the central scaling pressure for longer contexts.

Those differences are useful rather than accidental: the script isolates the parts needed to understand a modern autoregressive Transformer while preserving the core attention equations and training objective.

## References

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Local annotated copy: `Attention.pdf`.
- Andrej Karpathy, [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- Ronen Eldan and Yuanzhi Li, [TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories).
- Sennrich, Haddow, and Birch (2016), [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909), the BPE reference.
