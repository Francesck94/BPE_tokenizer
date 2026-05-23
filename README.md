# BPE Tokenizer

A from-scratch implementation of a Byte Pair Encoding (BPE) tokenizer in Python, following the same approach used by GPT-4's tokenizer.

## What is Byte Pair Encoding?

Byte Pair Encoding is a subword tokenization algorithm widely used in modern large language models (GPT-2, GPT-4, LLaMA, etc.). It works by iteratively merging the most frequent pair of consecutive tokens in a corpus until a desired vocabulary size is reached.

The algorithm starts from raw UTF-8 bytes (256 base tokens) and builds up a vocabulary of subword units through repeated merging:

1. Encode the training text as a sequence of UTF-8 bytes
2. Count all consecutive byte pairs in the sequence
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until the target vocabulary size is reached

This produces a vocabulary that balances between character-level and word-level representations — common words become single tokens, while rare words are split into meaningful subword pieces.

## Features

- **Training** — Learn BPE merges from any text corpus to build a custom vocabulary
- **Encoding** — Tokenize text into a sequence of integer token IDs
- **Decoding** — Reconstruct text from token IDs
- **Special tokens** — Support for special tokens (`<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`)
- **Regex pre-tokenization** — Uses the GPT-4 splitting pattern to segment text before applying merges, preventing merges across word boundaries
- **Save/Load** — Serialize trained tokenizers to `.model` and `.vocab` files

## Package Structure

```
src/bpe_tokenizer/
  tokenizer.py   # Tokenizer class with train, encode, decode, save, load
  utils.py       # Helper functions for rendering tokens
examples/
  train_tokenizer.py  # Example: train a tokenizer on sample data
example_data/
  wikipedia_page.txt  # Sample training corpus
notebooks/
  download_data.ipynb # Notebook to download training data
tests/
  test_tokenizer.py   # Unit tests
```

## Usage

```python
from bpe_tokenizer.tokenizer import Tokenizer

# Train a tokenizer
tokenizer = Tokenizer()
tokenizer.train(text, vocab_size=2000)

# Encode text to token IDs
tokens = tokenizer.encode("Hello world!", allowed_special="all")

# Decode token IDs back to text
text = tokenizer.decode(tokens)

# Save and load
tokenizer.save("my_tokenizer", save_path=".")
tokenizer.load("my_tokenizer.model")
```

## Installation

```bash
pip install -e .
```

Requires Python >= 3.13 and the `regex` package.
