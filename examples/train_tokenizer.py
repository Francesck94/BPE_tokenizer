import os
from bpe_tokenizer.tokenizer import Tokenizer
import time

start = time.perf_counter()

tokenizer = Tokenizer(verbose=False)

# print(vars(tokenizer))

print(len(tokenizer.vocab))

# for idx in range(0, 256 + 10):
#     print(f"{idx}: {tokenizer.vocab[idx].decode('utf-8', errors='replace')}")
# print(tokenizer.vocab)

# module path

filepath = os.path.join(
    os.path.dirname(__file__), "example_data", "wikipedia_page.txt"
)
print(f"filepath: {filepath}")
with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# text = text[:1000]
# #print(f"final vocab: {tokenizer.vocab}")


# enc_gpt = tiktoken.get_encoding("cl100k_base")

# text = "hello world!!!? (안녕하세요!) lol123 😉"
# text = "<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM"
#     text = """
# <|endoftext|>Hello world this is one document
# <|endoftext|>And this is another document
# <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
# <|endoftext|>Last document!!! 👋<|endofprompt|>
# """.strip()
# text = "a"
# print(f"Original text: {text}")

text_test = "Machine learning is a beautiful science! <|endoftext|>"

tokenizer.train(text, vocab_size=2000)
tokenizer.save("tokenizer", save_path=".")
tokens = tokenizer.encode(text_test, allowed_special="all")
print(f"Tokens: {tokens}")
elapsed = time.perf_counter() - start
print(f"Encoding took {elapsed:.4f}s")

# tokens_gpt_encoded = enc_gpt.encode(text, allowed_special="all")
# elapsed = time.perf_counter() - start
# print(f"TikToken tokenizer: {tokens_gpt_encoded}")
# print(f"Custom tokenizer: {tokens}")
decoded = tokenizer.decode(tokens)

print(f"Decoded text: {decoded}")
print(f"is decoded text same as original? {decoded == text_test}")


tokenizer_dir = os.path.join(os.path.dirname(__file__), "..", "saved_tokenizers")

tokenizer.save("tokenizer", save_path=tokenizer_dir)
