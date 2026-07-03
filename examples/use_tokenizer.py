from bpe_tokenizer.tokenizer import Tokenizer
import os

tokenizer = Tokenizer(verbose=False)
tokenizer.load(os.path.join(os.path.dirname(__file__), 'tokenizer.model'))

text = "Machine learning is a beautiful science! <|endoftext|>"
tokens = tokenizer.tokenize(text, allowed_special="all")
print(f"Tokens: {tokens}")