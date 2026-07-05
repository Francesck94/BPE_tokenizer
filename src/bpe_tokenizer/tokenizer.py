"""
Contains the GPT4 Tokenizer class and a few common helper functions.
"""

from typing import Optional
from .utils import replace_control_characters, render_token
import regex as re


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def _merge_stats(stats_list):
    """
    Combine a list of dict with pair frequency in a single dict.
    """
    full_stats = {}
    for s in stats_list:
        for k, v in s.items():
            if k not in full_stats.keys():
                full_stats[k] = v
            else:
                current_value = full_stats.get(k)
                full_stats[k] = current_value + v
                del current_value
    return full_stats


# gpt4_merges = recover_merges(tiktoken.get_encoding("cl100k_base")._mergeable_ranks)
ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"


class Tokenizer:
    """A BPE tokenizer that uses regular expressions to find pairs to merge"""

    def __init__(self, verbose=False):

        self.merges = {}  # (int, int) -> int
        self.special_tokens_str = [
            ENDOFTEXT,
            FIM_PREFIX,
            FIM_MIDDLE,
            FIM_SUFFIX,
            ENDOFPROMPT,
        ]
        self.special_tokens = {}
        self.vocab = self._build_vocab()  # int -> bytes
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        self.verbose = verbose
        # self.merges = self._shuffle_merges()
        self.vocab = self._build_vocab()  # int -> bytes

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        Create the merges and the vocab.
        """
        pattern = re.compile(self.pattern)

        # split the text using the pattern
        text_chunks = re.findall(pattern, text)

        # convert each chunk to utf-8 bytes
        ids_chunks = []
        for tc in text_chunks:
            ids_chunks.append(list(map(int, tc.encode("utf-8"))))

        last_vocab = max(list(self.vocab.keys())) + 1

        merges_to_do = vocab_size - 256
        for i in range(0, merges_to_do):
            # get the stats for each chunk
            #
            stats_for_chunk = [get_stats(ids) for ids in ids_chunks]

            # #TODO: fix this part
            # stats = {}
            # for ids in ids_chunks:
            #     stats = get_stats(ids, stats)
            stats = _merge_stats(stats_for_chunk)

            if len(stats) == 0:
                if verbose:
                    print("No more pairs to merge, stopping early.")
                    break

            top_pair = max(stats, key=stats.get)

            new_token = last_vocab + i

            ids_chunks = [merge(ids, top_pair, new_token) for ids in ids_chunks]

            self.merges[top_pair] = new_token

        # create special tokens dict
        self.special_tokens = {
            s: last_vocab + merges_to_do + i
            for i, s in enumerate(self.special_tokens_str)
        }
        # create new vocab
        self.vocab = self._build_vocab()

    def encode(self, text, allowed_special=None):
        # Tokenizer can encode a string into a list of integers

        text_chunks = []

        special_tokens = {}

        # split the text into segments that are either special tokens or not, and filter out empty segments
        segments, special_tokens = self._handle_special_tokens_in_encode(
            text, allowed_special
        )

        for s in segments:
            if s in special_tokens.keys():
                text_chunks.append(s)
            else:
                text_chunks.extend(re.findall(self.pattern, s))

        if self.verbose:
            print(f"text_chunks: {text_chunks}")

        # convert each chunk to utf-8 bytes
        if allowed_special is None:
            tokens_chunks = [list(map(int, t.encode("utf-8"))) for t in text_chunks]
        else:
            # if allowed_special is not None, we have to be careful to not encode the special tokens as utf-8 bytes, but rather use their assigned integer ids. So we check if each chunk is a special token, and if so we use the assigned integer id, otherwise we encode as utf-8 bytes.
            tokens_chunks = []
            for t in text_chunks:
                if t in special_tokens.keys():
                    tokens_chunks.append([special_tokens[t]])
                else:
                    tokens_chunks.append(list(map(int, t.encode("utf-8"))))

        # tokens_chunks_shuffled = self._apply_shuffle_in_encode(tokens_chunks)
        tokens_chunks_shuffled = tokens_chunks

        # apply the merges to each chunk
        if self.verbose:
            print(f"applying merges...")

        sorted_merges = sorted(
            self.merges.items(), key=lambda x: x[1]
        )  # sort by the new token id, which is the value in the merges dict

        tokens_enc = map(
            lambda tok_chunk: self._apply_merges_on_chunk(sorted_merges, tok_chunk),
            tokens_chunks_shuffled,
        )

        ids = [t for ts in tokens_enc for t in ts]

        return ids

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        original_str = b"".join([self.vocab[idx] for idx in ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str

    def tokenize(self, text, allowed_special=None):
        # Tokenizer can tokenize a string into a list of strings (tokens)
        ids = self.encode(text, allowed_special)
        text_chunks = [self.vocab[idx].decode("utf-8", errors="replace") for idx in ids]
        return text_chunks

    def _apply_merges_on_chunk(self, sorted_merges, tok_chunk):
        found_merge = True
        tok_chunk_pair = list(zip(tok_chunk[:-1], tok_chunk[1:]))
        while found_merge:
            found_merge = False
            for pair_to_merge, new_token in sorted_merges:
                if pair_to_merge in tok_chunk_pair:
                    # if the pair to merge is in the token list, then merge it
                    tok_chunk = merge(tok_chunk, pair_to_merge, new_token)
                    tok_chunk_pair = list(zip(tok_chunk[:-1], tok_chunk[1:]))
                    found_merge = True
                    break  # we have to break here because the merges have to be applied sequentially, and we can't apply multiple merges at the same time because they might interfere with each other. For example, if we have merges (a, b) -> x and (x, c) -> y, and our token chunk is [a, b, c], we have to first merge (a, b) to get [x, c], and then merge (x, c) to get [y]. If we tried to apply both merges at the same time, we would not know whether to merge (a, b) or (x, c) first.
        return tok_chunk

    def register_special_tokens(self, special_tokens, verbose=False):
        self.special_tokens = special_tokens

        # create new pattern
        for special_token, token_idx in special_tokens.items():
            escaped_token = re.escape(
                special_token
            )  # escape special characters in the token
            self.pattern = rf"{escaped_token}|" + self.pattern
            self.vocab[token_idx] = special_token.encode("utf-8")

        if verbose:
            print(f"New pattern: {self.pattern}")
        self.regex_pattern = re.compile(self.pattern)


    def _create_special_pattern(self, special_tokens):
        # create a regex pattern that matches any of the special tokens, sorted by length
        special_pattern = "|".join(
            "(" + re.escape(s) + ")"
            for s in sorted(special_tokens.keys(), key=len, reverse=True)
        )
        return special_pattern

    def _handle_special_tokens_in_encode(self, text, allowed_special):
        # if allowed_special is "all", we want to treat all special tokens as indivisible units
        # if allowed_special is a list of special tokens, we want to treat those as indivisible units
        # otherwise, we don't treat any special tokens as indivisible units

        special_tokens = {}
        if allowed_special == "all":
            special_tokens = self.special_tokens
        elif isinstance(allowed_special, (list, set)):
            special_tokens = {s: self.special_tokens[s] for s in allowed_special}
        else:
            return [text], special_tokens

        # split the text into segments that are either special tokens or not, and filter out empty segments
        special_pattern = self._create_special_pattern(special_tokens)
        segments = re.split(special_pattern, text, ignore_unused=True)
        segments = [s for s in segments if s is not None and s != ""]
        return segments, special_tokens

    def save(self, file_prefix, save_path=None):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        if save_path is not None:
            model_file = f"{save_path}/{file_prefix}.model"
        else:
            model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        if save_path is not None:
            vocab_file = f"{save_path}/{file_prefix}.vocab"
        else:
            vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


if __name__ == "__main__":
    import os

    tokenizer = Tokenizer(verbose=False)

    # print(vars(tokenizer))

    print(len(tokenizer.vocab))

    # for idx in range(0, 256 + 10):
    #     print(f"{idx}: {tokenizer.vocab[idx].decode('utf-8', errors='replace')}")
    # print(tokenizer.vocab)

    # module path

    filepath = os.path.join(
        os.path.dirname(__file__), "..", "example_data", "wikipedia_page.txt"
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
    import time

    start = time.perf_counter()
    text_test = "Machine learning is a beautiful science! <|endoftext|>"

    tokenizer.train(text, vocab_size=1000)
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
