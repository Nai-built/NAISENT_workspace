"""
Byte Pair Encoding (BPE) Tokenizer

A minimal, performance-friendly BPE tokenizer that learns subword tokens
from training text. Works in 3 steps:
  1. train()  - Learn merge rules from a text corpus
  2. encode() - Convert text to a list of integer token IDs
  3. decode() - Convert token IDs back to text

The base vocabulary is all 256 individual bytes, so any text can be
encoded without "unknown token" errors. Training discovers frequent
byte-pair merges and adds them as new tokens (IDs 256, 257, ...).
"""

import json
from array import array


class BPETokenizer:

    def __init__(self):
        # Ordered list of merge rules: each entry is (token_a, token_b)
        # The index in this list + 256 = the new token's ID
        self.merges = []
        # Maps token ID -> bytes for that token (for decoding)
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, target_vocab_size: int = 512, verbose: bool = False):
        """Learn BPE merge rules from training text.

        Args:
            text:              The training corpus as a string.
            target_vocab_size: Desired total vocabulary size (>= 256).
            verbose:           Print progress during training.
        """
        assert target_vocab_size >= 256, "Vocab size must be >= 256 (the byte base)"
        num_merges = target_vocab_size - 256

        # Start with raw bytes as the initial token sequence
        token_ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            # Count every adjacent pair
            pair_counts = {}
            for j in range(len(token_ids) - 1):
                pair = (token_ids[j], token_ids[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Nothing worth merging (frequency 1 means no compression gain)
            if pair_counts[best_pair] < 2:
                break

            # Create the new token
            new_id = 256 + len(self.merges)
            self.merges.append(best_pair)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Replace all occurrences of the pair in the token sequence
            token_ids = self._merge_pair(token_ids, best_pair, new_id)

            if verbose:
                pair_text = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  merge {i + 1}/{num_merges}: "
                    f"{best_pair} -> {new_id}  ({pair_counts[best_pair]}x)  "
                    f"'{pair_text}'"
                )

    @staticmethod
    def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `ids` with `new_id`."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    # ------------------------------------------------------------------
    # Encoding (text -> token IDs)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        token_ids = list(text.encode("utf-8"))

        # Apply each merge rule in the order they were learned
        for pair, merge_id in zip(self.merges, range(256, 256 + len(self.merges))):
            token_ids = self._merge_pair(token_ids, pair, merge_id)

        return token_ids

    def to_one_hot(self, ids: list[int]) -> array:
        """Convert a list of token IDs to a flat one-hot float array.

        Each token becomes a zero vector of length vocab_size with a
        single 1.0 at position [token_id]. The result is a flat
        array("f") of length len(ids) * vocab_size, matching the
        input format expected by CNL models.
        """
        v = self.vocab_size
        result = array("f", [0.0] * (len(ids) * v))
        print(ids)
        for i, token_id in enumerate(ids):
            print(token_id)
            result[i * v + token_id] = 1.0
        return result, len(ids)

    def from_one_hot(self, flat: array) -> list[int]:
        """Convert a flat one-hot float array back to token IDs.

        Picks the index of the highest value in each vocab_size chunk,
        matching how formSequence works in the original tokenizer.
        """
        v = self.vocab_size
        num_tokens = len(flat) // v
        ids = []
        for i in range(num_tokens):
            chunk = flat[i * v : i * v + v]
            ids.append(max(range(v), key=lambda j: chunk[j]))
        return ids

    # ------------------------------------------------------------------
    # Decoding (token IDs -> text)
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save the tokenizer (merge rules) to a JSON file."""
        data = {"merges": self.merges}
        with open(path, "w") as f:
            json.dump(data, f)

    def save_readable(self, path: str):
        """Save merges as human-readable strings to a JSON file.

        Each entry shows: 'token_a' + 'token_b' -> 'merged_result' (id: N)
        """
        readable = []
        for idx, (a, b) in enumerate(self.merges):
            token_id = 256 + idx
            str_a = self.vocab[a].decode("utf-8", errors="replace")
            str_b = self.vocab[b].decode("utf-8", errors="replace")
            str_merged = self.vocab[token_id].decode("utf-8", errors="replace")
            readable.append({
                "id": token_id,
                "pair": [str_a, str_b],
                "merged": str_merged,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(readable, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """Load merge rules from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.merges = [tuple(p) for p in data["merges"]]
        # Rebuild vocab from merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for idx, (a, b) in enumerate(self.merges):
            self.vocab[256 + idx] = self.vocab[a] + self.vocab[b]


# ======================================================================
# Demo
# ======================================================================

# if __name__ == "__main__":
#     # Sample training text
#     training_text = (
#         "the cat sat on the mat. the cat sat on the hat. "
#         "the dog sat on the mat. the dog sat on the log. "
#         "a cat and a dog sat on a mat. the cat and the dog are friends. "
#     ) * 20  # repeat to give BPE enough frequency signal

#     print("=" * 60)
#     print("BPE Tokenizer Demo")
#     print("=" * 60)

#     tok = BPETokenizer()
#     print(f"\nTraining on {len(training_text)} characters...")
#     tok.train(training_text, target_vocab_size=300, verbose=True)
#     print(f"\nVocab size after training: {tok.vocab_size}")

#     # Show the learned vocabulary (non-byte tokens only)
#     print("\n--- Learned tokens ---")
#     for token_id in range(256, tok.vocab_size):
#         token_bytes = tok.vocab[token_id]
#         print(f"  {token_id}: '{token_bytes.decode('utf-8', errors='replace')}'")

#     # Encode / decode round-trip
#     test = "the cat sat on the mat."
#     encoded = tok.encode(test)
#     decoded = tok.decode(encoded)

#     print(f"\n--- Round-trip test ---")
#     print(f"  Original:  '{test}'")
#     print(f"  Encoded:   {encoded}  ({len(encoded)} tokens)")
#     print(f"  Decoded:   '{decoded}'")
#     print(f"  Match:     {test == decoded}")

#     # Compression ratio
#     raw_bytes = len(test.encode("utf-8"))
#     print(f"\n  Raw bytes:    {raw_bytes}")
#     print(f"  Token count:  {len(encoded)}")
#     print(f"  Compression:  {raw_bytes / len(encoded):.1f}x")

#     # Save and reload test
#     tok.save("bpe_model.json")
#     tok.save_readable("bpe_model_readable.json")
#     tok2 = BPETokenizer()
#     tok2.load("bpe_model.json")
#     assert tok2.encode(test) == encoded
#     print("\n  Save/load round-trip: OK")
#     print("  Readable merges saved to: bpe_model_readable.json")

if __name__ == "__main__":
    # Sample training text
    training_text = (
        "USER: Hello, who are you?\nMODEL: I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hi.\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hi?\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hi\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hello.\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: hello.\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: hello\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hello, who are you?\nMODEL: I am NAISENT.<END OF SEQUENCE>\n"
        "USER: hello, who are you\nMODEL: I am NAISENT.<END OF SEQUENCE>\n"
        "USER: Hello there, who you are ?\nMODEL: Hello, I am NAISENT.<END OF SEQUENCE>\n"
    ) * 2000  # repeat to give BPE enough frequency signal

    print("=" * 60)
    print("BPE Tokenizer Demo")
    print("=" * 60)

    tok = BPETokenizer()
    print(f"\nTraining on {len(training_text)} characters...")
    tok.train(training_text, target_vocab_size=2000, verbose=True)
    print(f"\nVocab size after training: {tok.vocab_size}")

    # Show the learned vocabulary (non-byte tokens only)
    print("\n--- Learned tokens ---")
    for token_id in range(256, tok.vocab_size):
        token_bytes = tok.vocab[token_id]
        print(f"  {token_id}: '{token_bytes.decode('utf-8', errors='replace')}'")

    # Encode / decode round-trip
    test = "the cat sat on the mat."
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)

    print(f"\n--- Round-trip test ---")
    print(f"  Original:  '{test}'")
    print(f"  Encoded:   {encoded}  ({len(encoded)} tokens)")
    print(f"  Decoded:   '{decoded}'")
    print(f"  Match:     {test == decoded}")

    # Compression ratio
    raw_bytes = len(test.encode("utf-8"))
    print(f"\n  Raw bytes:    {raw_bytes}")
    print(f"  Token count:  {len(encoded)}")
    print(f"  Compression:  {raw_bytes / len(encoded):.1f}x")

    # Save and reload test
    tok.save("bpe_model.json")
    tok.save_readable("bpe_model_readable.json")
    tok2 = BPETokenizer()
    tok2.load("bpe_model.json")
    assert tok2.encode(test) == encoded
    print("\n  Save/load round-trip: OK")
    print("  Readable merges saved to: bpe_model_readable.json")
