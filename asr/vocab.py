"""Character vocabulary. Index 0 is always the CTC blank."""

import json
from pathlib import Path

BLANK = 0


class Vocab:
    def __init__(self, tokens):
        tokens = list(tokens)
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens in vocabulary")
        self.tokens = tokens
        self.stoi = {t: i + 1 for i, t in enumerate(tokens)}

    def __len__(self):
        return len(self.tokens) + 1  # + blank

    @classmethod
    def build(cls, texts, min_count=1):
        counts = {}
        for text in texts:
            for ch in text:
                counts[ch] = counts.get(ch, 0) + 1
        tokens = sorted(ch for ch, c in counts.items() if c >= min_count)
        return cls(tokens)

    def encode(self, text):
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise KeyError(f"character {exc.args[0]!r} not in vocabulary") from None

    def decode(self, ids):
        return "".join(self.tokens[i - 1] for i in ids if i != BLANK and 0 < i <= len(self.tokens))

    def save(self, path):
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self):
        return f"Vocab(size={len(self)}, tokens={''.join(self.tokens)!r})"
