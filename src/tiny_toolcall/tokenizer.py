"""8k BPE trained on the synth mix. Encode once, train on uint16 memmaps.

Pretokenization contract (load-bearing for grammar.py): JSON structural characters
{ } [ ] , : " are singleton tokens and never participate in merges, so no token
ever spans a value/structure boundary. Grammar-constrained decoding can therefore
force-feed structure exactly and only consult the model at choice points.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<tools>", "</tools>", "<query>", "</query>", "<call>", "</call>"]

STRUCTURAL = set('{}[],:"')
# words never contain structural chars or newlines; whitespace runs kept separate
_PRETOK = re.compile(r'[{}\[\],:"]|\n|[^\S\n]+|[^{}\[\],:"\s]+')
_SPECIAL = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def pretokenize(text: str) -> list[str]:
    return _PRETOK.findall(text)


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.id_to_tok = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self._encode_word = lru_cache(maxsize=65536)(self._encode_word_uncached)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_word_uncached(self, word: str) -> tuple[int, ...]:
        if word in self.vocab:  # whole-word hit (structural singletons land here)
            return (self.vocab[word],)
        toks = list(word)
        while len(toks) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(toks) - 1):
                r = self.rank.get((toks[i], toks[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            toks = toks[:best_i] + [toks[best_i] + toks[best_i + 1]] + toks[best_i + 2 :]
        unk = self.vocab["<unk>"]
        return tuple(self.vocab.get(t, unk) for t in toks)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in _SPECIAL.split(text):
            if not part:
                continue
            if part in self.vocab and part in SPECIALS:
                ids.append(self.vocab[part])
            else:
                for word in pretokenize(part):
                    ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_tok.get(i, "") for i in ids if i not in (PAD, BOS, EOS))

    def token_str(self, i: int) -> str:
        return self.id_to_tok.get(i, "")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"vocab": self.vocab, "merges": self.merges}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        raw = json.loads(path.read_text())
        merges = [tuple(p) for p in raw["merges"]]
        return cls(raw["vocab"], merges)


def train_bpe(texts: list[str], vocab_size: int = 8192) -> BPETokenizer:
    """Word-level BPE: dedupe words first, merge within words only.

    Structural singletons are excluded from merging entirely; each still gets a
    vocab slot so encode() hits the whole-word fast path.
    """
    vocab: dict[str, int] = {s: i for i, s in enumerate(SPECIALS)}
    freq: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        for word in pretokenize(text):
            if len(word) == 1 and word in STRUCTURAL:
                if word not in vocab:
                    vocab[word] = len(vocab)
                continue
            freq[tuple(word)] += 1

    for w in freq:
        for ch in w:
            if ch not in vocab:
                vocab[ch] = len(vocab)

    merges: list[tuple[str, str]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for w, n in freq.items():
        for a, b in zip(w, w[1:]):
            pair_counts[(a, b)] += n

    words = dict(freq)
    while len(vocab) < vocab_size and pair_counts:
        (a, b), top = pair_counts.most_common(1)[0]
        if top < 2:
            break
        merged = a + b
        merges.append((a, b))
        if merged not in vocab:
            vocab[merged] = len(vocab)
        changed: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
        for w, n in words.items():
            hit = False
            for i in range(len(w) - 1):
                if w[i] == a and w[i + 1] == b:
                    hit = True
                    break
            if not hit:
                continue
            out: list[str] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            changed.append((w, tuple(out), n))
        for old, new, n in changed:
            for x, y in zip(old, old[1:]):
                pair_counts[(x, y)] -= n
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
            for x, y in zip(new, new[1:]):
                pair_counts[(x, y)] += n
            del words[old]
            words[new] = words.get(new, 0) + n
    return BPETokenizer(vocab, merges)
