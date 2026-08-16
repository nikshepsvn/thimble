"""Pack rendered examples to fixed-length uint16 arrays + name-decision sidecar.

Alignment trick: BPE is lossless and structural chars are singleton tokens, so
concatenated token strings reconstruct the call text exactly; each token's tag is
the char-tag of its first character (keys/names/values are separate words under
the pretokenizer, so tags never straddle a token).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from tiny_toolcall.render import T_PAD, render_example
from tiny_toolcall.tokenizer import BOS, EOS, PAD, BPETokenizer

_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " ", "…": "..."}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    return s


def normalize_example(ex: dict[str, Any]) -> dict[str, Any]:
    """Consistent ASCII-leaning normalization of query and string arg values so
    copyable spans stay copyable after tokenization."""
    ex = dict(ex)
    ex["query"] = normalize_text(ex["query"])
    calls = []
    for c in ex["answers"]:
        args = {k: normalize_text(v) if isinstance(v, str) else v for k, v in (c.get("arguments") or {}).items()}
        calls.append({"name": c["name"], "arguments": args})
    ex["answers"] = calls
    return ex


def token_char_offsets(tok: BPETokenizer, ids: list[int]) -> list[int]:
    offs = [0]
    for i in ids:
        offs.append(offs[-1] + len(tok.token_str(i)))
    return offs


def name_spans_in_prompt(tok: BPETokenizer, prompt: str, prompt_ids: list[int], names: list[str]) -> dict[str, tuple[int, int]]:
    """Token [start,end) span of each tool's name value inside the prompt (0-based
    on prompt_ids; caller shifts by BOS offset)."""
    offs = token_char_offsets(tok, prompt_ids)

    def char_to_tok(c: int) -> int:
        # first token whose span contains char c
        lo, hi = 0, len(offs) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if offs[mid + 1] <= c:
                lo = mid + 1
            else:
                hi = mid
        return lo

    spans: dict[str, tuple[int, int]] = {}
    for name in names:
        marker = f"- {name} ("
        c = prompt.find(marker)
        if c < 0:
            continue
        vs = c + 2
        ve = vs + len(name)
        spans[name] = (char_to_tok(vs), char_to_tok(ve - 1) + 1)
    return spans


def pack_examples(
    examples: list[dict[str, Any]], tok: BPETokenizer, seq_len: int = 512
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (ids [N,L] uint16, tags [N,L] uint8, decisions, kept_rows).
    kept_rows aligns 1:1 with the arrays (over-length examples are skipped).

    decisions[i] = {"names": [...], "spans": [[s,e],...], "pos": [...], "gold": [...]}
    where pos[j] is the sequence index whose *next-token* prediction starts call
    j's name — the readout position for the name head.
    """
    rows_ids, rows_tags, rows_dec, kept = [], [], [], []
    skipped = bad_chars = 0
    for ex in examples:
        ex = normalize_example(ex)
        prompt, call, char_tags = render_example(ex)
        p_ids = tok.encode(prompt)
        c_ids = tok.encode(call)
        # rows with unencodable chars break char->token alignment (<unk> is a
        # 5-char token string standing in for 1 char) — drop them
        if tok.decode(c_ids) != call or tok.decode(p_ids) != prompt:
            bad_chars += 1
            continue
        total = 1 + len(p_ids) + len(c_ids) + 1
        if total > seq_len:
            skipped += 1
            continue
        # per-token tags for the call segment
        offs = token_char_offsets(tok, c_ids)
        c_tags = [char_tags[offs[j]] for j in range(len(c_ids))]
        ids = [BOS] + p_ids + c_ids + [EOS]
        tags = [T_PAD] * (1 + len(p_ids)) + c_tags + [1]  # EOS trained as structure
        ids += [PAD] * (seq_len - len(ids))
        tags += [T_PAD] * (seq_len - len(tags))

        # name decisions
        tool_names = [t["name"] for t in ex["tools"]]
        spans0 = name_spans_in_prompt(tok, prompt, p_ids, tool_names)
        spans = {n: (s + 1, e + 1) for n, (s, e) in spans0.items()}  # shift for BOS
        dec = {"names": [], "spans": [], "pos": [], "gold": []}
        call_base = 1 + len(p_ids)
        # locate each gold call's name value in the call string
        cursor = 0
        for gcall in ex["answers"]:
            marker = f'"name":"{gcall["name"]}"'
            c = call.find(marker, cursor)
            if c < 0:
                continue
            cursor = c + len(marker)
            vs = c + len('"name":"')
            # token index of first name token within call
            lo = 0
            while offs[lo + 1] <= vs:
                lo += 1
            pos = call_base + lo - 1  # position that predicts the first name token
            cand = [n for n in tool_names if n in spans]
            if gcall["name"] not in cand:
                continue
            dec["names"].append(cand)
            dec["spans"].append([list(spans[n]) for n in cand])
            dec["pos"].append(pos)
            dec["gold"].append(cand.index(gcall["name"]))
        rows_ids.append(ids)
        rows_tags.append(tags)
        rows_dec.append(dec)
        kept.append(ex)
    if skipped or bad_chars:
        print(f"pack: skipped {skipped} over {seq_len} tokens, {bad_chars} with unencodable chars")
    return (
        np.array(rows_ids, dtype=np.uint16),
        np.array(rows_tags, dtype=np.uint8),
        rows_dec,
        kept,
    )


def save_packed(path: Path, ids: np.ndarray, tags: np.ndarray, decisions: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "ids.npy", ids)
    np.save(path / "tags.npy", tags)
    (path / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions))


def load_packed(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    ids = np.load(path / "ids.npy")
    tags = np.load(path / "tags.npy")
    decisions = [json.loads(l) for l in (path / "decisions.jsonl").read_text().splitlines()]
    return ids, tags, decisions
