"""Grammar-constrained decoding for the call array.

Relies on the tokenizer contract that JSON structural chars are singleton tokens,
so structure can be force-fed exactly. The model is consulted only at choice
points:

  1. empty-vs-call        after `[`: emit `]` (refuse) or start a call
  2. name                 among top-k retrieved candidates — via the name head
                          (factorized) or LM logprob over a name trie (ablation)
  3. optional-include     for each optional key in canonical (sorted) order
  4. value content        string chars until closing quote / number / bool / enum
  5. stop-vs-continue     after a call: `,` (re-retrieve, next call) or `]`

Argument keys are grammar-forced from the schema: the model never generates a
key, killing the param-name-hallucination class outright. Between array elements
the candidate name set is refreshed from retrieve(query, tools, emitted=...) —
the DTDR fix; benches have no tool execution results, so we condition on the
partial plan only.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from tiny_toolcall.retrieve import retrieve
from tiny_toolcall.tokenizer import BOS, BPETokenizer

MAX_CALLS = 4
MAX_VALUE_TOKENS = 24


class _Decoder:
    def __init__(self, model, tok: BPETokenizer, device: torch.device):
        self.model = model
        self.tok = tok
        self.device = device
        self.ids: list[int] = []
        self._hidden_cache: tuple[int, torch.Tensor, torch.Tensor] | None = None

    def _run(self) -> tuple[torch.Tensor, torch.Tensor]:
        # naive full-prefix forward; fine at this scale, no KV cache needed yet
        key = len(self.ids)
        if self._hidden_cache and self._hidden_cache[0] == key:
            return self._hidden_cache[1], self._hidden_cache[2]
        x = torch.tensor([self.ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, hidden = self.model(x)
        self._hidden_cache = (key, logits, hidden)
        return logits, hidden

    def feed_str(self, s: str) -> None:
        self.ids.extend(self.tok.encode(s))
        self._hidden_cache = None

    def feed_id(self, i: int) -> None:
        self.ids.append(i)
        self._hidden_cache = None

    def next_logits(self) -> torch.Tensor:
        logits, _ = self._run()
        return logits[0, -1]

    def hidden(self) -> torch.Tensor:
        _, hidden = self._run()
        return hidden

    def choose_first(self, options: list[str]) -> str:
        """Branch decision by first-token logit only. Exact for structural
        branches, whose options always start with distinct singleton tokens
        ( ] vs { , vs ] , vs } " vs } true vs false ) — and free of the length
        bias that summed logprobs over different-length literals would have."""
        logits = self.next_logits()
        lp = torch.log_softmax(logits.float(), dim=-1)
        first_ids = [self.tok.encode(o)[0] for o in options]
        best = max(range(len(options)), key=lambda i: lp[first_ids[i]].item())
        return options[best]

    def choose_str(self, options: list[str]) -> str:
        """Pick among literal strings by length-normalized teacher-forced logprob
        (mean per-token). Used for names and enum values, which can share
        prefixes and differ in length."""
        best, best_lp = options[0], -float("inf")
        base = list(self.ids)
        for opt in options:
            ids = self.tok.encode(opt)
            lp = 0.0
            self.ids = list(base)
            self._hidden_cache = None
            for i in ids:
                logits = self.next_logits()
                lp += torch.log_softmax(logits.float(), dim=-1)[i].item()
                self.feed_id(i)
            mean_lp = lp / max(1, len(ids))
            if mean_lp > best_lp:
                best, best_lp = opt, mean_lp
        self.ids = base
        self._hidden_cache = None
        return best

    def gen_string_value(self, closing: str = '"') -> str:
        """Free-generate a string value; structural singletons mean the closing
        quote is always its own token, so we stop exactly there."""
        close_id = self.tok.vocab[closing]
        out: list[str] = []
        for _ in range(MAX_VALUE_TOKENS):
            logits = self.next_logits()
            # ban structural tokens other than the closer inside a string value
            masked = logits.clone()
            for ch in "{}[],:":
                tid = self.tok.vocab.get(ch)
                if tid is not None:
                    masked[tid] = -float("inf")
            nxt = int(masked.argmax().item())
            if nxt == close_id:
                break
            self.feed_id(nxt)
            out.append(self.tok.token_str(nxt))
        return "".join(out)

    def gen_number_value(self) -> str:
        """Numbers are single words under our pretokenizer ("-" and "." are not
        structural, digits merge). Generate one token, validate, fall back to 0."""
        logits = self.next_logits()
        order = torch.argsort(logits, descending=True)
        for cand in order[:64].tolist():
            s = self.tok.token_str(cand)
            try:
                json.loads(s)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(json.loads(s), (int, float)):
                self.feed_id(cand)
                return s
        self.feed_str("0")
        return "0"


def _sorted_props(tool: dict[str, Any]) -> tuple[list[str], set[str], dict[str, Any]]:
    params = tool.get("parameters", {})
    props = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])
    return sorted(props), required, props


def constrained_decode(
    model,
    tok: BPETokenizer,
    prompt: str,
    query: str,
    tools: list[dict[str, Any]],
    device: torch.device,
    k: int = 5,
    use_name_head: bool = True,
    name_spans: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Decode a canonical call array under the grammar. Returns parsed calls.

    name_spans: token [start,end) spans of each tool's name inside the prompt,
    needed for the name head; if absent (or use_name_head=False) the LM picks
    names by teacher-forced logprob (the heads-off ablation).
    """
    dec = _Decoder(model, tok, device)
    dec.feed_id(BOS)  # sequences start with BOS in training; match it here
    dec.feed_str(prompt)
    dec.feed_str("[")

    emitted: list[dict[str, Any]] = []
    for _ in range(MAX_CALLS):
        # choice 1 / 5: stop ( ] ) or (another) call ( { first, , later )
        open_opt = '{"name":"' if not emitted else ',{"name":"'
        choice = dec.choose_first(["]", open_opt])
        if choice == "]":
            dec.feed_str("]")
            break
        dec.feed_str(open_opt)

        # choice 2: name among refreshed candidates
        cands = retrieve(query, tools, k=k, emitted=emitted)
        cand_names = [t["name"] for t in cands]
        if use_name_head and name_spans:
            spans = [name_spans[n] for n in cand_names if n in name_spans]
            names_with_spans = [n for n in cand_names if n in name_spans]
            if spans:
                hidden = dec.hidden()
                scores = model.name_scores(hidden, len(dec.ids) - 1, spans)
                name = names_with_spans[int(scores.argmax().item())]
            else:
                name = dec.choose_str(cand_names)
        else:
            name = dec.choose_str(cand_names)
        dec.feed_str(name)
        tool = next(t for t in tools if t["name"] == name)
        dec.feed_str('","arguments":{')

        # choices 3/4: keys forced in sorted order; optionals are include/skip
        keys, required, props = _sorted_props(tool)
        args: dict[str, Any] = {}
        first = True
        for key in keys:
            spec = props.get(key, {})
            sep = "" if first else ","
            if key not in required:
                # include this optional? first chars differ: , or " vs }
                opener = f'{sep}"{key}":'
                pick = dec.choose_first([opener, "}"])
                if pick == "}":
                    continue
                dec.feed_str(opener)
            else:
                dec.feed_str(f'{sep}"{key}":')
            first = False
            typ = spec.get("type", "string")
            enum = spec.get("enum")
            if enum:
                dec.feed_str('"')
                val = dec.choose_str([str(e) for e in enum])
                dec.feed_str(val + '"')
                args[key] = val
            elif typ == "boolean":
                val = dec.choose_first(["true", "false"])
                dec.feed_str(val)
                args[key] = val == "true"
            elif typ in ("integer", "number"):
                s = dec.gen_number_value()
                args[key] = json.loads(s)
                if typ == "integer" and isinstance(args[key], float):
                    args[key] = int(args[key])
            else:
                dec.feed_str('"')
                val = dec.gen_string_value()
                dec.feed_str('"')
                args[key] = val
        dec.feed_str("}}")
        emitted.append({"name": name, "arguments": args})
    else:
        dec.feed_str("]")
    return emitted
