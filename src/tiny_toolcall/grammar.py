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
MAX_VALUE_TOKENS = 96  # email bodies and addresses run long; truncation was costing exact-match


class _Decoder:
    """Incremental KV-cached decode with exact snapshot/rollback (cache tensors
    are truncated back to a saved length, so candidate rollouts are free of
    recomputation and leave no trace)."""

    def __init__(self, model, tok: BPETokenizer, device: torch.device):
        self.model = model
        self.tok = tok
        self.device = device
        self.ids: list[int] = []
        self.caches = model.new_caches()
        self.hiddens: list[torch.Tensor] = []  # (1, t_i, d) chunks, order = feed order
        self.last_logits: torch.Tensor | None = None

    def _feed_ids(self, ids: list[int]) -> None:
        """Feed tokens, updating the KV cache. The vocabulary projection is done
        for the LAST position only — the decoder never reads logits for tokens it
        already force-fed, so projecting a whole chunk wastes |vocab| work per
        structural token (Needle's engine makes the same skip)."""
        if not ids:
            return
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, hidden = self.model(x, caches=self.caches, need_logits=False)
            last = torch.nn.functional.linear(hidden[0, -1], self.model.embed.weight)
        self.ids.extend(ids)
        self.hiddens.append(hidden)
        self.last_logits = last

    def feed_str(self, s: str) -> None:
        self._feed_ids(self.tok.encode(s))

    def feed_id(self, i: int) -> None:
        self._feed_ids([i])

    def next_logits(self) -> torch.Tensor:
        assert self.last_logits is not None
        return self.last_logits

    def hidden(self) -> torch.Tensor:
        return torch.cat(self.hiddens, dim=1)

    def snapshot(self) -> tuple[int, int, torch.Tensor | None]:
        return len(self.ids), len(self.hiddens), self.last_logits

    def rollback(self, snap: tuple[int, int, torch.Tensor | None]) -> None:
        n_ids, n_hid, logits = snap
        del self.ids[n_ids:]
        del self.hiddens[n_hid:]
        self.last_logits = logits
        for c in self.caches:
            if "k" in c and c["k"].shape[2] > n_ids:
                c["k"] = c["k"][:, :, :n_ids]
                c["v"] = c["v"][:, :, :n_ids]

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

    def score_str(self, options: list[str]) -> list[float]:
        """Length-normalized teacher-forced logprob (mean per-token) for each
        literal. Cache is rolled back after each rollout."""
        snap = self.snapshot()
        scores = []
        for opt in options:
            ids = self.tok.encode(opt)
            lp = 0.0
            for i in ids:
                logits = self.next_logits()
                lp += torch.log_softmax(logits.float(), dim=-1)[i].item()
                self.feed_id(i)
            self.rollback(snap)
            scores.append(lp / max(1, len(ids)))
        return scores

    def choose_str(self, options: list[str]) -> str:
        scores = self.score_str(options)
        return options[max(range(len(options)), key=lambda i: scores[i])]

    def gen_string_value(self, closing: str = '"') -> str:
        """Free-generate a string value until the closing quote token. String
        values legitimately contain structural chars (datetimes have ':'), so
        only specials are banned — the quote singleton is the exact terminator."""
        close_id = self.tok.vocab[closing]
        out: list[str] = []
        for _ in range(MAX_VALUE_TOKENS):
            logits = self.next_logits()
            masked = logits.clone()
            masked[:4] = -float("inf")  # pad/bos/eos/unk
            for sp in ("<tools>", "</tools>", "<query>", "</query>", "<call>", "</call>"):
                tid = self.tok.vocab.get(sp)
                if tid is not None:
                    masked[tid] = -float("inf")
            nxt = int(masked.argmax().item())
            if nxt == close_id:
                break
            self.feed_id(nxt)
            out.append(self.tok.token_str(nxt))
        return "".join(out)

    def gen_templated(self, template: str) -> str:
        """Fill a fixed-shape value: '#' positions are model-chosen digits, all
        other chars are force-fed. Used for datetime args, whose format is a
        constant the model should never have to spell (only the digits vary)."""
        out = []
        for ch in template:
            if ch != "#":
                self.feed_str(ch)
                out.append(ch)
                continue
            logits = self.next_logits()
            masked = torch.full_like(logits, -float("inf"))
            for d in "0123456789":
                tid = self.tok.vocab.get(d)
                if tid is not None:
                    masked[tid] = logits[tid]
            nxt = int(masked.argmax().item())
            self.feed_id(nxt)
            out.append(self.tok.token_str(nxt))
        return "".join(out)

    _NUM_CHARS = set("0123456789.-")

    def gen_number_value(self) -> str:
        """Generate number tokens until the next token would leave [0-9.-];
        numbers may span several BPE tokens (e.g. '1200' + '.0')."""
        out = ""
        for _ in range(6):
            logits = self.next_logits()
            masked = logits.clone()
            order = torch.argsort(masked, descending=True)
            nxt = None
            for cand in order[:96].tolist():
                s = self.tok.token_str(cand)
                if s and all(c in self._NUM_CHARS for c in s):
                    candidate = out + s
                    try:
                        float(candidate)
                    except ValueError:
                        continue
                    nxt = (cand, s)
                    break
            if nxt is None:
                break
            # stop if the model prefers a structural continuation over more digits
            struct_ids = [self.tok.vocab.get(c) for c in ',}']
            best_struct = max((logits[i].item() for i in struct_ids if i is not None), default=-float("inf"))
            if out and best_struct > logits[nxt[0]].item():
                break
            self.feed_id(nxt[0])
            out += nxt[1]
        if not out:
            self.feed_str("0")
            return "0"
        return out


def _value_template(key: str, spec: dict[str, Any]) -> str | None:
    """Reserved for values whose format is a verifiable constant.

    Deliberately disabled: Mobile Actions' own schema advertises
    `YYYY-MM-DDTHH:MM:SS` while its gold values are `2025-05-13 13:30:00`
    (space, not `T`), so schema text is NOT a trustworthy source for the shape,
    and our synth uses a third format. Forcing a template here would corrupt
    every datetime row on one dataset or the other. The convention is learned
    from data instead (MA-train is in the mix). gen_templated stays available
    for cases where the format is confirmed by gold, not by prose.
    """
    return None


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
    k: int = 0,
    use_name_head: bool = True,
    name_spans: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Decode a canonical call array under the grammar. Returns parsed calls.

    name_spans: token [start,end) spans of each tool's name inside the prompt,
    needed for the name head; if absent (or use_name_head=False) the LM picks
    names by teacher-forced logprob (the heads-off ablation).
    """
    # retrieval narrows only genuinely large catalogs; small ones stay whole so
    # retrieval recall never caps name accuracy below what the head can do
    if k <= 0:
        k = len(tools) if len(tools) <= 8 else 8
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

        # choice 2: name among refreshed candidates. heads-on ensembles the
        # trained readout (wins off-distribution) with the LM's teacher-forced
        # logprob (wins in-distribution); heads-off is the pure-LM ablation
        cands = retrieve(query, tools, k=k, emitted=emitted)
        cand_names = [t["name"] for t in cands]
        if use_name_head and name_spans and all(n in name_spans for n in cand_names):
            hidden = dec.hidden()
            head = model.name_scores(hidden, len(dec.ids) - 1, [name_spans[n] for n in cand_names])
            head_p = torch.softmax(head.float(), dim=-1)
            lm = torch.tensor(dec.score_str(cand_names), device=head_p.device)
            lm_p = torch.softmax(lm, dim=-1)
            name = cand_names[int((head_p + lm_p).argmax().item())]
        else:
            name = dec.choose_str(cand_names)
        dec.feed_str(name)
        tool = next(t for t in tools if t["name"] == name)
        dec.feed_str('","arguments":{')

        # choices 3/4: keys forced in sorted order; optionals are include/skip
        keys, required, props = _sorted_props(tool)
        args: dict[str, Any] = {}
        first = True
        for ki, key in enumerate(keys):
            spec = props.get(key, {})
            sep = "" if first else ","
            if key not in required:
                # include this optional? the skip-path continuation is the NEXT
                # key's opener (which also starts with `"`) or `}` — so the
                # decision must be scored on content, not the first token
                opener = f'{sep}"{key}":'
                nxt_key = keys[ki + 1] if ki + 1 < len(keys) else None
                skip = f'{sep}"{nxt_key}":' if nxt_key else "}"
                pick = dec.choose_str([opener, skip])
                if pick != opener:
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
                try:
                    val = json.loads(s)
                    if not isinstance(val, (int, float)):
                        raise ValueError(s)
                except (json.JSONDecodeError, ValueError):
                    # generator can assemble char-valid but JSON-invalid strings
                    # (e.g. "1-2", "3.4.5"); keep the longest valid numeric prefix
                    val = 0
                    for cut in range(len(s), 0, -1):
                        try:
                            cand = json.loads(s[:cut])
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if isinstance(cand, (int, float)):
                            val = cand
                            break
                args[key] = int(val) if typ == "integer" and isinstance(val, float) else val
            else:
                dec.feed_str('"')
                shape = _value_template(key, spec)
                val = dec.gen_templated(shape) if shape else dec.gen_string_value()
                dec.feed_str('"')
                args[key] = val
        dec.feed_str("}}")
        emitted.append({"name": name, "arguments": args})
    else:
        dec.feed_str("]")
    return emitted
