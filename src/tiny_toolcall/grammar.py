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

from tiny_toolcall.retrieve import lexical_scores, retrieve
from tiny_toolcall.tokenizer import BOS, BPETokenizer

MAX_CALLS = 4
# The lexical prior is weighted by how *discriminative* it is on this catalog,
# not by a fixed constant. Self-describing catalogs (getPostmodernTheory answering
# "postmodern theory") produce a sharply peaked prior and deserve weight; device
# verbs (set_lights for "dim the kitchen") produce a flat one and do not. A hard
# confidence gate cannot express this: the model is often confidently wrong on
# an unfamiliar catalog, so the gate never opens.
LEX_MAX_WEIGHT = 0.85   # ceiling on the prior's share when it is maximally peaked
LEX_SHARPNESS = 4.0     # how fast peakedness converts to weight
REFUSE_GATE = 0.35
REPEAT_BLOCK = True
PTR_GATE = 0.55   # min softmax mass on the chosen start AND end before copying
MAX_VALUE_TOKENS = 96  # email bodies and addresses run long; truncation was costing exact-match


class _Decoder:
    """Incremental KV-cached decode with exact snapshot/rollback (cache tensors
    are truncated back to a saved length, so candidate rollouts are free of
    recomputation and leave no trace)."""

    def __init__(self, model, tok: BPETokenizer, device: torch.device, temp: float = 0.0):
        self.model = model
        self.tok = tok
        self.device = device
        # temp > 0 turns every choice point into a sample instead of an argmax.
        # Only pass@k measurement uses this; scored runs stay deterministic.
        self.temp = temp
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

    def pick(self, scores: torch.Tensor) -> int:
        """argmax, or a temperature sample when measuring pass@k."""
        if self.temp <= 0:
            return int(scores.argmax().item())
        p = torch.softmax(scores.float() / self.temp, dim=-1)
        return int(torch.multinomial(p, 1).item())

    def choose_first(self, options: list[str]) -> str:
        """Branch decision by first-token logit only. Exact for structural
        branches, whose options always start with distinct singleton tokens
        ( ] vs { , vs ] , vs } " vs } true vs false ) — and free of the length
        bias that summed logprobs over different-length literals would have."""
        logits = self.next_logits()
        lp = torch.log_softmax(logits.float(), dim=-1)
        first_ids = [self.tok.encode(o)[0] for o in options]
        return options[self.pick(torch.tensor([lp[i] for i in first_ids]))]

    def score_first(self, options: list[str]) -> list[float]:
        """Per-option first-token logprob (branch decisions)."""
        lp = torch.log_softmax(self.next_logits().float(), dim=-1)
        return [lp[self.tok.encode(o)[0]].item() for o in options]

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
        return options[self.pick(torch.tensor(self.score_str(options)))]

    def pointer_copy(self, prompt_ids: list[int]) -> str | None:
        """Ask the trained pointer head for a span of the prompt to copy.

        DISABLED BY DEFAULT — measured 16 points worse on Seal-Tools
        (25.3 -> 9.3 exact, single-call 56.7 -> 23.3) with name accuracy
        unchanged, so the entire loss is in the arguments it was built to fix.

        This is the SECOND copy mechanism to fail here, after word-span copying
        (-30 points). The head does learn: roughly half of training samples hit
        both endpoints exactly. But its confidence does not correlate with its
        correctness, so a 0.55 two-endpoint gate still admits wrong spans, and a
        wrong span is a guaranteed row failure where free generation would often
        have produced the right value.

        The honest reading is that "copy, don't generate" — well supported in the
        literature — does not transfer to this setup, because grammar-constrained
        free generation is already a strong copier (~47% per-call argument
        accuracy) and both copy mechanisms we built are noisier than that floor.
        A better-calibrated head (batched training, confidence calibration on a
        held-out split, span-level rather than endpoint-level scoring) might beat
        it; ours does not.
        """
        if not hasattr(self.model, "ptr_start"):
            return None
        hidden = self.hidden()
        plen = len(prompt_ids)
        with torch.no_grad():
            s_log, e_log = self.model.pointer_scores(hidden, len(self.ids) - 1, plen)
        s_p, e_p = torch.softmax(s_log.float(), -1), torch.softmax(e_log.float(), -1)
        si, ei = int(s_p.argmax()), int(e_p.argmax())
        if ei < si or ei - si > 40:
            return None
        if float(s_p[si]) < PTR_GATE or float(e_p[ei]) < PTR_GATE:
            return None
        return "".join(self.tok.token_str(t) for t in prompt_ids[si : ei + 1])

    def copy_span_value(self, query: str, max_words: int = 8, hint: str = "",
                        top_k: int = 24) -> str | None:
        """Score candidate spans of the query and copy the best one verbatim.

        DISABLED BY DEFAULT — this measured 30 points WORSE than free generation
        on Seal-Tools single-call (10.0 vs 40.0 exact match, name accuracy
        unchanged), so the loss is entirely in the arguments.

        The hypothesis came from three sources (Seal-Tools' own error analysis:
        70% of parameter errors are keyword-extraction failures; LocalAgent, a
        28M tool-caller: "arg values must be copied, not generated"; FuncBenchGen
        on stale value propagation) plus our own measurement that 90% of gold
        values appear "verbatim" in the query.

        That measurement was wrong in a specific way: it tested SUBSTRING
        containment, while this function enumerates WORD-LEVEL spans. Gold
        'manta ray' is a substring of the query's "manta rays" but equals no word
        span of it. Morphology, punctuation and partial-word boundaries break
        word-span copying on a large fraction of values, and no amount of
        candidate scoring recovers a span that was never generated. Free
        generation handles these because it emits tokens, not spans.

        Kept in the tree because a character-level or learned pointer head could
        still be the right mechanism; the word-span approximation is not.
        """
        words = query.split()
        if not words or len(words) > 120:
            return None
        cands: list[str] = []
        seen = set()
        for i in range(len(words)):
            for n in range(1, min(max_words, len(words) - i) + 1):
                sp = " ".join(words[i : i + n]).strip(".,;:!?\"'()[]")
                if sp and sp not in seen:
                    seen.add(sp)
                    cands.append(sp)
        if not cands:
            return None
        # Model-scoring every span costs a rollout each and dominates decode
        # time. Pre-rank cheaply by relevance to the parameter (its name and
        # description) plus a mild shortness prior, then score only the top-k.
        if len(cands) > top_k:
            from tiny_toolcall.retrieve import _tok

            hint_w = _tok(hint) if hint else set()
            def prior(sp: str) -> float:
                w = _tok(sp)
                return len(w & hint_w) * 2.0 - 0.15 * len(sp.split())
            cands = sorted(cands, key=prior, reverse=True)[:top_k]
        scores = self.score_str(cands)
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        if scores[best] - scores[second] < 1e-9 and best != second:
            return None
        return cands[best]

    def gen_string_value(self, closing: str = '"') -> str:
        """Free-generate a string value until the closing quote token. String
        values legitimately contain structural chars (datetimes have ':'), so
        only specials are banned — the quote singleton is the exact terminator."""
        close_id = self.tok.vocab[closing]
        out: list[str] = []
        ids_out: list[int] = []
        for _ in range(MAX_VALUE_TOKENS):
            logits = self.next_logits()
            masked = logits.clone()
            masked[:4] = -float("inf")  # pad/bos/eos/unk
            for sp in ("<tools>", "</tools>", "<query>", "</query>", "<call>", "</call>"):
                tid = self.tok.vocab.get(sp)
                if tid is not None:
                    masked[tid] = -float("inf")
            # greedy decoding of a copied span degenerates into loops
            # ("the city in the city in..."), which fails exact match outright.
            # Ban any token that would complete a repeated trigram, and stop on a
            # repeated bigram — a copied value is never periodic.
            nxt = self.pick(masked)
            if nxt == close_id:
                break
            # Only PERIODIC repetition is degenerate. Natural prose repeats
            # bigrams constantly ("in the", "to the"), so banning repeated
            # trigrams mangles long values — that mistake cost 51 points on
            # Mobile Actions. A loop instead shows the last k tokens exactly
            # equalling the k before them.
            if REPEAT_BLOCK:
                cand = ids_out + [nxt]
                looped = False
                for k in (2, 3, 4):
                    if len(cand) >= 3 * k and cand[-k:] == cand[-2 * k:-k] == cand[-3 * k:-2 * k]:
                        looped = True
                        break
                if looped:
                    break
            self.feed_id(nxt)
            ids_out.append(nxt)
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


# Inference only: no caller backprops through the decode loop (MRT computes its
# gradients from seq_logprob instead), so building an autograd graph here is pure
# waste — and it made every decode warn about tensor->scalar conversion.
@torch.no_grad()
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
    gated: bool = True,
    copy_spans: bool = False,  # measured WORSE — see note on copy_span_value
    use_pointer: bool = False,  # measured 16 points WORSE — see pointer_copy
    force_names: list[str] | None = None,  # DIAGNOSTIC ONLY — never on a scored run
    max_calls: int = MAX_CALLS,
    temp: float = 0.0,  # >0 samples every choice point (pass@k only)
) -> list[dict[str, Any]]:
    """Decode a canonical call array under the grammar. Returns parsed calls.

    name_spans: token [start,end) spans of each tool's name inside the prompt,
    needed for the name head; if absent (or use_name_head=False) the LM picks
    names by teacher-forced logprob (the heads-off ablation).

    force_names pins the call sequence to a given list, bypassing both the
    stop decision and the name choice. It exists so error analysis can attribute
    a failure to naming/ordering versus argument filling; it must never be used
    on a run whose number gets reported.
    """
    # An empty catalog has exactly one correct answer and no decision to make.
    # BFCL live_irrelevance carries 4 such rows and they crashed the name choice
    # (argmax over zero candidates) rather than refusing, which is what an empty
    # tool list means.
    if not tools:
        return []
    # retrieval narrows only genuinely large catalogs; small ones stay whole so
    # retrieval recall never caps name accuracy below what the head can do
    if k <= 0:
        k = len(tools) if len(tools) <= 8 else 8
    dec = _Decoder(model, tok, device, temp=temp)
    prompt_ids = tok.encode(prompt)
    dec.feed_id(BOS)  # sequences start with BOS in training; match it here
    dec.feed_str(prompt)
    dec.feed_str("[")

    emitted: list[dict[str, Any]] = []
    budget = len(force_names) if force_names is not None else max_calls
    for _ in range(budget):
        # choice 1 / 5: stop ( ] ) or (another) call ( { first, , later )
        open_opt = '{"name":"' if not emitted else ',{"name":"'
        if force_names is not None:
            refuse = False
        else:
            stop_lp = dec.score_first(["]", open_opt])
            refuse = dec.pick(torch.tensor(stop_lp)) == 0
        if refuse and gated and not emitted:
            # a confident refusal stands; a marginal one loses to clear evidence
            # that some tool in the catalog answers the query
            # `peak` measures how far the prior is from uniform, i.e. how well it
            # DISCRIMINATES between candidates. That quantity is undefined with a
            # single candidate: lexical_scores normalises to sum 1.0, so one tool
            # always scores 1.0 and always reads as maximally peaked — even at
            # zero token overlap. The override therefore fired unconditionally on
            # one-tool catalogs and the model could never refuse there, which is
            # why BFCL irrelevance (mean 1.0 tools) scored exactly 0.0.
            lex = lexical_scores(query, tools) if len(tools) > 1 else {}
            if lex:
                peak = max(lex.values()) - 1.0 / max(2, len(lex))
                margin = abs(stop_lp[0] - stop_lp[1])
                # strong evidence overrides a refusal outright; weak evidence only
                # overrides a marginal one
                if peak > 0.25 or (margin < REFUSE_GATE and peak > 0.08):
                    refuse = False
        if refuse:
            dec.feed_str("]")
            break
        dec.feed_str(open_opt)

        # choice 2: name among refreshed candidates. heads-on ensembles the
        # trained readout (wins off-distribution) with the LM's teacher-forced
        # logprob (wins in-distribution); heads-off is the pure-LM ablation
        if force_names is not None:
            name = force_names[len(emitted)]
            dec.feed_str(name)
            tool = next(t for t in tools if t["name"] == name)
            dec.feed_str('","arguments":{')
            emitted.append(_fill_args(dec, tool, prompt_ids, query, copy_spans, use_pointer, name))
            continue
        cands = retrieve(query, tools, k=k, emitted=emitted)
        cand_names = [t["name"] for t in cands]
        if use_name_head and name_spans and all(n in name_spans for n in cand_names):
            hidden = dec.hidden()
            head = model.name_scores(hidden, len(dec.ids) - 1, [name_spans[n] for n in cand_names])
            head_p = torch.softmax(head.float(), dim=-1)
            lm = torch.tensor(dec.score_str(cand_names), device=head_p.device)
            lm_p = torch.softmax(lm, dim=-1)
            probs = (head_p + lm_p) / 2
        else:
            lm = torch.tensor(dec.score_str(cand_names))
            probs = torch.softmax(lm, dim=-1)
        if gated and len(cand_names) > 1:
            lex = lexical_scores(query, cands, emitted=emitted)
            lp = torch.tensor([lex.get(n, 0.0) for n in cand_names], device=probs.device)
            if float(lp.sum()) > 0:
                lp = lp / lp.sum()
                # peakedness: how far the prior is from uniform (0 = no signal)
                uniform = 1.0 / len(cand_names)
                peak = float(lp.max()) - uniform
                # two independent conditions must both hold before the prior gets
                # weight: it must discriminate, AND the model must be unsure.
                # Sharpness alone cost 2.7 points on catalogs the model knows well.
                srt = torch.sort(probs, descending=True).values
                confidence = float(srt[0] - srt[1])
                w = min(LEX_MAX_WEIGHT, max(0.0, peak * LEX_SHARPNESS)) * (1.0 - confidence)
                probs = (1.0 - w) * probs + w * lp
        name = cand_names[dec.pick(torch.log(probs.clamp_min(1e-9)))]
        dec.feed_str(name)
        tool = next(t for t in tools if t["name"] == name)
        dec.feed_str('","arguments":{')

        emitted.append(_fill_args(dec, tool, prompt_ids, query, copy_spans, use_pointer, name))
    else:
        dec.feed_str("]")
    return emitted


def _fill_args(
    dec: "_Decoder",
    tool: dict[str, Any],
    prompt_ids: list[int],
    query: str,
    copy_spans: bool,
    use_pointer: bool,
    name: str,
) -> dict[str, Any]:
    """Choices 3/4: keys forced in sorted order; optionals are include/skip.

    The decoder is positioned just after `"arguments":{` and is left just after
    the closing `}}` of the call.
    """
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
            ptr = dec.pointer_copy(prompt_ids) if (use_pointer and prompt_ids) else None
            if shape:
                val = dec.gen_templated(shape)
            elif ptr is not None:
                dec.feed_str(ptr)
                val = ptr
            elif copy_spans:
                span = dec.copy_span_value(
                    query, hint=f"{key} {spec.get('description','')}")
                if span is not None:
                    dec.feed_str(span)
                    val = span
                else:
                    val = dec.gen_string_value()
            else:
                val = dec.gen_string_value()
            dec.feed_str('"')
            args[key] = val
    dec.feed_str("}}")
    return {"name": name, "arguments": args}
