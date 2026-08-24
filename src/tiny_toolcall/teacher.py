"""OpenRouter teacher synth: stepwise-validated traces, spend-capped.

Turnstile-style: generate role-by-role with deterministic validation and one
error-feedback retry; abort a trace rather than keep a bad one. Groundedness
check: string/number arg values must be evidenced in the query (kills
hallucinated args at the source). Uses Exacto routing per the plan.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
import os
import random
import re
from pathlib import Path


import httpx

from tiny_toolcall.catalog import TRAIN, schema as tool_schema
from tiny_toolcall.schema import canon_calls

API = "https://openrouter.ai/api/v1/chat/completions"
VOLUME_MODEL = "deepseek/deepseek-v4-flash-0731"

# Round 4 targets the two measured failures: the conjunction (row accuracy is
# P(names) x p^n, and Seal-Tools is 56% three-call while our corpus was 5%) and
# small-catalog refusal (every one of the ~10,700 refusal rows we had offered 3+
# tools, so `irrelevance` scored exactly 0.0 -- the model had never seen "one
# tool, it does not fit, say no").
# Matched to Seal-Tools' actual shape (28.6% one, 56.3% three, 12.7% four)
# rather than pushed as far into long chains as possible: a 25-row dry run at
# four=22/five=8 produced padded answers — calls with no arguments and no
# motivation in the query, emitted only to reach the requested count. Teaching
# that would deepen the over-calling we already have. Refusals are held high
# because half of them now carry 1-2 tool catalogs, the case with zero coverage.
CHAIN_MIX = [12, 14, 38, 16, 4, 16]  # one / two / three / four / five / refuse

# rejection-reason tally, printed by synth_teacher at the end of a batch —
# acceptance economics are part of the recipe, so failures must be attributable
REJECTS: "Counter[str]" = Counter()

NAMING = [
    "snake_case (set_lights)", "camelCase (setLights)", "PascalCase (SetLights)",
    "dot.notation (lights.set)", "camelCase with a get/search/create verb prefix",
    "SCREAMING_SNAKE for intent-style names (ACTION_SEND_EMAIL)",
]

# beyond device control: the domains our first corpus never touched
WIDE_DOMAINS = [
    "academic sociology research APIs", "film and media databases", "chemistry lab instruments",
    "genomics pipelines", "clinical trial registries", "legal case search", "patent search",
    "enterprise CRM", "HR and payroll", "supply chain logistics", "warehouse robotics",
    "freight tracking", "insurance claims", "mortgage underwriting", "stock and options trading",
    "crypto exchange", "tax filing", "invoice reconciliation", "hotel and flight booking",
    "restaurant reservations", "museum collections", "library catalogues", "weather modelling",
    "seismology sensors", "satellite imagery", "agricultural sensors", "energy grid telemetry",
    "water treatment control", "manufacturing QA", "CI/CD pipelines", "Kubernetes operations",
    "observability and alerting", "feature flags", "A/B experiment platforms", "ad campaign management",
    "email marketing", "customer support ticketing", "e-commerce order management",
    "sports statistics", "election data", "public transit schedules", "geological surveys",
]

DOMAINS = [
    "smart home lighting", "thermostats and HVAC", "door locks and security", "robot vacuum",
    "coffee machine", "washing machine", "EV charging", "garage door", "plant watering",
    "pet feeder", "aquarium control", "air purifier", "smart blinds", "sprinkler system",
    "car controls", "dashcam", "phone settings", "camera app", "flashlight and torch",
    "alarms and timers", "calendar", "reminders and todos", "contacts", "sms messaging",
    "email", "notes app", "voice memos", "music playback", "podcast player", "audiobooks",
    "tv and streaming", "smart speaker", "navigation and maps", "rideshare booking",
    "food delivery", "grocery list", "fitness tracker", "sleep tracking", "meditation app",
    "smartwatch health", "translation", "weather", "news briefing", "stock quotes",
    "banking transfers", "expense tracking", "file manager", "printer", "screen casting",
    "wifi and bluetooth settings", "drone control", "3d printer", "sous vide cooker",
]

STYLES = [
    "terse, like a command", "polite full sentence", "casual with a typo or two",
    "as a question", "impatient and short", "verbose with extra context",
    "lowercase no punctuation", "mentions a person or place by name",
]

# v6 focus modes, each aimed at a measured v5 failure bucket (diag 2026-08-19:
# 66/193 failing calls added exactly one unmentioned optional; 129/193 bound a
# wrong value). "plain" keeps a control slice of the v5 distribution.
FOCUS_OMIT = (
    "Every invented tool must have 3-5 parameters, at most one of them required. "
    "The query must state values for only a minority of the optional parameters. "
    "Answers must include ONLY the parameters whose values the query states — "
    "the point of this example is OMITTING optionals the query never mentioned. "
    "Never include an argument whose exact value the query does not state.\n"
)
FOCUS_CANON = (
    "One tool parameter must expect a DATE in ISO format (YYYY-MM-DD) and the "
    "query must state that date in natural language instead ('March 5th 2024', "
    "'the 12th of January next year' is NOT allowed - the year must be stated). "
    "The answer uses the ISO form. All OTHER argument values must still be "
    "copyable verbatim from the query.\n"
)
FOCUS_DISTRACT = (
    "The query must contain at least two different values of the SAME type (two "
    "dates, two numbers, two names, or two places) that belong in different "
    "argument slots or different calls. Binding each value to the right slot "
    "must require reading the query carefully; the values must never be equal. "
    "Copy every value character-for-character as the query writes it — dates, "
    "numbers, and names exactly as written, never reformatted.\n"
)

SYSTEM = """You generate training data for a tiny on-device function-calling model.
Produce ONE example as strict JSON. When asked to invent tools, output:
{"tools": [{"name": "...", "description": "...", "parameters": {"type": "object", "properties": {"arg": {"type": "string|integer|number|boolean"}}, "required": ["..."]}}, ...],
 "query": "...", "answers": [{"name": "...", "arguments": {...}}, ...]}
When tools are provided, output only {"query": ..., "answers": ...}.
Rules:
- tools (when inventing): output EXACTLY n_tools plausible tools for the given
  domain, named in the EXACT naming convention requested, 0-4 parameters each,
  some optional. When n_tools >= 4, include 1-2 tools NOT needed by the query as
  distractors. Optionally use an enum for one parameter. Some tools may
  legitimately take zero parameters.
- query: natural user phrasing in the requested style. Vary vocabulary. For
  multi-call templates the query must genuinely require every call — a compound
  request, a sequence, or several facts asked at once. Terse instruction phrasing
  ("Retrieve X and Y and report Z") is as valid as conversational phrasing.
- answers: the exact calls the query asks for, in order. [] if the query is
  off-topic for every tool (template=refuse) — for refuse, still invent tools
  but make the query about something none of them can do.
- Include an argument ONLY if the query gives evidence for its value. Never
  invent optional arguments. String argument values must be copyable from the
  query; numbers may appear as digits in both.
- No prose, no markdown, JSON only."""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]


def _canon_evidenced(v: str, q: str) -> bool:
    """Deterministic canonical-form evidence: an ISO date counts as evidenced
    when its year, month, and day each appear naturally in the query. Extends
    the verbatim rule without opening the door to invented values."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
    if not m:
        return False
    year, month, day = m.group(1), int(m.group(2)), int(m.group(3))
    if year not in q:
        return False
    if not (1 <= month <= 12) or _MONTHS[month - 1] not in q and f"{month}/" not in q:
        return False
    return re.search(rf"\b0?{day}(st|nd|rd|th)?\b", q) is not None


def _strip_unevidenced(ex: dict, tools_by_name: dict[str, dict]) -> None:
    """Repair-not-reject: drop OPTIONAL args whose value the query never states.

    The teacher model itself pads calls with unmentioned optionals (measured:
    73% rejection on the v6 smoke, concentrated in focus rows). Dropping the
    padded arg yields exactly the omission labeling the focus mode exists to
    teach; required args are left for _validate to reject as before.
    """
    q = (ex.get("query") or "").lower()
    for call in ex.get("answers") or []:
        tool = tools_by_name.get(call.get("name"))
        args = call.get("arguments")
        if not tool or not isinstance(args, dict):
            continue
        params = tool["parameters"]
        required = set(params.get("required", []) or [])
        for k in list(args):
            if k in required:
                continue
            v = args[k]
            if isinstance(v, str) and len(v) > 2 and v.lower() not in q \
                    and not _canon_evidenced(v, q):
                del args[k]
            elif isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and str(v) not in ex["query"] and abs(float(v)) > 1:
                del args[k]


def _validate(ex: dict, tools_by_name: dict[str, dict], template: str) -> str | None:
    """Deterministic checks. Returns an error string or None if clean."""
    if not isinstance(ex.get("query"), str) or not ex["query"].strip():
        return "missing query"
    answers = ex.get("answers")
    if not isinstance(answers, list):
        return "answers must be a list"
    if template == "refuse":
        return None if answers == [] else "refuse template must have answers=[]"
    if not answers:
        return "empty answers for non-refuse template"
    want = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(template)
    if want and len(answers) != want:
        return f"template {template} needs {want} calls, got {len(answers)}"
    q = ex["query"].lower()
    for call in answers:
        name = call.get("name")
        if name not in tools_by_name:
            return f"unknown tool {name}"
        params = tools_by_name[name]["parameters"]
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        args = call.get("arguments")
        if not isinstance(args, dict):
            return "arguments must be an object"
        missing = required - set(args)
        if missing:
            return f"{name} missing required {sorted(missing)}"
        for k, v in args.items():
            if k not in props:
                return f"{name} has no parameter {k}"
            spec = props[k] or {}
            typ = spec.get("type", "string")
            if spec.get("enum") and v not in spec["enum"]:
                return f"{k} not in enum {spec['enum']}"
            if typ == "integer" and not isinstance(v, int):
                return f"{k} must be integer"
            if typ == "number" and not isinstance(v, (int, float)):
                return f"{k} must be number"
            if typ == "boolean" and not isinstance(v, bool):
                return f"{k} must be boolean"
            if typ == "string":
                if not isinstance(v, str):
                    return f"{k} must be string"
                # groundedness: value must be evidenced in the query, verbatim
                # or via a deterministic canonical transform (ISO dates)
                if len(v) > 2 and v.lower() not in q and not _canon_evidenced(v, q):
                    return f"value {v!r} for {k} not evidenced in query"
            if typ in ("integer", "number") and str(v) not in ex["query"]:
                if not isinstance(v, bool) and abs(float(v)) > 1:
                    return f"number {v} for {k} not evidenced in query"
    return None


class SpendCap:
    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.spent = 0.0
        self.lock = asyncio.Lock()

    async def add(self, usage: dict | None) -> None:
        # DeepSeek V4 Flash worst-case list price ~$0.14/M in, $0.28/M out
        if not usage:
            return
        cost = usage.get("prompt_tokens", 0) * 0.14e-6 + usage.get("completion_tokens", 0) * 0.28e-6
        async with self.lock:
            self.spent += cost
            if self.spent > self.cap:
                raise RuntimeError(f"spend cap hit: ${self.spent:.2f} > ${self.cap}")


def _valid_invented_tools(raw) -> list[dict] | None:
    if not isinstance(raw, list) or not (2 <= len(raw) <= 9):
        return None
    out = []
    for t in raw:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            return None
        params = t.get("parameters") or {}
        props = params.get("properties")
        if props is None and not params:
            props = {}
        if not isinstance(props, dict):
            return None
        for k, spec in props.items():
            if not isinstance(spec, dict):
                return None
            if spec.get("type") not in (None, "string", "integer", "number", "boolean", "object", "array"):
                return None
        req = [r for r in (params.get("required") or []) if r in props]
        out.append({
            "name": t["name"].strip(),
            "description": str(t.get("description", "")).strip(),
            "parameters": {"type": "object", "properties": props, "required": req},
        })
    if len({t["name"] for t in out}) != len(out):
        return None
    return out


async def _one_trace(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cap: SpendCap,
    rng: random.Random,
    model: str,
    seen: set[str],
) -> dict | None:
    template = rng.choices(["one", "two", "three", "four", "five", "refuse"],
                           weights=CHAIN_MIX)[0]
    style = rng.choice(STYLES)
    focus = "plain" if template == "refuse" else \
        rng.choices(["omit", "distract", "canon", "plain"], weights=[30, 25, 25, 20])[0]
    invent = rng.random() < 0.7  # 30% stays on the device-control anchor catalog
    if focus in ("omit", "canon"):
        invent = True  # these modes need control over the invented schemas
    if invent:
        # half the invented rows come from domains far outside device control, and
        # the naming convention is sampled — the first corpus was 99.6% snake_case,
        # which is why the model failed on camelCase benchmarks
        domain = rng.choice(WIDE_DOMAINS if rng.random() < 0.6 else DOMAINS)
        naming = rng.choice(NAMING)
        # a refusal is only hard to learn when the catalog is small: with 8 tools
        # on offer "none of these fit" is nearly implied by the odds
        n_want = rng.randint(1, 2) if (template == "refuse" and rng.random() < 0.5) else rng.randint(4, 6)
        numeric = rng.random() < 0.35
        user = (
            f"template={template}\nstyle={style}\ndomain={domain}\nnaming={naming}\n"
            f"n_tools={n_want}\n"
            + ("At least two parameters across the catalog must be integer or number typed, "
               "and the query must state those values as digits.\n" if numeric else "")
            + {"omit": FOCUS_OMIT, "distract": FOCUS_DISTRACT, "canon": FOCUS_CANON}.get(focus, "")
            + "Invent tools for this domain using that naming convention, then generate one example now."
        )
        schemas: list[dict] = []
        tools_by_name: dict[str, dict] = {}
    else:
        n_tools = rng.randint(1, 2) if template == "refuse" and rng.random() < 0.5 else rng.randint(3, 8)
        tools = rng.sample(TRAIN, min(n_tools, len(TRAIN)))
        schemas = [tool_schema(t) for t in tools]
        tools_by_name = {s["name"]: s for s in schemas}
        user = (
            f"template={template}\nstyle={style}\ntools:\n{json.dumps(schemas, ensure_ascii=False)}\n"
            + (FOCUS_DISTRACT if focus == "distract" else "")
            + "Generate one example now."
        )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    err: str | None = "no attempt completed"
    for _ in range(2):
        async with sem:
            try:
                r = await client.post(
                    API,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 1.0,
                        # 0731 spends 400-800 reasoning tokens even with the flag
                        # off (measured); budget must cover reasoning + full JSON
                        "max_tokens": 3200 if invent else 1600,
                        "reasoning": {"enabled": False},  # thinking off per plan
                        "provider": {"sort": "throughput"},
                    },
                    timeout=90,
                )
                r.raise_for_status()
            except RuntimeError:
                return None
            except httpx.HTTPError:
                await asyncio.sleep(2 + rng.random() * 3)
                continue
        body = r.json()
        await cap.add(body.get("usage"))
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
        try:
            ex = _extract_json(text)
            err = None
            row_schemas = schemas
            row_by_name = tools_by_name
            if ex is None:
                err = "not valid JSON"
            elif invent:
                row_schemas = _valid_invented_tools(ex.get("tools")) or []
                row_by_name = {s["name"]: s for s in row_schemas}
                if not row_schemas:
                    err = "invalid invented tools"
            if err is None and ex is not None:
                _strip_unevidenced(ex, row_by_name)
                err = _validate(ex, row_by_name, template)
            if err is None and ex is not None:
                q_norm = " ".join(ex["query"].lower().split())
                if q_norm in seen:
                    err = "duplicate query, produce a different one"
                else:
                    seen.add(q_norm)
        except Exception as e:  # a malformed teacher reply must never kill the batch
            ex, err = None, f"validator error: {e}"
        if err is None and ex is not None:
            return {
                "query": ex["query"].strip(),
                "tools": row_schemas,
                "answers": canon_calls(ex["answers"]),
                "kind": template,
                "focus": focus,
                "split": "teacher",
            }
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Invalid: {err}. Emit corrected JSON only."})
    REJECTS[f"{focus}: {(err or 'http error')[:60]}"] += 1
    return None


def _resolve_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env = Path(__file__).resolve().parents[2] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY")
    return key


async def _run_synth(n: int, out: Path, trace_fn, concurrency: int, seed: int) -> dict:
    """Shared driver: rolling window so one straggler never stalls the batch."""
    key = _resolve_key()
    cap = SpendCap(float(os.environ.get("OPENROUTER_SPEND_CAP", "80")))
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    ok = bad = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out.exists():  # dedup against earlier runs appending to the same file
        for line in out.read_text().splitlines():
            try:
                seen.add(" ".join(json.loads(line)["query"].lower().split()))
            except (json.JSONDecodeError, KeyError):
                pass
    limits = httpx.Limits(max_connections=400, max_keepalive_connections=100)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, limits=limits) as client:
        with out.open("a") as f:
            for start in range(0, n, 5000):
                chunk = min(5000, n - start)
                tasks = [
                    asyncio.ensure_future(trace_fn(client, sem, cap, rng, seen))
                    for _ in range(chunk)
                ]
                for fut in asyncio.as_completed(tasks):
                    tr = await fut
                    if tr is None:
                        bad += 1
                    else:
                        f.write(json.dumps(tr, ensure_ascii=False) + "\n")
                        ok += 1
                    if (ok + bad) % 2000 == 0:
                        print(f"teacher: {ok} ok / {bad} rejected, ~${cap.spent:.2f} spent")
                        f.flush()
    print(f"teacher final: {ok} ok / {bad} rejected, ~${cap.spent:.2f} spent")
    for reason, cnt in REJECTS.most_common(8):
        print(f"  reject x{cnt}: {reason}")
    return {"ok": ok, "rejected": bad, "spent_usd": round(cap.spent, 2)}


async def synth_teacher(n: int, out: Path, model: str = VOLUME_MODEL, concurrency: int = 24, seed: int = 0) -> dict:
    async def fn(client, sem, cap, rng, seen):
        return await _one_trace(client, sem, cap, rng, model, seen)

    return await _run_synth(n, out, fn, concurrency, seed)


# --- adaptation: synthesis against a caller-supplied catalog ---------------

# The omission bucket was the single largest v5 failure (66 of 193 failing
# calls added exactly one unmentioned optional), so it is worth targeting on a
# user catalog too. FOCUS_OMIT cannot be reused verbatim: it dictates the shape
# of tools the teacher invents, and here the tools are fixed.
FOCUS_OMIT_GIVEN = (
    "The query must state values for only a minority of the optional parameters "
    "of the tools it uses. The answer must include ONLY the parameters whose "
    "values the query actually states — omitting the optionals the query never "
    "mentions is the entire point of this example.\n"
)


async def _one_trace_catalog(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cap: SpendCap,
    rng: random.Random,
    model: str,
    seen: set[str],
    catalog: list[dict],
    max_calls: int = 4,
) -> dict | None:
    """One trace against a fixed, caller-supplied catalog.

    Mirrors the invent=False branch of _one_trace: the teacher is handed real
    schemas and asked only for a query and its calls, so every row is validated
    against the caller's own parameter types and evidence rules.
    """
    chain = ["one", "two", "three", "four", "five", "refuse"]
    weights = list(CHAIN_MIX)
    for i, name in enumerate(chain):
        n_calls = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(name)
        if n_calls and (n_calls > max_calls or n_calls > len(catalog)):
            weights[i] = 0
    if not any(weights):
        weights[0] = 1
    template = rng.choices(chain, weights=weights)[0]
    style = rng.choice(STYLES)
    focus = "plain" if template == "refuse" else \
        rng.choices(["omit", "distract", "plain"], weights=[40, 30, 30])[0]

    # a refusal is only hard to learn when the catalog is small
    if template == "refuse" and rng.random() < 0.5:
        n_tools = rng.randint(1, min(2, len(catalog)))
    else:
        n_tools = rng.randint(min(3, len(catalog)), min(8, len(catalog)))
    tools = rng.sample(catalog, n_tools)
    schemas = [dict(t) for t in tools]
    tools_by_name = {s["name"]: s for s in schemas}
    user = (
        f"template={template}\nstyle={style}\n"
        f"tools:\n{json.dumps(schemas, ensure_ascii=False)}\n"
        + {"omit": FOCUS_OMIT_GIVEN, "distract": FOCUS_DISTRACT}.get(focus, "")
        + "Generate one example now."
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    err: str | None = "no attempt completed"
    for _ in range(2):
        async with sem:
            try:
                r = await client.post(
                    API,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 1.0,
                        "max_tokens": 1600,
                        "reasoning": {"enabled": False},
                        "provider": {"sort": "throughput"},
                    },
                    timeout=90,
                )
                r.raise_for_status()
            except RuntimeError:  # spend cap
                return None
            except httpx.HTTPError:
                await asyncio.sleep(2 + rng.random() * 3)
                continue
        body = r.json()
        await cap.add(body.get("usage"))
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            ex = _extract_json(text)
            err = None if ex is not None else "not valid JSON"
            if err is None and ex is not None:
                _strip_unevidenced(ex, tools_by_name)
                err = _validate(ex, tools_by_name, template)
            if err is None and ex is not None:
                q_norm = " ".join(ex["query"].lower().split())
                if q_norm in seen:
                    err = "duplicate query, produce a different one"
                else:
                    seen.add(q_norm)
        except Exception as e:  # a malformed reply must never kill the batch
            ex, err = None, f"validator error: {e}"
        if err is None and ex is not None:
            return {
                "query": ex["query"].strip(),
                "tools": schemas,
                "answers": canon_calls(ex["answers"]),
                "kind": template,
                "focus": focus,
                "split": "adapt",
            }
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Invalid: {err}. Emit corrected JSON only."})
    REJECTS[f"adapt/{focus}: {(err or 'http error')[:60]}"] += 1
    return None


async def synth_for_catalog(
    catalog: list[dict],
    n: int,
    out: Path,
    model: str = VOLUME_MODEL,
    concurrency: int = 24,
    seed: int = 0,
    max_calls: int = 4,
) -> dict:
    """Synthesize n validated training rows for a caller-supplied tool catalog."""
    if not catalog:
        raise SystemExit("empty catalog")

    async def fn(client, sem, cap, rng, seen):
        return await _one_trace_catalog(client, sem, cap, rng, model, seen, catalog, max_calls)

    return await _run_synth(n, out, fn, concurrency, seed)
