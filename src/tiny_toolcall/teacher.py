"""OpenRouter teacher synth: stepwise-validated traces, spend-capped.

Turnstile-style: generate role-by-role with deterministic validation and one
error-feedback retry; abort a trace rather than keep a bad one. Groundedness
check: string/number arg values must be evidenced in the query (kills
hallucinated args at the source). Uses Exacto routing per the plan.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path


import httpx

from tiny_toolcall.catalog import TRAIN, schema as tool_schema
from tiny_toolcall.schema import canon_calls

API = "https://openrouter.ai/api/v1/chat/completions"
VOLUME_MODEL = "deepseek/deepseek-v4-flash"

SYSTEM = """You generate training data for a tiny on-device function-calling model.
Given tool schemas and a template, produce ONE example as strict JSON:
{"query": "...", "answers": [{"name": "...", "arguments": {...}} , ...]}
Rules:
- query: natural, casual, sometimes terse or messy user phrasing. Vary style.
- answers: the exact calls the query asks for, in order. [] if the query is
  off-topic for every tool (template=refuse).
- Include an argument ONLY if the query gives evidence for its value. Never
  invent optional arguments. Argument values must be copyable from the query
  (numbers may be written as digits in both).
- No prose, no markdown, JSON only."""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


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
    want = {"one": 1, "two": 2, "three": 3}.get(template)
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
                # groundedness: value must be evidenced in the query
                if len(v) > 2 and v.lower() not in q:
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


async def _one_trace(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cap: SpendCap,
    rng: random.Random,
    model: str,
) -> dict | None:
    template = rng.choices(["one", "two", "three", "refuse"], weights=[45, 30, 10, 15])[0]
    n_tools = rng.randint(3, 8)
    tools = rng.sample(TRAIN, min(n_tools, len(TRAIN)))
    schemas = [tool_schema(t) for t in tools]
    tools_by_name = {s["name"]: s for s in schemas}
    user = (
        f"template={template}\ntools:\n{json.dumps(schemas, ensure_ascii=False)}\n"
        "Generate one example now."
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for _ in range(2):
        async with sem:
            try:
                r = await client.post(
                    API,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.9,
                        "max_tokens": 500,
                        "provider": {"sort": "exacto"},
                    },
                    timeout=90,
                )
                r.raise_for_status()
            except (httpx.HTTPError, RuntimeError):
                return None
        body = r.json()
        await cap.add(body.get("usage"))
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        ex = _extract_json(text)
        err = _validate(ex, tools_by_name, template) if ex else "not valid JSON"
        if err is None and ex is not None:
            return {
                "query": ex["query"].strip(),
                "tools": schemas,
                "answers": canon_calls(ex["answers"]),
                "kind": template,
                "split": "teacher",
            }
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Invalid: {err}. Emit corrected JSON only."})
    return None


async def synth_teacher(n: int, out: Path, model: str = VOLUME_MODEL, concurrency: int = 24, seed: int = 0) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env = Path(__file__).resolve().parents[2] / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY")
    cap = SpendCap(float(os.environ.get("OPENROUTER_SPEND_CAP", "80")))
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    ok = bad = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as client:
        with out.open("a") as f:
            for start in range(0, n, 200):
                batch = min(200, n - start)
                results = await asyncio.gather(
                    *(_one_trace(client, sem, cap, rng, model) for _ in range(batch))
                )
                for tr in results:
                    if tr is None:
                        bad += 1
                    else:
                        f.write(json.dumps(tr, ensure_ascii=False) + "\n")
                        ok += 1
                print(f"teacher: {ok} ok / {bad} rejected, ~${cap.spent:.2f} spent")
    return {"ok": ok, "rejected": bad, "spent_usd": round(cap.spent, 2)}
