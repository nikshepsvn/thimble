"""Local validated synth. No API. Mix targets Needle's holes: 2-call, refuse, name-mask."""

from __future__ import annotations

import random
from typing import Any

from tiny_toolcall.catalog import (
    BRIGHT, CITIES, HANDLES, NAMES, OOD, ROOMS, TEMPS, TRAIN, Tool, by_name, schema,
)
from tiny_toolcall.schema import canon_calls

PREFIX = ["", "please ", "can you ", "hey ", "i need you to ", ""]
ANDS = [" and ", " then ", ", also ", " plus "]
OFFTOPIC = [
    "what's the capital of France",
    "write me a haiku about rain",
    "who won the world cup in 2018",
    "explain quantum entanglement",
    "what should I cook tonight",
    "is a hot dog a sandwich",
    "tell me a joke",
    "how do I learn piano",
]


def _pick(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _lights(rng: random.Random) -> tuple[str, dict]:
    # args must be licensed by the chosen stem: brightness only when a number
    # appears in the query, on only when on/off is stated or implied by 0/percent
    room, b = _pick(rng, ROOMS), _pick(rng, BRIGHT)
    on = b > 0
    variants: list[tuple[str, dict[str, Any]]] = [
        (f"dim the {room} to {b}", {"room": room, "on": on, "brightness": b}),
        (f"set {room} lights to {b}", {"room": room, "on": on, "brightness": b}),
        (f"lights in the {room} at {b} percent", {"room": room, "on": on, "brightness": b}),
        (f"turn the {room} lights {'on' if on else 'off'}", {"room": room, "on": on}),
    ]
    q, args = _pick(rng, variants)
    return q, {"name": "set_lights", "arguments": args}


def _therm(rng: random.Random) -> tuple[str, dict]:
    # mode only when the stem states it
    t = _pick(rng, TEMPS)
    variants: list[tuple[str, dict[str, Any]]] = [
        (f"set the thermostat to {t}", {"temperature": t}),
        (f"make it {t} degrees", {"temperature": t}),
        (f"cool the house to {t}", {"temperature": t, "mode": "cool"}),
        (f"heat the place to {t}", {"temperature": t, "mode": "heat"}),
    ]
    q, args = _pick(rng, variants)
    return q, {"name": "set_thermostat", "arguments": args}


def _weather(rng: random.Random) -> tuple[str, dict]:
    city = _pick(rng, CITIES)
    stems = [f"what's the weather in {city}", f"how's {city} looking", f"forecast for {city}"]
    return _pick(rng, stems), {"name": "get_weather", "arguments": {"city": city}}


def _timer(rng: random.Random) -> tuple[str, dict]:
    m = _pick(rng, [1, 3, 5, 10, 15, 20, 30])
    stems = [f"set a timer for {m} minutes", f"{m} minute timer", f"remind me in {m} minutes"]
    return _pick(rng, stems), {"name": "set_timer", "arguments": {"minutes": m}}


def _msg(rng: random.Random) -> tuple[str, dict]:
    to, body = _pick(rng, HANDLES), _pick(rng, ["on my way", "running late", "got it", "call me"])
    return f"text {to} {body}", {"name": "send_message", "arguments": {"to": to, "body": body}}


def _cal(rng: random.Random) -> tuple[str, dict]:
    title = _pick(rng, ["standup", "dentist", "flight", "lunch with Maya"])
    dt = _pick(rng, ["2026-09-01T09:00", "2026-09-02T14:30", "2026-10-12T18:00"])
    return f"add {title} on {dt}", {"name": "create_calendar_event", "arguments": {"title": title, "datetime": dt}}


def _music(rng: random.Random) -> tuple[str, dict]:
    q = _pick(rng, ["lofi", "coltrane", "rain sounds", "80s pop"])
    return f"play {q}", {"name": "play_music", "arguments": {"query": q}}


def _lock(rng: random.Random) -> tuple[str, dict]:
    door = _pick(rng, ["front", "back", "garage"])
    locked = bool(rng.randrange(2))
    verb = "lock" if locked else "unlock"
    return f"{verb} the {door} door", {"name": "lock_door", "arguments": {"door": door, "locked": locked}}


def _flash(rng: random.Random) -> tuple[str, dict]:
    on = bool(rng.randrange(2))
    name = "turn_on_flashlight" if on else "turn_off_flashlight"
    return ("flashlight on" if on else "kill the flashlight"), {"name": name, "arguments": {}}


def _email(rng: random.Random) -> tuple[str, dict]:
    to = _pick(rng, ["maya@ex.com", "omar@ex.com"])
    sub = _pick(rng, ["invoice", "hello", "reschedule"])
    return f"email {to} subject {sub}", {"name": "send_email", "arguments": {"to": to, "subject": sub}}


def _map(rng: random.Random) -> tuple[str, dict]:
    p = _pick(rng, CITIES + ["central park", "gate 4"])
    return f"show {p} on the map", {"name": "show_map", "arguments": {"place": p}}


def _invoice(rng: random.Random) -> tuple[str, dict]:
    v, tot, due = _pick(rng, ["Acme Corp", "GreenMart", "Nimbus"]), _pick(rng, [12.5, 1200.0, 48.0]), "2026-09-01"
    text = f"Invoice from {v}, ${tot:.2f}, due {due}"
    return text, {"name": "extract_invoice", "arguments": {"vendor": v, "total": tot, "due_date": due}}


def _robot(rng: random.Random) -> tuple[str, dict]:
    m, h = _pick(rng, [0.5, 1.0, 3.0]), _pick(rng, ["forward", "back"])
    return f"walk {h} {m} meters", {"name": "robot_move", "arguments": {"meters": m, "heading": h}}


MAKERS = [_lights, _therm, _weather, _timer, _msg, _cal, _music, _lock, _flash, _email, _map, _invoice, _robot]


def _ood_lamp(rng: random.Random) -> tuple[str, dict]:
    room, b = _pick(rng, ROOMS), _pick(rng, BRIGHT)
    return f"put the {room} lamp at {b}", {"name": "adjust_lamp", "arguments": {"room": room, "brightness": b, "on": b > 0}}


def _ood_climate(rng: random.Random) -> tuple[str, dict]:
    t = _pick(rng, TEMPS)
    return f"climate to {t} degrees", {"name": "set_climate", "arguments": {"temperature": t}}


def _ood_agenda(rng: random.Random) -> tuple[str, dict]:
    title = _pick(rng, ["retro", "gym", "call with Ana"])
    dt = _pick(rng, ["2026-09-05T10:00", "2026-09-08T16:00"])
    return f"put {title} on my agenda for {dt}", {"name": "add_agenda_item", "arguments": {"title": title, "datetime": dt}}


def _ood_compose(rng: random.Random) -> tuple[str, dict]:
    to, body = _pick(rng, HANDLES), _pick(rng, ["see you soon", "meeting moved"])
    return f"compose a message to {to} saying {body}", {"name": "compose_message", "arguments": {"to": to, "body": body}}


def _ood_forecast(rng: random.Random) -> tuple[str, dict]:
    city = _pick(rng, CITIES)
    return f"look up the forecast for {city}", {"name": "lookup_forecast", "arguments": {"city": city}}


OOD_MAKERS = [_ood_lamp, _ood_climate, _ood_agenda, _ood_compose, _ood_forecast]


def _distractors(rng: random.Random, used: set[str], k: int, split: str = "train") -> list[Tool]:
    # only the ood split may draw OOD tools; "eval" is in-distribution held-out
    pool = [t for t in (TRAIN + OOD if split == "ood" else TRAIN) if t.name not in used]
    rng.shuffle(pool)
    return pool[:k]


def _name_mask(tool: Tool, alias: str) -> Tool:
    return Tool(alias, tool.description, tool.properties, tool.required, tool.split, tool.domain)


def make_example(rng: random.Random, split: str = "train") -> dict:
    makers = OOD_MAKERS + MAKERS if split == "ood" else MAKERS
    roll = rng.random()
    if roll < 0.15:
        kind, gold_src = "refuse", []
        query = _pick(rng, OFFTOPIC)
        used: set[str] = set()
    elif roll < 0.40:
        kind = "two"
        a, ca = _pick(rng, makers)(rng)
        b, cb = _pick(rng, makers)(rng)
        if ca["name"] == cb["name"]:
            # one retry for variety; a repeat after that is a legitimate
            # parallel-same-tool example, keep it
            b, cb = _pick(rng, makers)(rng)
        query = _pick(rng, PREFIX) + a + _pick(rng, ANDS) + b
        gold_src = [ca, cb]
        used = {ca["name"], cb["name"]}
    else:
        kind = "one"
        q, c = _pick(rng, makers)(rng)
        query = _pick(rng, PREFIX) + q
        gold_src = [c]
        used = {c["name"]}

    gold = canon_calls(gold_src)
    tools = [by_name(n) for n in used] if used else []
    extras = _distractors(rng, used, rng.randint(3, 6), split)
    catalog = tools + extras
    # Hammer: sometimes rename gold tools in the catalog + answers
    if used and rng.random() < 0.12 and kind != "refuse":
        mapping = {}
        new_cat = []
        for t in catalog:
            if t.name in used and rng.random() < 0.7:
                alias = t.name + "_v2"
                mapping[t.name] = alias
                new_cat.append(_name_mask(t, alias))
            else:
                new_cat.append(t)
        catalog = new_cat
        gold = [{"name": mapping.get(c["name"], c["name"]), "arguments": c["arguments"]} for c in gold]

    rng.shuffle(catalog)
    return {
        "query": query.strip(),
        "tools": [schema(t) for t in catalog],
        "answers": gold,
        "kind": kind,
        "split": split,
    }


def generate(n: int, seed: int = 0, split: str = "train") -> list[dict]:
    rng = random.Random(seed)
    return [make_example(rng, split) for _ in range(n)]
