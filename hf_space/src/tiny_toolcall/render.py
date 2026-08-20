"""Prompt render + per-token loss tags (structure/keys/names/values/stop)."""

from __future__ import annotations

from typing import Any

from tiny_toolcall.schema import dumps_calls

# tag ids used by the weighted CE
T_PAD, T_PROMPT, T_STRUCT, T_KEY, T_NAME, T_VAL, T_STOP = 0, 0, 1, 2, 3, 4, 5


_TYPE_ABBR = {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj", "array": "arr"}


def tool_signature(t: dict[str, Any]) -> str:
    """Compact declaration: `- name(param:str!, mode:enum[heat|cool]) description`.

    JSON-schema prompts waste tokens under the structural-singleton tokenizer
    contract (every brace/quote is its own token); the contract only pays off on
    the generated call, so the prompt uses this terse form. `!` marks required.
    """
    params = t.get("parameters", {})
    props = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])
    parts = []
    for k in sorted(props):
        spec = props.get(k) or {}
        if spec.get("enum"):
            typ = "enum(" + "|".join(str(e) for e in spec["enum"]) + ")"
        else:
            typ = _TYPE_ABBR.get(spec.get("type", "string"), str(spec.get("type", "str")))
        parts.append(f"{k}={typ}{'!' if k in required else ''}")
    desc = t.get("description", "").strip()
    # no structural chars ( : " { } [ ] , ) anywhere: everything merges into
    # words. space after the name keeps its token span clean for the name head.
    return f"- {t['name']} ({' '.join(parts)}) {desc}"


def compact_tools(tools: list[dict[str, Any]]) -> str:
    return "\n".join(tool_signature(t) for t in tools)


def prompt_text(query: str, tools: list[dict[str, Any]]) -> str:
    return f"<tools>\n{compact_tools(tools)}\n</tools>\n<query>\n{query}\n</query>\n<call>\n"


def tag_call(call_json: str) -> list[int]:
    """Char-level tags aligned to the call JSON string (before BPE).

    Single-pass scanner tracking string state and bracket depth: the stop decision
    is only the `,` / `]` at top-level array depth; commas inside string values
    stay T_VAL, commas between argument pairs stay T_STRUCT.
    """
    n = len(call_json)
    tags = [T_STRUCT] * n
    depth = 0  # [ and { nesting
    in_str = False
    str_start = 0
    pending_key: str | None = None  # last completed string, if it turns out to be a key
    pending_span = (0, 0)
    i = 0
    while i < n:
        ch = call_json[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
                pending_span = (str_start, i)
                pending_key = call_json[str_start:i]
            i += 1
            continue
        if ch == '"':
            in_str = True
            str_start = i + 1
            i += 1
            continue
        if ch == ":":
            # the string we just closed was a key
            s, e = pending_span
            tag = T_KEY
            for j in range(s, e):
                tags[j] = tag
            # peek: if the value is a string, tag it after it closes below
            # mark whether this key was "name" for the value tag
            tags_val = T_NAME if pending_key == "name" else T_VAL
            # scan the value if it is a string
            j = i + 1
            if j < n and call_json[j] == '"':
                k = j + 1
                while k < n:
                    if call_json[k] == "\\":
                        k += 2
                        continue
                    if call_json[k] == '"':
                        break
                    k += 1
                for m in range(j + 1, k):
                    tags[m] = tags_val
                i = k + 1
                continue
            if j < n and call_json[j] in "{[":
                # object/array value: let the main loop track brackets; nested
                # keys/values get tagged when their own ':' is reached
                i = j
                continue
            # non-string scalar value (number / bool / null)
            k = j
            while k < n and call_json[k] not in ',}]':
                tags[k] = T_VAL
                k += 1
            i = k
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if ch == "]" and depth == 0:
                tags[i] = T_STOP
        elif ch == "," and depth == 1:
            tags[i] = T_STOP
        i += 1
    return tags


def render_example(ex: dict[str, Any]) -> tuple[str, str, list[int]]:
    prompt = prompt_text(ex["query"], ex["tools"])
    call = dumps_calls(ex["answers"])
    return prompt, call, tag_call(call)
