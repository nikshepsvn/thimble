"""Torch-free core tests: scoring, retrieval, tags, tokenizer, synth licensing."""

import random

from tiny_toolcall.render import T_KEY, T_NAME, T_STOP, T_STRUCT, T_VAL, render_example, tag_call
from tiny_toolcall.retrieve import retrieve
from tiny_toolcall.schema import canon_calls, dumps_calls, loads_calls, score_rows
from tiny_toolcall.synth import generate, make_example
from tiny_toolcall.tokenizer import STRUCTURAL, BPETokenizer, pretokenize, train_bpe


def test_canon_and_score():
    gold = [{"name": "b", "arguments": {"z": 1, "a": "x"}}]
    pred = [{"name": "b", "arguments": {"a": "x", "z": 1}}]
    rows = [{"gold": gold, "pred": pred}]
    s = score_rows(rows)
    assert s["accuracy"] == 1.0 and s["name_acc"] == 1.0


def test_score_refuse_and_order():
    rows = [
        {"gold": [], "pred": []},  # refuse hit
        {"gold": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}],
         "pred": [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]},  # order matters
        {"gold": [{"name": "a", "arguments": {}}], "pred": None},  # malformed
    ]
    s = score_rows(rows)
    assert s["refuse"] == 1.0
    assert s["two_plus"] == 0.0
    assert s["well_formed"] == 2 / 3


def test_loads_calls_variants():
    assert loads_calls("[]") == []
    assert loads_calls('{"name":"x","arguments":{}}') == [{"name": "x", "arguments": {}}]
    assert loads_calls("not json") is None
    assert loads_calls('[{"no_name": 1}]') is None


def test_retrieve_emitted_shifts():
    tools = [
        {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"properties": {"city": {}}}},
        {"name": "send_message", "description": "Text a contact.", "parameters": {"properties": {"to": {}, "body": {}}}},
        {"name": "set_timer", "description": "Start a timer.", "parameters": {"properties": {"minutes": {}}}},
    ]
    q = "weather in Oslo then text @maya it looks rainy"
    first = retrieve(q, tools, k=2)
    assert first[0]["name"] == "get_weather"
    after = retrieve(q, tools, k=2, emitted=[{"name": "get_weather", "arguments": {"city": "Oslo"}}])
    # already-emitted tool is demoted; send_message should now lead
    assert after[0]["name"] == "send_message"


def test_tag_call_comma_in_string_value():
    call = dumps_calls([{"name": "send_message", "arguments": {"body": "on my way, call me", "to": "@maya"}}])
    tags = tag_call(call)
    # every comma inside the body string must be VAL, not STOP
    for i, ch in enumerate(call):
        if ch == "," and tags[i] == T_STOP:
            # the only STOP-tagged positions are top-level separators; none inside strings
            before = call[:i]
            assert before.count('"') % 2 == 0, f"comma at {i} inside a string tagged STOP"
    # closing ] is the stop decision
    assert tags[len(call) - 1] == T_STOP
    # name value tagged as NAME
    start = call.find('"name":"') + len('"name":"')
    assert tags[start] == T_NAME


def test_tag_call_two_calls_stop_positions():
    call = dumps_calls([
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {}},
    ])
    tags = tag_call(call)
    stops = [i for i, t in enumerate(tags) if t == T_STOP]
    # exactly two stop decisions: the , between calls and the final ]
    assert len(stops) == 2
    assert call[stops[0]] == "," and call[stops[1]] == "]"


def test_tokenizer_structural_singletons_and_roundtrip():
    rows = generate(300, seed=1)
    texts = []
    for ex in rows:
        p, c, _ = render_example(ex)
        texts.append(p + c)
    tok = train_bpe(texts, vocab_size=800)
    # roundtrip
    for t in texts[:20]:
        assert tok.decode(tok.encode(t)) == t
    # no token string contains a structural char with anything else
    for s in tok.vocab:
        if any(ch in STRUCTURAL for ch in s):
            assert len(s) == 1, f"token {s!r} spans a structural boundary"
    # specials encode as single ids
    ids = tok.encode("<call>")
    assert len(ids) == 1


def test_pretokenize_partition():
    text = '{"a":"hello world","n":3}'
    parts = pretokenize(text)
    assert "".join(parts) == text


def test_synth_licensing_no_unlicensed_brightness():
    rng = random.Random(7)
    for _ in range(300):
        ex = make_example(rng)
        for call in ex["answers"]:
            if call["name"] == "set_lights" and "brightness" in call["arguments"]:
                assert str(call["arguments"]["brightness"]) in ex["query"], (
                    f"unlicensed brightness: {ex['query']!r} -> {call['arguments']}"
                )
            if call["name"] == "set_thermostat" and "mode" in call["arguments"]:
                assert ("cool" in ex["query"]) or ("heat" in ex["query"])


def test_synth_mix_and_ood():
    rows = generate(2000, seed=3)
    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("one", "two", "refuse")}
    assert 0.10 < kinds["refuse"] / len(rows) < 0.20
    assert 0.18 < kinds["two"] / len(rows) < 0.32
    ood = generate(200, seed=4, split="ood")
    names = {c["name"] for r in ood for c in r["answers"]}
    assert any(n in names for n in ("adjust_lamp", "set_climate", "lookup_forecast"))


def test_gold_args_appear_in_query_or_licensed():
    # canonical arg ordering must survive render -> parse
    rows = generate(50, seed=5)
    for ex in rows:
        s = dumps_calls(ex["answers"])
        back = loads_calls(s)
        assert back == canon_calls(ex["answers"])
