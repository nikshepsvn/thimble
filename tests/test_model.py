"""Torch integration: param budget, forward shapes, grammar decode round-trip."""

import torch

from tiny_toolcall.eval import make_model_predictor
from tiny_toolcall.model import Config, ToolTransformer, build
from tiny_toolcall.render import render_example
from tiny_toolcall.schema import canon_calls
from tiny_toolcall.synth import generate
from tiny_toolcall.tokenizer import train_bpe


def test_param_budget_45m():
    model = build(8192)
    p = model.count_params()
    assert 42e6 < p < 47e6, f"{p/1e6:.1f}M is outside the 45M envelope"


def test_forward_shapes():
    cfg = Config(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv=2)
    model = ToolTransformer(cfg)
    ids = torch.randint(0, 256, (2, 32))
    logits, hidden = model(ids)
    assert logits.shape == (2, 32, 256)
    assert hidden.shape == (2, 32, 64)
    scores = model.name_scores(hidden, 10, [(2, 5), (6, 9)], batch=1)
    assert scores.shape == (2,)


def test_grammar_decode_well_formed_untrained():
    """An untrained model must still produce schema-valid calls under the grammar
    (possibly wrong ones) — well-formedness is structural, not learned."""
    rows = generate(120, seed=11)
    texts = []
    for ex in rows:
        p, c, _ = render_example(ex)
        texts.append(p + c)
    tok = train_bpe(texts, vocab_size=600)
    cfg = Config(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv=2)
    model = ToolTransformer(cfg).eval()
    device = torch.device("cpu")
    for heads in (True, False):
        predict = make_model_predictor(model, tok, device, use_name_head=heads)
        for ex in rows[:4]:
            calls = predict(ex)
            assert calls is not None
            assert calls == canon_calls(calls)
            tool_names = {t["name"] for t in ex["tools"]}
            for c in calls:
                assert c["name"] in tool_names  # grammar constrains names to the catalog
                tool = next(t for t in ex["tools"] if t["name"] == c["name"])
                props = tool["parameters"]["properties"]
                for k in c["arguments"]:
                    assert k in props  # keys are grammar-forced, never invented
