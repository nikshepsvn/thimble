"""Export checkpoint + tokenizer to flat binaries for the C engine.

    uv run python cengine/export.py --ckpt thimble-v6

Writes cengine/thimble.bin (fp32 weights, mmap-able) and cengine/tokenizer.bin.
The tensor order is the contract with thimble.c — change one, change both.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LAYER_TENSORS = [
    "n1.weight", "attn.q.weight", "attn.k.weight", "attn.v.weight",
    "attn.o.weight", "attn.gate.weight", "attn.q_norm.weight",
    "attn.k_norm.weight", "n2.weight", "n3.weight", "n4.weight",
    "w1.weight", "w2.weight", "w3.weight",
]


def export_weights(ckpt: Path, out: Path, quant: bool = False) -> None:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, sd = blob["cfg"], blob["model"]
    ffn = int(cfg["d_model"] * cfg["ffn_mult"])
    ffn = (ffn + 7) // 8 * 8
    version = 2 if quant else 1
    with open(out, "wb") as f:
        f.write(struct.pack("<II", 0x424D4854, version))  # 'THMB'
        f.write(struct.pack("<7i", cfg["vocab_size"], cfg["d_model"], cfg["n_layers"],
                            cfg["n_heads"], cfg["n_kv"], ffn, cfg["max_seq"]))
        f.write(struct.pack("<f", cfg["rope_theta"]))

        def w(name: str) -> None:
            t = sd[name].float().contiguous()
            if quant and t.dim() == 2:   # 2-D == exactly the matmul weights
                # per-row absmax int8: scale fp32, rows int8
                s = t.abs().amax(dim=1).clamp_min(1e-12) / 127.0
                q = torch.round(t / s[:, None]).clamp(-127, 127).to(torch.int8)
                f.write(s.numpy().tobytes())
                f.write(q.numpy().tobytes())
            else:
                f.write(t.numpy().tobytes())

        w("embed.weight")
        for layer in range(cfg["n_layers"]):
            for t in LAYER_TENSORS:
                w(f"blocks.{layer}.{t}")
        w("norm.weight")
        w("name_head.weight")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, quant={quant})")


def export_tokenizer(tok_json: Path, out: Path) -> None:
    raw = json.loads(tok_json.read_text())
    vocab, merges = raw["vocab"], raw["merges"]
    id_to_tok = {i: t for t, i in vocab.items()}
    n = len(vocab)
    assert sorted(id_to_tok) == list(range(n)), "vocab ids must be contiguous"
    with open(out, "wb") as f:
        f.write(struct.pack("<II", 0x4B4F5454, 1))  # 'TTOK', version
        f.write(struct.pack("<I", n))
        for i in range(n):
            b = id_to_tok[i].encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(merges)))
        for a, b in merges:
            ab, bb = a.encode("utf-8"), b.encode("utf-8")
            f.write(struct.pack("<H", len(ab)))
            f.write(ab)
            f.write(struct.pack("<H", len(bb)))
            f.write(bb)
    print(f"wrote {out} ({n} tokens, {len(merges)} merges)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="thimble-v6")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    export_weights(ROOT / "checkpoints" / f"{a.ckpt}.pt", here / "thimble.bin")
    export_weights(ROOT / "checkpoints" / f"{a.ckpt}.pt", here / "thimble-q8.bin", quant=True)
    export_tokenizer(ROOT / "data" / "tokenizer.json", here / "tokenizer.bin")


if __name__ == "__main__":
    main()
