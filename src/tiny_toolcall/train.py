"""SFT loop: weighted CE from loss tags + name-head aux CE. Muon on trunk 2D
weights, AdamW on embeddings / norms / name head."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tiny_toolcall.model import ToolTransformer
from tiny_toolcall.render import T_KEY, T_NAME, T_PAD, T_STOP, T_STRUCT, T_VAL


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        xxt = x @ x.T
        x = a * x + (b * xxt + c * xxt @ xxt) @ x
    if transposed:
        x = x.T
    return x.type_as(g)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                m.mul_(mom).add_(p.grad)
                u = _newton_schulz(m.add(p.grad, alpha=mom))  # nesterov-style
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr * scale)


def split_params(model: ToolTransformer):
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and name.startswith("blocks."):
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def loss_weights_from_cfg(loss_cfg: dict[str, float], device: torch.device) -> torch.Tensor:
    w = torch.zeros(6, device=device)
    w[T_PAD] = 0.0
    w[T_STRUCT] = loss_cfg.get("structure", 1.0)
    w[T_KEY] = loss_cfg.get("keys", 1.5)
    w[T_NAME] = loss_cfg.get("names", 2.0)
    w[T_VAL] = loss_cfg.get("values", 4.0)
    w[T_STOP] = loss_cfg.get("stop", 6.0)
    return w


def step_loss(
    model: ToolTransformer,
    ids: torch.Tensor,
    tags: torch.Tensor,
    decisions: list[dict[str, Any]],
    weights: torch.Tensor,
    name_loss_w: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, hidden = model(ids)
    # next-token prediction: logits[t] predicts ids[t+1], weighted by tag[t+1]
    tgt = ids[:, 1:]
    w = weights[tags[:, 1:].long()]
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1).long(), reduction="none"
    ).view(tgt.shape)
    denom = w.sum().clamp(min=1.0)
    lm_loss = (ce * w).sum() / denom

    name_losses = []
    correct = total = 0
    for b, dec in enumerate(decisions):
        for pos, spans, gold in zip(dec["pos"], dec["spans"], dec["gold"]):
            scores = model.name_scores(hidden, pos, [tuple(s) for s in spans], batch=b)
            name_losses.append(F.cross_entropy(scores[None].float(), torch.tensor([gold], device=ids.device)))
            correct += int(scores.argmax().item() == gold)
            total += 1
    name_loss = torch.stack(name_losses).mean() if name_losses else lm_loss.new_zeros(())
    loss = lm_loss + name_loss_w * name_loss
    stats = {
        "lm": lm_loss.item(),
        "name": name_loss.item() if name_losses else 0.0,
        "name_acc": correct / total if total else 0.0,
    }
    return loss, stats


def train(
    model: ToolTransformer,
    ids: np.ndarray,
    tags: np.ndarray,
    decisions: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device | None = None,
    log_every: int = 20,
    save_path: Path | None = None,
) -> dict[str, float]:
    if device is None:
        device = pick_device()
    model.to(device)
    model.train()
    weights = loss_weights_from_cfg(cfg.get("loss", {}), device)
    tr = cfg.get("train", {})
    epochs = int(tr.get("epochs", 4))
    seq_len = ids.shape[1]
    batch = max(1, int(tr.get("global_tokens", 65536)) // seq_len)
    if device.type != "cuda":
        batch = min(batch, 16)

    muon_p, adamw_p = split_params(model)
    opt_m = Muon(muon_p, lr=float(tr.get("lr_muon", 0.02)), weight_decay=float(tr.get("wd", 0.01)))
    opt_a = torch.optim.AdamW(adamw_p, lr=float(tr.get("lr_adam", 3e-4)), weight_decay=float(tr.get("wd", 0.01)))

    n = ids.shape[0]
    steps_total = max(1, math.ceil(n / batch)) * epochs
    warmup = max(10, steps_total // 20)
    step = 0
    stats: dict[str, float] = {}
    t0 = time.time()
    ids_t = torch.from_numpy(ids.astype(np.int64))
    tags_t = torch.from_numpy(tags.astype(np.int64))
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            sel = perm[s : s + batch]
            bi = ids_t[sel].to(device)
            bt = tags_t[sel].to(device)
            bd = [decisions[j] for j in sel]
            loss, stats = step_loss(model, bi, bt, bd, weights)
            for opt in (opt_m, opt_a):
                opt.zero_grad(set_to_none=True)
            loss.backward()
            # warmup-stable-decay
            frac = step / steps_total
            mult = min(1.0, (step + 1) / warmup) * (1.0 if frac < 0.8 else max(0.05, (1 - frac) / 0.2))
            for opt, base in ((opt_m, float(tr.get("lr_muon", 0.02))), (opt_a, float(tr.get("lr_adam", 3e-4)))):
                for g in opt.param_groups:
                    g["lr"] = base * mult
            opt_m.step()
            opt_a.step()
            step += 1
            if step % log_every == 0:
                tps = step * batch * seq_len / (time.time() - t0)
                print(
                    f"ep{ep} step{step}/{steps_total} loss={loss.item():.4f} lm={stats['lm']:.4f} "
                    f"name={stats['name']:.4f} name_acc={stats['name_acc']:.3f} tok/s={tps:,.0f}"
                )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, save_path)
    return stats
