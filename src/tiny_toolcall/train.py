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
    if ids.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, hidden = model(ids)
    else:
        logits, hidden = model(ids)
    # next-token prediction: logits[t] predicts ids[t+1], weighted by tag[t+1]
    tgt = ids[:, 1:]
    w = weights[tags[:, 1:].long()]
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1).long(), reduction="none"
    ).view(tgt.shape)
    denom = w.sum().clamp(min=1.0)
    lm_loss = (ce * w).sum() / denom
    # z-loss on supervised positions: Needle's recipe carries it as the logit
    # stabilizer, and our 16k vocab doubles the room for logit drift
    lse = torch.logsumexp(logits[:, :-1].float(), dim=-1)
    z_loss = ((lse ** 2) * (w > 0)).sum() / denom * 1e-4
    lm_loss = lm_loss + z_loss

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
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.to(device)
    model.train()
    weights = loss_weights_from_cfg(cfg.get("loss", {}), device)
    tr = dict(cfg.get("train", {}))
    scale = float(tr.pop("lr_scale", 1.0))
    if scale != 1.0:
        tr["lr_muon"] = float(tr.get("lr_muon", 0.02)) * scale
        tr["lr_adam"] = float(tr.get("lr_adam", 3e-4)) * scale
        print(f"lr scaled x{scale}: muon={tr['lr_muon']:.4f} adam={tr['lr_adam']:.2e}")
    epochs = int(tr.get("epochs", 4))
    seq_len = ids.shape[1]
    batch = max(1, int(tr.get("global_tokens", 65536)) // seq_len)
    if device.type != "cuda":
        batch = min(batch, 16)
    else:
        batch = min(batch, 32)  # 24GB with bf16 activations; raise on bigger cards

    muon_p, adamw_p = split_params(model)
    opt_m = Muon(muon_p, lr=float(tr.get("lr_muon", 0.02)), weight_decay=float(tr.get("wd", 0.01)))
    try:
        opt_a = torch.optim.AdamW(adamw_p, lr=float(tr.get("lr_adam", 3e-4)),
                                  weight_decay=float(tr.get("wd", 0.01)),
                                  fused=(device.type == "cuda"))
    except (RuntimeError, TypeError):
        opt_a = torch.optim.AdamW(adamw_p, lr=float(tr.get("lr_adam", 3e-4)),
                                  weight_decay=float(tr.get("wd", 0.01)))

    # dev split: the LAST dev_rows of the packed order are held out entirely.
    # Dev is train-mix-derived — it exists to pick checkpoints, never to tune on
    # eval suites — and it is excluded before batching so no gradient sees it.
    dev_rows = int(tr.get("dev_rows", 5000))
    n_all = ids.shape[0]
    dev_idx = np.arange(max(0, n_all - dev_rows), n_all)
    keep = np.arange(0, max(0, n_all - dev_rows))
    dev_ids, dev_tags = ids[dev_idx], tags[dev_idx]
    dev_dec = [decisions[j] for j in dev_idx.tolist()]
    ids, tags = ids[keep], tags[keep]
    decisions = [decisions[j] for j in keep.tolist()]
    print(f"dev split: {len(dev_idx)} rows held out, {len(keep)} train")

    # EMA shadow (weight averaging is the best-supported free win in the 2026
    # pretraining literature); dev picks between raw/EMA at the end
    ema_beta = float(tr.get("ema_beta", 0.999))
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}

    n = ids.shape[0]
    step = 0
    best_dev = float("inf")
    stats: dict[str, float] = {}
    t0 = time.time()
    ids_t = torch.from_numpy(ids.astype(np.int64))
    tags_t = torch.from_numpy(tags.astype(np.int64))
    # token-budget batching: sort by real length, fill each batch to a fixed
    # token budget (rows x padded-len), so short rows train in big batches and
    # long rows in small ones — ~3x less pad compute at constant memory
    real_len = (ids != 0).sum(axis=1)
    order = np.argsort(real_len, kind="stable")
    budget = batch * 512  # tokens per step, same memory envelope as before
    batches = []
    cur: list[int] = []
    cur_max = 0
    for idx in order.tolist():
        length = min(seq_len, (int(real_len[idx]) + 63) // 64 * 64)
        if cur and ((len(cur) + 1) * max(cur_max, length) > budget or len(cur) >= 128):
            batches.append(np.array(cur))
            cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, length)
    if cur:
        batches.append(np.array(cur))
    steps_total = len(batches) * epochs
    warmup = max(10, steps_total // 20)
    for ep in range(epochs):
        np.random.shuffle(batches)
        for sel in batches:
            blen = int(real_len[sel].max())
            blen = min(seq_len, (blen + 63) // 64 * 64)
            bi = ids_t[sel][:, :blen].to(device)
            bt = tags_t[sel][:, :blen].to(device)
            bd = [decisions[j] for j in sel]
            loss, stats = step_loss(model, bi, bt, bd, weights)
            for opt in (opt_m, opt_a):
                opt.zero_grad(set_to_none=True)
            loss.backward()
            # one pathological imported batch must not get to move the weights
            # arbitrarily; 1.0 only engages on outliers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # warmup-stable-decay
            frac = step / steps_total
            mult = min(1.0, (step + 1) / warmup) * (1.0 if frac < 0.8 else max(0.05, (1 - frac) / 0.2))
            for opt, base in ((opt_m, float(tr.get("lr_muon", 0.02))), (opt_a, float(tr.get("lr_adam", 3e-4)))):
                for g in opt.param_groups:
                    g["lr"] = base * mult
            opt_m.step()
            opt_a.step()
            with torch.no_grad():
                msd = model.state_dict()
                for k, v in ema.items():
                    v.mul_(ema_beta).add_(msd[k].float(), alpha=1.0 - ema_beta)
            step += 1
            if step % log_every == 0:
                tps = step * batch * seq_len / (time.time() - t0)
                print(
                    f"ep{ep} step{step}/{steps_total} loss={loss.item():.4f} lm={stats['lm']:.4f} "
                    f"name={stats['name']:.4f} name_acc={stats['name_acc']:.3f} tok/s={tps:,.0f}"
                )
            if step % 1000 == 0 and len(dev_idx):
                dl = _dev_loss(model, dev_ids[:1500], dev_tags[:1500], dev_dec[:1500],
                               weights, device)
                marker = ""
                if dl < best_dev:
                    best_dev = dl
                    if save_path:
                        _save(model, save_path.with_name(save_path.stem + "_devbest.pt"))
                    marker = "  <- new best"
                print(f"  dev@{step}: {dl:.4f}{marker}")
            if save_path and step % 500 == 0:
                _save(model, save_path)
    if save_path:
        _save(model, save_path)
        # EMA candidate: swap in, save, evaluate, restore
        raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema_sd = dict(raw_sd)
        for k, v in ema.items():
            ema_sd[k] = v.to(raw_sd[k].dtype)
        model.load_state_dict(ema_sd)
        _save(model, save_path.with_name(save_path.stem + "_ema.pt"))
        if len(dev_idx):
            dl_ema = _dev_loss(model, dev_ids, dev_tags, dev_dec, weights, device)
            model.load_state_dict(raw_sd)
            dl_final = _dev_loss(model, dev_ids, dev_tags, dev_dec, weights, device)
            print(f"SELECTION dev(full): final={dl_final:.4f} ema={dl_ema:.4f} "
                  f"best_periodic={best_dev:.4f}")
            stats["dev_final"], stats["dev_ema"] = dl_final, dl_ema
        else:
            model.load_state_dict(raw_sd)
    return stats


@torch.no_grad()
def _dev_loss(model, dev_ids, dev_tags, dev_dec, weights, device, rows_per=16) -> float:
    was_training = model.training
    model.eval()
    tot = n = 0.0
    ids_t = torch.from_numpy(dev_ids.astype(np.int64))
    tags_t = torch.from_numpy(dev_tags.astype(np.int64))
    real = (dev_ids != 0).sum(axis=1)
    for s in range(0, len(dev_ids), rows_per):
        sl = slice(s, s + rows_per)
        blen = int(real[sl].max())
        blen = min(dev_ids.shape[1], (blen + 63) // 64 * 64)
        loss, _ = step_loss(model, ids_t[sl][:, :blen].to(device),
                            tags_t[sl][:, :blen].to(device), dev_dec[sl], weights)
        tot += loss.item() * (sl.stop - s if sl.stop <= len(dev_ids) else len(dev_ids) - s)
        n += min(rows_per, len(dev_ids) - s)
    if was_training:
        model.train()
    return tot / max(1.0, n)


def _save(model: ToolTransformer, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__}, tmp)
    tmp.replace(save_path)
