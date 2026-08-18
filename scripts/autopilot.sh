#!/usr/bin/env bash
# Local orchestrator: tokenizer -> verify gate -> pack 768 -> push both pods ->
# launch both runs -> checkpoint pull loop. Any gate failure stops the line.
set -uo pipefail
cd /Users/nikshepsvn/tiny-toolcall
V=.venv/bin/python
log() { echo "[autopilot $(date +%H:%M:%S)] $*"; }

log "waiting for tokenizer"
for i in $(seq 1 90); do grep -q "vocab=" /tmp/tok_v5_fast.log 2>/dev/null && break; sleep 60; done
grep -q "vocab=" /tmp/tok_v5_fast.log || { log "ABORT: tokenizer never finished"; exit 1; }
log "tokenizer done: $(grep vocab= /tmp/tok_v5_fast.log)"

log "verification gate"
PYTHONPATH=src $V scripts/verify_tokenizer.py || { log "ABORT: tokenizer failed verification"; exit 1; }

log "packing at seq 768"
PYTHONPATH=src $V -m tiny_toolcall.cli pack --seq-len 768 || { log "ABORT: pack failed"; exit 1; }
grep -q "packed" /dev/null 2>&1 || true
$V - <<'PY' || { echo "[autopilot] ABORT: pack sanity failed"; exit 1; }
import numpy as np
ids = np.load("data/packed/train/ids.npy", mmap_mode="r")
assert ids.shape[0] > 500000, f"suspiciously few rows: {ids.shape}"
assert ids.shape[1] == 768, f"wrong seq len: {ids.shape}"
print(f"pack sane: {ids.shape}, mean real len {(ids[:5000] != 0).sum(1).mean():.0f}")
PY

log "pushing arrays + tokenizer to both pods"
for HP in "1898 193.183.22.57" "1519 193.183.22.62"; do
  set -- $HP
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $1" \
    data/packed/train/ids.npy data/packed/train/tags.npy data/packed/train/decisions.jsonl \
    root@$2:/workspace/tiny-toolcall/data/packed/train/ || { log "ABORT: push to $2 failed"; exit 1; }
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $1" data/tokenizer.json \
    root@$2:/workspace/tiny-toolcall/data/ || { log "ABORT: tokenizer push to $2 failed"; exit 1; }
done

log "launching both pods"
bash scripts/launch_v5.sh || { log "ABORT: launch failed"; exit 1; }

log "starting checkpoint pull loop"
bash scripts/pull_checkpoints.sh
log "AUTOPILOT COMPLETE — both runs finished, checkpoints pulled"
