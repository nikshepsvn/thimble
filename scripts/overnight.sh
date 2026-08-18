#!/usr/bin/env bash
# v5 overnight chain — runs ON the pod, fully self-driving so a dropped SSH
# monitor cannot repeat v4's seven idle hours.
set -uo pipefail
cd /workspace/tiny-toolcall
LOG=/workspace/overnight.log
exec >> "$LOG" 2>&1

echo "=== $(date) run 1: v5 main (current recipe, solo, full speed) ==="
python -m tiny_toolcall.cli train --name v5 --epochs 4
echo "=== $(date) run 1 done ==="

echo "=== $(date) run 2: v5 RFT twin (forced tokens at 0.1x) ==="
python -m tiny_toolcall.cli train --name v5rft --epochs 4 --w-structure 0.1 --w-keys 0.1
echo "=== $(date) run 2 done ==="

echo "=== $(date) probes (150-row Seal-in + MA per candidate; FOR THE RECORD —"
echo "    champion selection is by dev loss only, printed in each train log) ==="
for ck in v5 v5_devbest v5_ema v5rft v5rft_devbest v5rft_ema; do
  [ -f "checkpoints/${ck}.pt" ] || continue
  echo "--- ${ck} $(date) ---"
  python scripts/ab_maxcalls.py --ckpt "${ck}" --n 150 --caps 4 2>/dev/null | grep max_calls
done
echo "OVERNIGHT_CHAIN_COMPLETE $(date)"
