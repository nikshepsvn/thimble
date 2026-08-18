#!/usr/bin/env bash
# Fires both pods once the pack is pushed. Each pod: one training run, then its
# own on-record probes. Self-driving on-pod via setsid; nothing depends on this
# machine staying connected.
set -euo pipefail
A="ssh -o StrictHostKeyChecking=no -p 1898 root@193.183.22.57"
B="ssh -o StrictHostKeyChecking=no -p 1519 root@193.183.22.62"

$A "cd /workspace/tiny-toolcall && cat > /workspace/run.sh <<'POD'
cd /workspace/tiny-toolcall
echo \"=== \$(date) v5 main ===\"
python -m tiny_toolcall.cli train --name v5 --epochs 4
for ck in v5 v5_devbest v5_ema; do
  [ -f checkpoints/\${ck}.pt ] || continue
  echo \"--- probe \${ck} \$(date) ---\"
  python scripts/ab_maxcalls.py --ckpt \${ck} --n 150 --caps 4 2>/dev/null | grep max_calls
done
echo \"POD_A_COMPLETE \$(date)\"
POD
setsid nohup bash /workspace/run.sh > /workspace/overnight.log 2>&1 < /dev/null &"

$B "cd /workspace/tiny-toolcall && cat > /workspace/run.sh <<'POD'
cd /workspace/tiny-toolcall
echo \"=== \$(date) v5 RFT twin ===\"
python -m tiny_toolcall.cli train --name v5rft --epochs 4 --w-structure 0.1 --w-keys 0.1
for ck in v5rft v5rft_devbest v5rft_ema; do
  [ -f checkpoints/\${ck}.pt ] || continue
  echo \"--- probe \${ck} \$(date) ---\"
  python scripts/ab_maxcalls.py --ckpt \${ck} --n 150 --caps 4 2>/dev/null | grep max_calls
done
echo \"POD_B_COMPLETE \$(date)\"
POD
setsid nohup bash /workspace/run.sh > /workspace/overnight.log 2>&1 < /dev/null &"
echo "both pods launched"
