#!/usr/bin/env bash
# v6 twin launch — run FROM the laptop; each pod gets a self-driving chain.
#   pod A (1898): v6c — continued from v5_mid4 plateau, no re-warmup,
#                 decay-phase data annealing (the treated run)
#   pod B (1434): v6s — from scratch, canonical recipe (the control)
# Champion by canonical dev ONLY across all candidates; probes are records.
set -euo pipefail
HOST=193.183.22.57

ssh -o StrictHostKeyChecking=no -p 1898 root@$HOST 'cat > /workspace/chain_v6c.sh << "EOF"
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tiny-toolcall
LOG=/workspace/overnight_v6.log
exec >> "$LOG" 2>&1
echo "=== $(date) v6c: continued from v5_mid4, no-warmup, anneal decay ==="
python -m tiny_toolcall.cli train --name v6c --epochs 2 --init v5_mid4 --no-warmup --anneal-data anneal
echo "=== $(date) v6c done ==="
for ck in v6c v6c_devbest v6c_ema; do
  [ -f "checkpoints/${ck}.pt" ] || continue
  echo "--- probe ${ck} $(date) ---"
  python scripts/ab_maxcalls.py --ckpt "${ck}" --n 150 --caps 4 2>/dev/null | grep max_calls
done
echo "POD_A_COMPLETE $(date)"
EOF
chmod +x /workspace/chain_v6c.sh
setsid nohup /workspace/chain_v6c.sh > /dev/null 2>&1 < /dev/null &
echo "v6c chain launched"'

ssh -o StrictHostKeyChecking=no -p 1434 root@$HOST 'cat > /workspace/chain_v6s.sh << "EOF"
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tiny-toolcall
LOG=/workspace/overnight_v6.log
exec >> "$LOG" 2>&1
echo "=== $(date) v6s: from scratch, canonical, 3 epochs ==="
python -m tiny_toolcall.cli train --name v6s --epochs 3
echo "=== $(date) v6s done ==="
for ck in v6s v6s_devbest v6s_ema; do
  [ -f "checkpoints/${ck}.pt" ] || continue
  echo "--- probe ${ck} $(date) ---"
  python scripts/ab_maxcalls.py --ckpt "${ck}" --n 150 --caps 4 2>/dev/null | grep max_calls
done
echo "POD_B_COMPLETE $(date)"
EOF
chmod +x /workspace/chain_v6s.sh
setsid nohup /workspace/chain_v6s.sh > /dev/null 2>&1 < /dev/null &
echo "v6s chain launched"'
