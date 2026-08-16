#!/usr/bin/env bash
# Full SFT on a RunPod 4090. Usage: scripts/remote_sft.sh <host> <port>
# (get host/port from: python scripts/runpod.py status <pod_id> -> portMappings["22"])
set -euo pipefail
HOST=${1:?host}
PORT=${2:?port}
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
RS="rsync -az -e \"ssh -o StrictHostKeyChecking=no -p $PORT\""

echo "== push repo (code + packed data + tokenizer, no .env) =="
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  --exclude .git --exclude .venv --exclude .env --exclude 'data/raw' \
  --exclude 'data/seeds' --exclude 'data/eval' --exclude checkpoints \
  ./ root@"$HOST":/workspace/tiny-toolcall/

$SSH bash -s <<'EOF'
set -euo pipefail
cd /workspace/tiny-toolcall
pip install -q -e . 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m tiny_toolcall.cli train --name sft 2>&1 | tail -20
EOF

echo "== pull checkpoint =="
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  root@"$HOST":/workspace/tiny-toolcall/checkpoints/sft.pt checkpoints/
echo "done — remember: python scripts/runpod.py terminate <pod_id>"
