#!/usr/bin/env bash
# One train/eval cycle: tokenizer -> pack -> push -> SFT on pod -> parallel evals.
# Usage: scripts/cycle.sh <host> <port> [epochs]
set -euo pipefail
HOST=${1:?host}
PORT=${2:?port}
EPOCHS=${3:-2}
S="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
cd "$(dirname "$0")/.."

echo "== tokenizer (diverse mix, 8k) =="
.venv/bin/python -m tiny_toolcall.cli tok --vocab 8192 --sample 16000

echo "== pack train + eval splits =="
.venv/bin/python -m tiny_toolcall.cli pack --split train --seq-len 640
.venv/bin/python -m tiny_toolcall.cli pack --split eval
.venv/bin/python -m tiny_toolcall.cli pack --split ood

echo "== push =="
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  src/tiny_toolcall/ root@"$HOST":/workspace/tiny-toolcall/src/tiny_toolcall/
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  scripts/pod_eval.py root@"$HOST":/workspace/tiny-toolcall/scripts/
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  data/tokenizer.json root@"$HOST":/workspace/tiny-toolcall/data/
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  data/packed/ root@"$HOST":/workspace/tiny-toolcall/data/packed/
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  data/seeds/local_eval.jsonl data/seeds/local_ood.jsonl root@"$HOST":/workspace/tiny-toolcall/data/seeds/

echo "== SFT on pod (epochs=$EPOCHS) =="
# tee to a pod-side file so step progress (stepN/total) is tail-able live
$S "cd /workspace/tiny-toolcall && env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m tiny_toolcall.cli train --name sft --epochs $EPOCHS 2>&1 | tee train.log | tail -8"

echo "== parallel evals =="
$S 'pkill -f "[p]od_eval" 2>/dev/null || true; cd /workspace/tiny-toolcall; nohup env PYTHONUNBUFFERED=1 python scripts/pod_eval.py --suite local-eval --n-eval 400 > eval_le.log 2>&1 < /dev/null & nohup env PYTHONUNBUFFERED=1 python scripts/pod_eval.py --suite local-ood --n-ood 400 > eval_ood.log 2>&1 < /dev/null & nohup env PYTHONUNBUFFERED=1 python scripts/pod_eval.py --suite mobile-actions --n-ma 400 > eval_ma.log 2>&1 < /dev/null & sleep 2; echo evals-running'
echo "cycle launched; results land in eval_le.log / eval_ood.log / eval_ma.log on the pod"
