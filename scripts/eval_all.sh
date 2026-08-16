#!/usr/bin/env bash
# Run every suite for a checkpoint, in parallel, gate on. Usage: eval_all.sh <ckpt>
set -u
CK=${1:-sft2}
cd /workspace/tiny-toolcall
for s in local-eval local-ood mobile-actions seal-tools-in seal-tools-out; do
  nohup env PYTHONUNBUFFERED=1 python scripts/pod_eval.py --suite "$s" --ckpt "$CK" \
    --n-eval 400 --n-ood 400 > "ev_${CK}_${s}.log" 2>&1 < /dev/null &
  sleep 1
done
echo "launched 5 suites for $CK"
