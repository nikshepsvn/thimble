#!/usr/bin/env bash
# 30-min pull loop for the v6 twins; both pods share one host, different ports.
set -u
cd "$(dirname "$0")/.."
HOST=193.183.22.57
mkdir -p checkpoints/pulls/v6c checkpoints/pulls/v6s
while true; do
  rsync -az -e "ssh -p 1898 -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/tiny-toolcall/checkpoints/v6c*.pt" checkpoints/pulls/v6c/ 2>/dev/null
  rsync -az -e "ssh -p 1898 -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/overnight_v6.log" checkpoints/pulls/v6c_overnight.log 2>/dev/null
  rsync -az -e "ssh -p 1434 -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/tiny-toolcall/checkpoints/v6s*.pt" checkpoints/pulls/v6s/ 2>/dev/null
  rsync -az -e "ssh -p 1434 -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/overnight_v6.log" checkpoints/pulls/v6s_overnight.log 2>/dev/null
  a=$(grep -c POD_A_COMPLETE checkpoints/pulls/v6c_overnight.log 2>/dev/null || echo 0)
  b=$(grep -c POD_B_COMPLETE checkpoints/pulls/v6s_overnight.log 2>/dev/null || echo 0)
  [ "$a" -ge 1 ] && [ "$b" -ge 1 ] && { echo "both complete $(date)"; break; }
  sleep 1800
done
