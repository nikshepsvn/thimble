#!/usr/bin/env bash
# Insurance loop: pull checkpoints + logs from both pods every 30 min so a pod
# death costs at most half an hour. Run locally, backgrounded.
mkdir -p checkpoints/pulls
for i in $(seq 1 24); do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p 1898" root@193.183.22.57:/workspace/tiny-toolcall/checkpoints/ checkpoints/pulls/A/ 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p 1898" root@193.183.22.57:/workspace/overnight.log checkpoints/pulls/A_overnight.log 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p 1519" root@193.183.22.62:/workspace/tiny-toolcall/checkpoints/ checkpoints/pulls/B/ 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p 1519" root@193.183.22.62:/workspace/overnight.log checkpoints/pulls/B_overnight.log 2>/dev/null
  grep -q "POD_A_COMPLETE" checkpoints/pulls/A_overnight.log 2>/dev/null && \
  grep -q "POD_B_COMPLETE" checkpoints/pulls/B_overnight.log 2>/dev/null && break
  sleep 1800
done
echo "pull loop done"
