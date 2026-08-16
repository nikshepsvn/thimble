"""RunPod community-cloud 4090 lifecycle: create / status / stop / terminate.

Usage:
  .venv/bin/python scripts/runpod.py create [--gpu "NVIDIA GeForce RTX 4090"]
  .venv/bin/python scripts/runpod.py status <pod_id>
  .venv/bin/python scripts/runpod.py terminate <pod_id>

Kill the pod when idle — the $80 cap is wall-clock money.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"


def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "")
    if not k:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("no RUNPOD_API_KEY")
    return k


def _client() -> httpx.Client:
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {_key()}"}, timeout=60)


def create(gpu: str) -> None:
    body = {
        "name": "tiny-toolcall-sft",
        "cloudType": "COMMUNITY",
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {},
    }
    with _client() as c:
        r = c.post("/pods", json=body)
        if r.status_code >= 400:
            sys.exit(f"create failed {r.status_code}: {r.text}")
        pod = r.json()
        print(json.dumps({"id": pod.get("id"), "status": pod.get("desiredStatus")}, indent=2))


def status(pod_id: str) -> None:
    with _client() as c:
        r = c.get(f"/pods/{pod_id}")
        r.raise_for_status()
        pod = r.json()
        out = {
            "id": pod.get("id"),
            "desiredStatus": pod.get("desiredStatus"),
            "costPerHr": pod.get("costPerHr"),
            "publicIp": pod.get("publicIp"),
            "portMappings": pod.get("portMappings"),
        }
        print(json.dumps(out, indent=2))


def terminate(pod_id: str) -> None:
    with _client() as c:
        r = c.delete(f"/pods/{pod_id}")
        print(r.status_code, r.text[:200])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    p = sub.add_parser("status")
    p.add_argument("pod_id")
    p = sub.add_parser("terminate")
    p.add_argument("pod_id")
    args = ap.parse_args()
    if args.cmd == "create":
        create(args.gpu)
    elif args.cmd == "status":
        status(args.pod_id)
    else:
        terminate(args.pod_id)


if __name__ == "__main__":
    main()
