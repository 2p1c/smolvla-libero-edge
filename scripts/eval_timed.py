"""lerobot-eval with BF16 autocast and per-inference latency logging. Env, seeds and rollout are untouched.

lerobot-eval wraps rollouts in torch.autocast(device_type=...) when --policy.use_amp=true, which on CUDA
defaults to float16. This sets the default autocast dtype to bfloat16 first, so use_amp means BF16, then
times every SmolVLA action-chunk inference (CUDA-synchronized) and writes <output_dir>/runtime.json.

    LEROBOT_EVAL="python scripts/eval_timed.py" EPISODES=10 N_ACTION_STEPS=10 NUM_STEPS=5 \
        scripts/eval.sh <ckpt> <tag> --policy.use_amp=true --policy.compile_model=true --policy.compile_mode=default
"""

import atexit
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_eval

AUTOCAST_DTYPE = os.environ.get("AUTOCAST_DTYPE", "bfloat16")
torch.set_autocast_dtype("cuda", getattr(torch, AUTOCAST_DTYPE))

times_ms = []
_original = SmolVLAPolicy._get_action_chunk


def _timed(self, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = _original(self, *args, **kwargs)
    torch.cuda.synchronize()
    times_ms.append((time.perf_counter() - start) * 1000)
    return out


SmolVLAPolicy._get_action_chunk = _timed


def _arg(name):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


@atexit.register
def _dump():
    out_dir = _arg("output_dir")
    if not out_dir or not times_ms:
        return
    # The first calls include torch.compile warm-up; report them separately.
    warm = times_ms[:20]
    steady = sorted(times_ms[20:]) or sorted(times_ms)
    report = {
        "autocast_dtype_when_use_amp": AUTOCAST_DTYPE,
        "use_amp": _arg("policy.use_amp"),
        "compile_model": _arg("policy.compile_model"),
        "compile_mode": _arg("policy.compile_mode"),
        "eval_batch_size": _arg("eval.batch_size"),
        "note": "latency per batched action-chunk inference (all envs of a task at once), CUDA-synchronized",
        "n_inferences": len(times_ms),
        "warmup_first20_ms_max": round(max(warm), 1),
        "steady_p50_ms": round(statistics.median(steady), 2),
        "steady_p90_ms": round(steady[int(0.9 * (len(steady) - 1))], 2),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "runtime.json").write_text(json.dumps(report, indent=2))
    print("runtime:", report)


if __name__ == "__main__":
    lerobot_eval.main()
