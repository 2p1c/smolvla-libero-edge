"""Inference latency of one SmolVLA action-chunk prediction under deployment settings.

Feeds a real LIBERO frame through the checkpoint's own preprocessor, then times
policy.predict_action_chunk (VLM prefix + num_steps denoising steps) per config:
precision (fp32 / bf16 autocast), denoising steps, and torch.compile.

    python scripts/bench_latency.py --policy HuggingFaceVLA/smolvla_libero
    python scripts/bench_latency.py --policy outputs/smolvla_libero_ft/checkpoints/last/pretrained_model
"""

import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = Path(__file__).resolve().parent.parent
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CONTROL_HZ = 20  # LIBERO / robosuite control frequency

# (name, dtype, num_steps, compile). Compiled configs go last: compiling patches the model in place.
CONFIGS = [
    ("fp32", None, 10, False),
    ("bf16", torch.bfloat16, 10, False),
    ("bf16", torch.bfloat16, 5, False),
    ("bf16+compile", torch.bfloat16, 10, True),
    ("bf16+compile", torch.bfloat16, 5, True),
]


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_policy(path, device):
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.config.device = device
    policy.to(device).eval()
    expects_camera1 = "observation.images.camera1" in policy.config.input_features
    rename_map = RENAME_MAP if expects_camera1 else {}
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return policy, preprocessor


def load_observation(dataset_id):
    """One real frame (two cameras, state, instruction) from the training data, without the action."""
    dataset = LeRobotDataset(dataset_id, episodes=[0])
    item = dataset[0]
    return {k: v for k, v in item.items() if k.startswith("observation.") or k == "task"}


def bench(policy, batch, device, dtype, num_steps, warmup, iters):
    policy.config.num_steps = num_steps
    autocast = torch.autocast(device_type=device, dtype=dtype) if dtype else nullcontext()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times_ms = []
    with torch.inference_mode(), autocast:
        for i in range(warmup + iters):
            policy.reset()
            sync(device)
            start = time.perf_counter()
            policy.predict_action_chunk(dict(batch))
            sync(device)
            if i >= warmup:
                times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    p50 = statistics.median(times_ms)
    p90 = times_ms[int(0.9 * (len(times_ms) - 1))]
    step_ms = 1000 / CONTROL_HZ
    return {
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "inference_hz": round(1000 / p50, 1),
        # With async execution the robot keeps running the current chunk while the next one is computed,
        # so a chunk must cover at least p90 latency worth of control steps.
        "min_n_action_steps_at_20hz": max(1, math.ceil(p90 / step_ms)),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile configs")
    parser.add_argument("--out", default=str(ROOT / "results" / "latency.json"))
    args = parser.parse_args()

    device = pick_device(args.device)
    policy, preprocessor = load_policy(args.policy, device)
    batch = preprocessor(load_observation(args.dataset))
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"policy={args.policy} device={hardware} chunk_size={policy.config.chunk_size}")

    results = []
    compiled = False
    for name, dtype, num_steps, compile_model in CONFIGS:
        if compile_model and args.no_compile:
            continue
        if dtype is torch.bfloat16 and device == "cpu":
            continue
        if compile_model and not compiled:
            policy.model.sample_actions = torch.compile(policy.model.sample_actions)
            compiled = True
        row = {"config": name, "num_steps": num_steps, **bench(
            policy, batch, device, dtype, num_steps, args.warmup, args.iters
        )}
        results.append(row)
        print(row)

    print("\n| config | denoise steps | p50 ms | p90 ms | Hz | min n @20Hz | peak GB |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['config']} | {r['num_steps']} | {r['p50_ms']} | {r['p90_ms']} | {r['inference_hz']} | "
            f"{r['min_n_action_steps_at_20hz']} | {r['peak_mem_gb'] if r['peak_mem_gb'] is not None else '-'} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "device": hardware, "results": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
