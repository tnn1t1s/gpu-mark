"""Command line entry point: `gpu-mark run` and `gpu-mark params`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import health, rental
from .params import MethodParams


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def format_summary(doc: dict) -> str:
    lines = []
    inv = doc["inventory"]
    sw, drv = inv["software"], inv["driver"]
    lines.append(f"gpu-mark {doc['tool']['version']}  provider={doc['run']['provider']}  "
                 f"duration={doc['run']['duration_s']:.1f}s")
    lines.append(f"driver {drv['driver_version']}  torch {sw['torch']}  CUDA {sw['torch_cuda']}  cuDNN {sw['cudnn']}")
    reps = doc["method"]["parameters"]["repetitions"]["value"]
    for g in doc["results"]["per_gpu"]:
        lines.append("")
        lines.append(f"GPU {g['index']}  {g['torch_name']}  {g['uuid']}")
        lines.append(f"  {'metric':34s} {'median':>10s} {'min':>10s} {'max':>10s}  unit (n={reps})")
        for name, m in g["metrics"].items():
            lines.append(f"  {name:34s} {m['median']:10.1f} {m['min']:10.1f} {m['max']:10.1f}  {m['unit']}")
        for name, why in g["skipped"].items():
            lines.append(f"  {name:34s} skipped: {why}")
    ag = doc["results"]["all_gpu"]
    lines.append("")
    if ag["status"] == "ran":
        lines.append(f"All GPUs at once {ag['gpus']} (sum across GPUs)")
        for name, m in ag["metrics"].items():
            lines.append(f"  {name:34s} {m['median']:10.1f} {m['min']:10.1f} {m['max']:10.1f}  {m['unit']}")
    else:
        lines.append(f"All-GPU pass skipped: {ag['reason']}")
    for ph in doc["telemetry"]["phases"]:
        for gi, s in ph["summary"].items():
            if s["n_samples"]:
                lines.append(f"  telemetry {ph['name']:28s} power mean {s['power_w_mean']:.0f} W "
                             f"(limit {s['power_limit_w']:.0f})  temp max {s['temp_c_max']:.0f} C  "
                             f"SM clock {s['sm_clock_mhz_min']:.0f}-{s['sm_clock_mhz_max']:.0f} MHz")
    lines.append("")
    lines.append(health.format_findings(doc["health"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu-mark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark on this machine")
    r.add_argument("--provider", required=True, help="who supplies the machine (a cloud or owner name); recorded as given")
    r.add_argument("--out", required=True, help="path of the JSON result file to write")
    r.add_argument("--gpus", help="comma-separated CUDA device indices; default all")
    r.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a method parameter (see `gpu-mark params`)")
    r.add_argument("--json-stdout", action="store_true", help="also print the result JSON to stdout")
    rental.add_arguments(r)
    c = sub.add_parser("check", help="inventory and health checks only; no benchmark, no torch, a few seconds")
    c.add_argument("--provider", required=True, help="who supplies the machine; recorded as given")
    c.add_argument("--out", required=True, help="path of the JSON result file to write")
    c.add_argument("--json-stdout", action="store_true", help="also print the result JSON to stdout")
    rental.add_arguments(c)
    sub.add_parser("params", help="print the method parameters and their reasons")
    i = sub.add_parser("inference", help="serve an LLM with vLLM and measure throughput (needs `vllm` on PATH)")
    i.add_argument("--provider", required=True, help="who supplies the machine; recorded as given")
    i.add_argument("--out", required=True, help="path of the JSON result file, rewritten after each level")
    i.add_argument("--server-log", help="path for the vLLM server log; default a temporary directory")
    i.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override an inference parameter (see `gpu-mark inference-params`)")
    i.add_argument("--json-stdout", action="store_true", help="also print the result JSON to stdout")
    rental.add_arguments(i)
    sub.add_parser("inference-params", help="print the inference parameters and their reasons")
    args = ap.parse_args(argv)

    if args.cmd == "params":
        print(json.dumps(MethodParams().as_dict(), indent=2))
        return 0
    if args.cmd == "inference-params":
        from .inference_params import InferenceParams
        print(json.dumps(InferenceParams().as_dict(), indent=2))
        return 0
    if args.cmd == "check":
        from . import check
        doc = check.run(args.provider, rental.from_args(args))
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(check.format_summary(doc))
        print(f"\nresult file: {args.out}")
        if args.json_stdout:
            print("GPU_MARK_JSON_BEGIN")
            print(json.dumps(doc, separators=(",", ":")))
            print("GPU_MARK_JSON_END")
        return 0
    if args.cmd == "inference":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        from . import inference
        from .inference_params import InferenceParams
        ip = InferenceParams().with_values(**_parse_set(args.set))
        doc = inference.run(args.provider, ip, args.out, args.server_log, rental.from_args(args))
        print(inference.format_summary(doc))
        print(f"\nresult file: {args.out}")
        if args.json_stdout:
            print("GPU_MARK_JSON_BEGIN")
            print(json.dumps(doc, separators=(",", ":")))
            print("GPU_MARK_JSON_END")
        return 1 if any("fatal" in e for e in doc["errors"]) else 0

    # nvidia-smi numbers GPUs by PCI bus id. CUDA must use the same order so
    # inventory and results describe the same GPU; it reads this at init.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from . import runner  # imports torch

    p = MethodParams().with_values(**_parse_set(args.set))
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    doc = runner.run(args.provider, p, gpus, rental.from_args(args))
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(format_summary(doc))
    print(f"\nresult file: {args.out}")
    if args.json_stdout:
        print("GPU_MARK_JSON_BEGIN")
        print(json.dumps(doc, separators=(",", ":")))
        print("GPU_MARK_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
