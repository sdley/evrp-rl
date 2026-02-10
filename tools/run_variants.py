#!/usr/bin/env python3
"""Create variant config files from a base unified config by varying agent types and encoders.

Examples:
  python tools/run_variants.py --base configs/experiment_config.yaml --agents a2c,sac --encoders gat,mlp --out configs/variants
  python tools/run_variants.py --base configs/experiment_config.yaml --agents a2c --encoders gat --run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path: Path):
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def build_variants(base_cfg, agents, encoders, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for agent in agents:
        for encoder in encoders:
            cfg = dict(base_cfg) if base_cfg is not None else {}
            # Ensure we set `agent` as a dict following the unified schema
            cfg["agent"] = {
                "type": agent,
                "encoder": encoder,
            }
            # optionally provide minimal run defaults if missing
            cfg.setdefault("run", {})
            name = f"{agent}_{encoder}"
            out_path = out_dir / f"{name}.yaml"
            dump_yaml(cfg, out_path)
            created.append(out_path)
    return created


def run_train(config_path: Path):
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=False, help="Base config YAML path (optional)")
    p.add_argument("--agents", required=True, help="Comma-separated agent types (e.g. a2c,sac)")
    p.add_argument("--encoders", required=True, help="Comma-separated encoders (e.g. gat,mlp)")
    p.add_argument("--out", default="configs/variants", help="Output directory for generated configs")
    p.add_argument("--run", action="store_true", help="If set, invoke `train.py --config` for each variant")
    args = p.parse_args()

    base_cfg = None
    if args.base:
        base_path = Path(args.base)
        if not base_path.exists():
            print(f"Base config not found: {base_path}", file=sys.stderr)
            sys.exit(2)
        base_cfg = load_yaml(base_path)

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    out_dir = Path(args.out)

    created = build_variants(base_cfg, agents, encoders, out_dir)
    print(f"Created {len(created)} variant configs in {out_dir}")

    if args.run:
        for cfg in created:
            run_train(cfg)


if __name__ == "__main__":
    main()
