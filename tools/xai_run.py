#!/usr/bin/env python3
"""CLI to run XAI attributions and export visualizations.

Usage (example):
  python tools/xai_run.py --env-config configs/agents_examples.yaml --example a2c_example --out results/xai

This script focuses on reward-based perturbation importance for a chosen
next-action returned by a policy. It's intentionally lightweight and
works with `EnvFactory` from this repo.
"""
import argparse
import os
from pathlib import Path
import yaml

from src.framework import EnvFactory
from src.xai.attribution import perturbation_importance, plot_route_importance, what_if_run


def load_example(config_path: Path, example_name: str):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    examples = data.get("examples", {})
    return examples.get(example_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", required=True, help="Path to examples YAML containing env scenarios")
    p.add_argument("--example", required=True, help="Example key in the YAML to run")
    p.add_argument("--out", default="results/xai", help="Output directory for figures")
    args = p.parse_args()

    cfg = load_example(Path(args.env_config), args.example)
    if cfg is None:
        raise SystemExit(f"Example {args.example} not found in {args.env_config}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvFactory.create(cfg["env"]) if "env" in cfg else EnvFactory.create(cfg)
    obs, _ = env.reset()

    # Simple policy: use EnvFactory/placeholder random policy (user can replace)
    valid = list(i for i, v in enumerate(obs["valid_actions_mask"]) if v)
    if len(valid) == 0:
        raise SystemExit("No valid actions in initial observation")
    next_action = valid[0]

    # Build a minimal state dict for features
    state = {
        "current_battery": float(env.current_battery),
        "distance_to_nearest_charger": float(min(env.distance_matrix[env.current_node, env.charger_start_idx:env.charger_end_idx+1])) if env.num_chargers>0 else 0.0,
        "distance_to_nearest_customer": float(min([env.distance_matrix[env.current_node, c] for c in range(env.customer_start_idx, env.customer_end_idx+1) if not env.visited_mask[c]])) if env.num_customers>0 else 0.0,
    }

    def predict_fn(sdict):
        # apply sdict to env temporarily and compute reward for chosen next_action
        old_batt = env.current_battery
        try:
            env.current_battery = float(sdict.get("current_battery", old_batt))
            return float(env._compute_reward(next_action))
        finally:
            env.current_battery = old_batt

    def perturb_fn(sdict, key):
        ns = dict(sdict)
        # Simple multiplicative perturbation
        ns[key] = ns.get(key, 0.0) * (0.5 + 1.0 * os.urandom(1)[0] / 255.0)
        return ns

    feats = list(state.keys())
    importances = perturbation_importance(state, feats, predict_fn, perturb_fn, n_samples=30)

    # Build a tiny graph for visualization using env node positions if available
    import networkx as nx
    G = nx.Graph()
    n = env.num_nodes
    for i in range(n):
        G.add_node(i, pos=tuple(env.node_coords[i]))
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j)

    # Map node importance: set battery importance on depot node as example
    node_imp = {i: (importances.get("current_battery", 0.0) if i == env.depot_idx else 0.0) for i in G.nodes()}
    ax = plot_route_importance(G, [env.current_node], node_importance=node_imp, save_path=str(out_dir / "xai_route.png"))
    print("Saved XAI figure to", out_dir / "xai_route.png")


if __name__ == "__main__":
    main()
