"""
Command-line interface for evrp-rl.

Commands
--------
evrp-train   Train an agent (config file + override flags).
evrp-eval    Evaluate a saved checkpoint.
evrp-config  Generate a starter YAML config file.

Examples
--------
# Train with a config file:
evrp-train --config configs/experiment_config.yaml

# Quick override without editing the YAML:
evrp-train --config configs/experiment_config.yaml --agent sac --encoder gat --episodes 200

# Hyperparameter tweak:
evrp-train --config configs/experiment_config.yaml --lr 1e-3 --customers 15 --seed 0

# Resume from a checkpoint:
evrp-train --config configs/experiment_config.yaml --resume checkpoints/best_model.pt

# Evaluation only:
evrp-eval --config configs/experiment_config.yaml --checkpoint checkpoints/best_model.pt

# Generate a starter config:
evrp-config --agent sac --encoder gat --out my_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_config(config: dict, path: str) -> None:
    import yaml
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to {out}")


def _apply_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """
    Merge CLI overrides into the loaded config.

    Overrides follow the convention ``section.key = value`` so a caller
    can pass ``{"agent.type": "sac"}`` to set ``config["agent"]["type"]``.
    Simple (non-dotted) keys are set at the top level.
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def _build_env(config: dict, seed: int):
    from evrp_rl.env import EVRPEnvironment

    env_cfg = config.get("env", config.get("training", {}))
    return EVRPEnvironment(
        num_customers=env_cfg.get("num_customers", 10),
        num_chargers=env_cfg.get("num_chargers", 3),
        max_battery=env_cfg.get("battery_capacity", env_cfg.get("max_battery", 100.0)),
        max_cargo=env_cfg.get("cargo_capacity", env_cfg.get("max_cargo", 100.0)),
        time_limit=env_cfg.get("time_limit", 200),
        seed=seed,
    )


def _build_agent(config: dict, action_dim: int):
    import torch
    from evrp_rl.framework import AgentFactory
    return AgentFactory.create_from_dict(config, action_dim)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def _print_banner(title: str, fields: dict) -> None:
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    for k, v in fields.items():
        print(f"  {k:<22} {v}")
    print("-" * width)


# ---------------------------------------------------------------------------
# evrp-train
# ---------------------------------------------------------------------------

def _train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evrp-train",
        description="Train an EVRP RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument("--config", required=True, metavar="FILE",
                   help="Path to YAML experiment config.")

    # --- agent / encoder overrides ---
    g = p.add_argument_group("agent / encoder overrides")
    g.add_argument("--agent", choices=["a2c", "sac"], metavar="TYPE",
                   help="Override agent type  (a2c | sac).")
    g.add_argument("--encoder", choices=["gat", "mlp"], metavar="TYPE",
                   help="Override encoder type (gat | mlp).")
    g.add_argument("--embed-dim", type=int, metavar="N",
                   help="Override encoder embedding dimension.")
    g.add_argument("--lr", type=float, metavar="LR",
                   help="Override learning rate.")
    g.add_argument("--gamma", type=float, metavar="G",
                   help="Override discount factor γ.")
    g.add_argument("--hidden-dim", type=int, metavar="N",
                   help="Override network hidden dimension.")

    # --- environment overrides ---
    e = p.add_argument_group("environment overrides")
    e.add_argument("--customers", type=int, metavar="N",
                   help="Override number of customers.")
    e.add_argument("--chargers", type=int, metavar="N",
                   help="Override number of charging stations.")
    e.add_argument("--max-battery", type=float, metavar="B",
                   help="Override vehicle battery capacity.")
    e.add_argument("--max-cargo", type=float, metavar="C",
                   help="Override vehicle cargo capacity.")

    # --- run overrides ---
    r = p.add_argument_group("run overrides")
    r.add_argument("--episodes", type=int, metavar="N",
                   help="Override number of training episodes.")
    r.add_argument("--max-steps", type=int, metavar="N",
                   help="Override max steps per episode.")
    r.add_argument("--eval-freq", type=int, metavar="N",
                   help="Override evaluation frequency (episodes).")
    r.add_argument("--eval-episodes", type=int, metavar="N",
                   help="Override number of evaluation episodes.")
    r.add_argument("--save-freq", type=int, metavar="N",
                   help="Override checkpoint save frequency.")
    r.add_argument("--no-eval", action="store_true",
                   help="Disable evaluation during training.")

    # --- output ---
    o = p.add_argument_group("output")
    o.add_argument("--name", metavar="NAME",
                   help="Experiment name (used in log/checkpoint dirs).")
    o.add_argument("--log-dir", default="results", metavar="DIR",
                   help="Root directory for logs.")
    o.add_argument("--checkpoint-dir", default="checkpoints", metavar="DIR",
                   help="Root directory for checkpoints.")

    # --- system ---
    s = p.add_argument_group("system")
    s.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Compute device.")
    s.add_argument("--seed", type=int, default=42, help="Random seed.")
    s.add_argument("--resume", metavar="CKPT",
                   help="Path to checkpoint to resume training from.")

    return p


def train() -> None:
    """Entry point for ``evrp-train``."""
    import torch
    from evrp_rl.framework import ExperimentRunner

    args = _train_parser().parse_args()
    config = _load_config(args.config)

    # --- apply overrides ---
    agent_section = config.get("agent", {})
    is_dict_agent = isinstance(agent_section, dict)

    overrides: dict[str, Any] = {}

    # agent type
    if args.agent:
        if is_dict_agent:
            overrides["agent.type"] = args.agent
        else:
            config["agent"] = args.agent

    # encoder
    if args.encoder:
        if is_dict_agent:
            overrides["agent.encoder.type"] = args.encoder
        else:
            overrides["encoder.type"] = args.encoder
    if args.embed_dim:
        key = "agent.encoder.embed_dim" if is_dict_agent else "encoder.embed_dim"
        overrides[key] = args.embed_dim

    # hyperparams
    for flag, param in [("lr", "lr"), ("gamma", "gamma"), ("hidden_dim", "hidden_dim")]:
        val = getattr(args, flag.replace("-", "_"), None)
        if val is not None:
            key = f"agent.hyperparameters.{param}" if is_dict_agent else f"hyperparameters.{param}"
            overrides[key] = val

    # environment
    env_section = "env" if "env" in config else "training"
    for flag, param in [
        ("customers", "num_customers"),
        ("chargers",  "num_chargers"),
        ("max_battery", "max_battery"),
        ("max_cargo",   "max_cargo"),
    ]:
        val = getattr(args, flag.replace("-", "_"), None)
        if val is not None:
            overrides[f"{env_section}.{param}"] = val

    # run
    run_section = "run" if "run" in config else "training"
    for flag, param in [
        ("episodes",      "epochs"),
        ("max_steps",     "max_steps_per_episode"),
        ("eval_freq",     "eval_frequency"),
        ("eval_episodes", "num_eval_episodes"),
        ("save_freq",     "save_frequency"),
    ]:
        val = getattr(args, flag, None)
        if val is not None:
            overrides[f"{run_section}.{param}"] = val

    if args.name:
        overrides[f"{run_section}.name"] = args.name

    config = _apply_overrides(config, overrides)

    # --- seed + device ---
    _seed_everything(args.seed)
    device = torch.device(args.device)

    # --- build env + agent ---
    env   = _build_env(config, args.seed)
    agent = _build_agent(config, env.action_space.n)
    agent.to(device)

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    # --- resolve run config ---
    run_cfg     = config.get("run", config.get("training", {}))
    num_episodes = run_cfg.get("epochs", run_cfg.get("num_episodes", 1000))
    max_steps    = run_cfg.get("max_steps_per_episode", 200)
    eval_freq    = run_cfg.get("eval_frequency", 50)
    eval_eps     = run_cfg.get("num_eval_episodes", 5)
    save_freq    = run_cfg.get("save_frequency", 100)
    exp_name     = run_cfg.get("name", "experiment")

    _print_banner("evrp-train", {
        "config":       args.config,
        "agent":        config.get("agent", {}).get("type", config.get("agent", "?")),
        "encoder":      (config.get("agent", {}).get("encoder", {}) or config.get("encoder", {})).get("type", "?"),
        "customers":    env.num_customers,
        "chargers":     env.num_chargers,
        "episodes":     num_episodes,
        "device":       args.device,
        "seed":         args.seed,
        "log_dir":      args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
    })

    runner = ExperimentRunner(
        env, agent, config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    # patch runner settings from resolved config
    runner.num_epochs          = num_episodes
    runner.max_steps_per_episode = max_steps
    runner.eval_frequency      = 999999 if args.no_eval else eval_freq
    runner.num_eval_episodes   = eval_eps
    runner.save_frequency      = save_freq

    runner.train()


# ---------------------------------------------------------------------------
# evrp-eval
# ---------------------------------------------------------------------------

def _eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evrp-eval",
        description="Evaluate a saved EVRP agent checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     required=True, metavar="FILE",
                   help="Path to YAML experiment config used during training.")
    p.add_argument("--checkpoint", required=True, metavar="CKPT",
                   help="Path to saved model checkpoint (.pt file).")
    p.add_argument("--episodes",   type=int, default=100, metavar="N",
                   help="Number of evaluation episodes.")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed",       type=int, default=0, help="Random seed.")
    p.add_argument("--out",        metavar="DIR",
                   help="Directory to save eval_results.json (optional).")
    p.add_argument("--customers",  type=int, metavar="N",
                   help="Override num_customers (test on a different problem size).")
    p.add_argument("--chargers",   type=int, metavar="N",
                   help="Override num_chargers.")
    return p


def evaluate() -> None:
    """Entry point for ``evrp-eval``."""
    import json
    import torch

    args = _eval_parser().parse_args()
    config = _load_config(args.config)

    _seed_everything(args.seed)

    # apply size overrides
    overrides: dict[str, Any] = {}
    env_section = "env" if "env" in config else "training"
    if args.customers:
        overrides[f"{env_section}.num_customers"] = args.customers
    if args.chargers:
        overrides[f"{env_section}.num_chargers"] = args.chargers
    config = _apply_overrides(config, overrides)

    env   = _build_env(config, args.seed)
    agent = _build_agent(config, env.action_space.n)
    agent.to(torch.device(args.device))
    agent.load(args.checkpoint)

    _print_banner("evrp-eval", {
        "config":     args.config,
        "checkpoint": args.checkpoint,
        "episodes":   args.episodes,
        "customers":  env.num_customers,
        "chargers":   env.num_chargers,
        "device":     args.device,
        "seed":       args.seed,
    })

    import numpy as np
    rewards, lengths, route_lengths, charge_visits, successes = [], [], [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward, steps, route_len, charges = 0.0, 0, 0.0, 0

        while True:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            route_len += info.get("distance", 0.0)
            if info.get("node_type") == "charger":
                charges += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        route_lengths.append(route_len)
        charge_visits.append(charges)
        successes.append(1.0 if info.get("all_customers_visited") else 0.0)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"  [{ep + 1:>4}/{args.episodes}] reward {total_reward:.2f}")

    results = {
        "episodes":           args.episodes,
        "mean_reward":        float(np.mean(rewards)),
        "std_reward":         float(np.std(rewards)),
        "min_reward":         float(np.min(rewards)),
        "max_reward":         float(np.max(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_route_length":  float(np.mean(route_lengths)),
        "mean_charge_visits": float(np.mean(charge_visits)),
        "success_rate":       float(np.mean(successes)),
    }

    print("\nResults")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v}")

    if args.out:
        out_path = Path(args.out) / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


# ---------------------------------------------------------------------------
# evrp-config
# ---------------------------------------------------------------------------

def _config_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evrp-config",
        description="Generate a starter YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--agent",     default="sac", choices=["a2c", "sac"],
                   help="Agent algorithm.")
    p.add_argument("--encoder",   default="gat", choices=["gat", "mlp"],
                   help="Encoder architecture.")
    p.add_argument("--customers", type=int, default=10, metavar="N",
                   help="Number of customers.")
    p.add_argument("--chargers",  type=int, default=3,  metavar="N",
                   help="Number of charging stations.")
    p.add_argument("--episodes",  type=int, default=500, metavar="N",
                   help="Number of training episodes.")
    p.add_argument("--embed-dim", type=int, default=128, metavar="N",
                   help="Encoder embedding dimension.")
    p.add_argument("--lr",        type=float, default=3e-4, metavar="LR",
                   help="Learning rate.")
    p.add_argument("--device",    default="cpu", choices=["cpu", "cuda"],
                   help="Compute device.")
    p.add_argument("--out",       default="config.yaml", metavar="FILE",
                   help="Output path for the generated YAML.")
    return p


def generate_config() -> None:
    """Entry point for ``evrp-config``."""
    args = _config_parser().parse_args()

    encoder_cfg: dict[str, Any] = {"type": args.encoder, "embed_dim": args.embed_dim, "dropout": 0.1}
    if args.encoder == "gat":
        encoder_cfg.update({"num_layers": 3, "num_heads": 8})
    else:
        encoder_cfg.update({"num_layers": 3, "hidden_dim": args.embed_dim * 2})

    hyperparams: dict[str, Any] = {"lr": args.lr, "gamma": 0.99, "hidden_dim": 256}
    if args.agent == "sac":
        hyperparams.update({
            "tau": 0.005,
            "alpha": "auto",
            "buffer_size": 100000,
            "batch_size": 64,
            "learning_starts": 1000,
        })
    else:  # a2c
        hyperparams.update({
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
        })

    config = {
        "env": {
            "num_customers": args.customers,
            "num_chargers":  args.chargers,
            "battery_capacity": 100.0,
            "cargo_capacity":   50.0,
        },
        "agent": {
            "type":            args.agent,
            "encoder":         encoder_cfg,
            "hyperparameters": hyperparams,
        },
        "run": {
            "epochs":               args.episodes,
            "eval_frequency":       max(1, args.episodes // 10),
            "save_frequency":       max(1, args.episodes // 5),
            "max_steps_per_episode": 200,
            "num_eval_episodes":    10,
            "device":               args.device,
            "seed":                 42,
        },
    }

    _save_config(config, args.out)

    print(f"\nQuickstart:")
    print(f"  evrp-train --config {args.out}")
    print(f"  evrp-train --config {args.out} --episodes 100  # quick test")


# ---------------------------------------------------------------------------
# evrp-xai
# ---------------------------------------------------------------------------

def _xai_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evrp-xai",
        description="Run XAI (explainability) analysis on a trained EVRP agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-config", required=True, metavar="FILE",
                   help="Path to YAML config file (unified or agents_examples format).")
    p.add_argument("--example", metavar="NAME",
                   help="Name of the example to load from the 'examples' section of the config.")
    p.add_argument("--checkpoint", metavar="CKPT",
                   help="Path to a saved model checkpoint (optional — uses random weights if omitted).")
    p.add_argument("--out", default="results/xai", metavar="DIR",
                   help="Directory to write output files.")
    p.add_argument("--n-samples", type=int, default=50, metavar="N",
                   help="Perturbation samples per feature.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p


def xai() -> None:
    """Entry point for ``evrp-xai``."""
    import numpy as np
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")  # headless

    from evrp_rl.xai import perturbation_importance, plot_route_importance

    args = _xai_parser().parse_args()
    raw = _load_config(args.env_config)

    # Support both flat config and configs with an 'examples' section
    if args.example:
        examples = raw.get("examples", {})
        if args.example not in examples:
            sys.exit(f"Example '{args.example}' not found in {args.env_config}. "
                     f"Available: {list(examples.keys())}")
        config = examples[args.example]
    else:
        config = raw

    _seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build env
    env = _build_env(config, args.seed)
    agent = _build_agent(config, env.action_space.n)

    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — using randomly initialised weights.")

    _print_banner("evrp-xai", {
        "config":    args.env_config,
        "example":   args.example or "(none)",
        "out":       str(out_dir),
        "n_samples": args.n_samples,
        "customers": env.num_customers,
        "chargers":  env.num_chargers,
    })

    # --- run one greedy episode and collect the route ---
    obs, _ = env.reset(seed=args.seed)
    route: list[int] = [int(obs["current_node"])]

    while True:
        action, _ = agent.select_action(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        route.append(int(obs["current_node"]))
        if terminated or truncated:
            break

    print(f"Route length: {len(route)} nodes")

    # --- perturbation importance on the final observation ---
    feature_keys = ["node_coords", "distance_matrix", "node_demands", "node_types"]

    def _predict(state: dict) -> float:
        a, _ = agent.select_action(state, deterministic=True)
        # return the index as a scalar proxy; callers can swap in Q-value if needed
        return float(a)

    def _perturb(state: dict, key: str) -> dict:
        import copy
        s = copy.deepcopy(state)
        arr = s[key]
        noise = np.random.randn(*arr.shape) * float(np.std(arr) + 1e-8)
        s[key] = (arr + noise).astype(arr.dtype)
        return s

    importances = perturbation_importance(
        obs, feature_keys, _predict, _perturb, n_samples=args.n_samples
    )

    print("\nFeature importances (mean reward delta):")
    for feat, val in sorted(importances.items(), key=lambda x: -abs(x[1])):
        print(f"  {feat:<25} {val:+.4f}")

    # --- build NetworkX graph for plotting ---
    G = env.get_graph()

    # Map feature importances to nodes (use distance_matrix importance as node weight)
    node_imp: dict[int, float] = {n: importances.get("distance_matrix", 0.0) for n in G.nodes()}

    save_path = str(out_dir / "xai_route.png")
    plot_route_importance(G, route, node_importance=node_imp, save_path=save_path)
    print(f"\nRoute plot saved to {save_path}")

    # Also save a JSON summary
    import json
    summary = {
        "route": route,
        "feature_importances": importances,
    }
    json_path = out_dir / "xai_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")
