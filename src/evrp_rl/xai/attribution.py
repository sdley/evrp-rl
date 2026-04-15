"""XAI attribution helpers for EVRP agents.

This module provides lightweight perturbation-based and Monte-Carlo
Shapley approximations for feature importance, plus visualization helpers
to plot route/node/edge importances using NetworkX/Matplotlib.

Design notes:
- Functions are written to accept small callables (`predict_fn`, `value_fn`)
  so they are decoupled from concrete agent/env types and easy to test.
- Shapley is approximated via random permutations (Monte Carlo).
"""
from typing import Callable, Dict, Iterable, List, Tuple, Any, Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm


def perturbation_importance(
    state: Dict[str, Any],
    feature_keys: Iterable[str],
    predict_fn: Callable[[Dict[str, Any]], float],
    perturb_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    n_samples: int = 50,
) -> Dict[str, float]:
    """Estimate importance by perturbing each feature and measuring delta.

    Args:
        state: Original state dictionary.
        feature_keys: Iterable of keys in `state` to perturb.
        predict_fn: Function(state) -> scalar value (reward or logit).
        perturb_fn: Function(state, key) -> new_state where key is perturbed.
        n_samples: Number of random perturbation samples per feature.

    Returns:
        Mapping feature -> mean delta (predict_fn(original) - predict_fn(perturbed)).
    """
    base_value = predict_fn(state)
    importances: Dict[str, float] = {}

    for key in feature_keys:
        deltas = []
        for _ in range(n_samples):
            pert_state = perturb_fn(state, key)
            val = predict_fn(pert_state)
            deltas.append(base_value - val)
        importances[key] = float(np.mean(deltas))

    return importances


def approximate_shapley(
    state: Dict[str, Any],
    feature_keys: List[str],
    value_fn: Callable[[Dict[str, Any], Iterable[str]], float],
    n_permutations: int = 100,
) -> Dict[str, float]:
    """Approximate Shapley values by random permutations.

    Args:
        state: Original state dict.
        feature_keys: Ordered list of feature keys to consider.
        value_fn: Function(state, present_features) -> scalar value. The
            `present_features` iterable contains the feature keys that are
            kept (others are set to baseline by the implementation of
            `value_fn` or by the caller via copying the state).
        n_permutations: Number of random permutations to average over.

    Returns:
        Mapping feature -> estimated Shapley value.
    """
    m = len(feature_keys)
    shap = {k: 0.0 for k in feature_keys}

    for _ in range(n_permutations):
        perm = list(np.random.permutation(feature_keys))
        prev_value = value_fn(state, [])
        present: List[str] = []
        for f in perm:
            present.append(f)
            cur_value = value_fn(state, present)
            marginal = cur_value - prev_value
            shap[f] += marginal
            prev_value = cur_value

    for k in shap:
        shap[k] /= float(n_permutations)

    return shap


def plot_route_importance(
    G: nx.Graph,
    route: List[int],
    node_importance: Optional[Dict[int, float]] = None,
    edge_importance: Optional[Dict[Tuple[int, int], float]] = None,
    ax: Optional[plt.Axes] = None,
    cmap_name: str = "viridis",
    node_size: int = 200,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot graph with route highlighted and importance heatmap on nodes/edges.

    Args:
        G: NetworkX graph with node positions in `pos` attribute or as `pos` arg.
        route: Ordered list of node indices visited by the agent.
        node_importance: Mapping node -> importance (higher means more important).
        edge_importance: Mapping (u,v) -> importance.
        ax: Optional Matplotlib axes to draw on.
        cmap_name: Name of Matplotlib colormap.
        node_size: Base node size.
        save_path: If provided, save figure to this path.

    Returns:
        The Matplotlib Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Positions: allow pos stored on graph nodes
    if all("pos" in G.nodes[n] for n in G.nodes()):
        pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node coloring
    if node_importance is not None:
        vals = np.array([node_importance.get(n, 0.0) for n in G.nodes()])
        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
        cmap = cm.get_cmap(cmap_name)
        node_colors = [cmap(norm(node_importance.get(n, 0.0))) for n in G.nodes()]
    else:
        node_colors = "lightblue"

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, ax=ax)

    # Edge coloring
    if edge_importance is not None:
        edges = list(G.edges())
        e_vals = np.array([edge_importance.get((u, v), edge_importance.get((v, u), 0.0)) for u, v in edges])
        if len(e_vals) > 0:
            norm_e = plt.Normalize(vmin=e_vals.min(), vmax=e_vals.max())
            cmap_e = cm.get_cmap(cmap_name)
            edge_colors = [cmap_e(norm_e(edge_importance.get((u, v), edge_importance.get((v, u), 0.0)))) for u, v in edges]
        else:
            edge_colors = "gray"
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2.0, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)

    # Draw route with thicker line
    if route and len(route) > 1:
        route_edges = list(zip(route[:-1], route[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color="red", width=3.0, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_axis_off()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def what_if_run(
    env_create_fn: Callable[[Dict[str, Any]], Any],
    agent_action_fn: Callable[[Any, Dict[str, Any]], int],
    scenario: Dict[str, Any],
    modifier: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Run policy under a modified scenario and return route and metrics.

    Args:
        env_create_fn: Function(env_config) -> env instance
        agent_action_fn: Function(env, obs) -> action
        scenario: Base env configuration dict
        modifier: Function(scenario) -> modified_scenario
        max_steps: Max steps to run

    Returns:
        Dict with `route`, `total_reward`, `steps`, and `info`.
    """
    modified = modifier(dict(scenario))
    env = env_create_fn(modified)
    obs, _ = env.reset()
    route = [env.current_node]
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent_action_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        route.append(env.current_node)
        if terminated or truncated:
            break

    return {"route": route, "total_reward": total_reward, "steps": len(route), "info": info}
