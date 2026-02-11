**XAI Module**

This document describes the `src/xai/attribution.py` utilities for
interpreting agent decisions in EVRP.

Functions

- `perturbation_importance(state, feature_keys, predict_fn, perturb_fn, n_samples)`: Monte-Carlo perturbation importance per feature.
- `approximate_shapley(state, feature_keys, value_fn, n_permutations)`: Monte-Carlo Shapley value approximation.
- `plot_route_importance(G, route, node_importance, edge_importance)`: Visualize importance on a graph with route overlay.
- `what_if_run(env_create_fn, agent_action_fn, scenario, modifier)`: Rerun policy under a modified scenario (e.g., changed battery or chargers).

Usage notes

- The attribution functions operate on lightweight callables and are decoupled from agent implementations. Provide `predict_fn` or `value_fn` that returns a scalar reward/logit from a state or masked state.
- The `perturbation_importance` utility requires a `perturb_fn(state, key)` that returns a perturbed state copy.
- Visualizations use NetworkX layouts or node `pos` attributes.

Examples

- See `examples/case_study.ipynb` for a full example (route plotting, heatmaps, what-if scenarios).
