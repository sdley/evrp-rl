"""
Explainability (XAI) utilities for EVRP agents.

Provides perturbation-based feature importance, Monte-Carlo Shapley
approximations, and route visualisation helpers.
"""

from .attribution import (
    perturbation_importance,
    approximate_shapley,
    plot_route_importance,
    what_if_run,
)

__all__ = [
    "perturbation_importance",
    "approximate_shapley",
    "plot_route_importance",
    "what_if_run",
]
