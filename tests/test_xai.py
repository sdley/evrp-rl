import numpy as np

from evrp_rl.xai import perturbation_importance, approximate_shapley


def test_perturbation_importance_simple():
    # state with numeric features
    state = {"battery": 100.0, "distance_to_charger": 10.0, "time_slack": 5.0}

    def predict_fn(s):
        # simple value: battery - distance
        return float(s["battery"] - s["distance_to_charger"])

    def perturb_fn(s, key):
        ns = dict(s)
        # random small perturbation
        ns[key] = ns[key] * (0.9 + 0.2 * np.random.random())
        return ns

    feats = ["battery", "distance_to_charger", "time_slack"]
    imp = perturbation_importance(state, feats, predict_fn, perturb_fn, n_samples=20)
    assert set(imp.keys()) == set(feats)
    for v in imp.values():
        assert isinstance(v, float)


def test_shapley_simple():
    state = {"a": 1.0, "b": 2.0, "c": 3.0}

    def value_fn(s, present):
        # value is sum of present features (or 0 if none)
        return float(sum(s[k] for k in present))

    feats = ["a", "b", "c"]
    shap = approximate_shapley(state, feats, value_fn, n_permutations=50)
    # shapley of additive function equals feature value
    # because value is sum of features, expected Shapley for a is 1.0, etc.
    assert pytest_approx(shap["a"], 1.0, tol=0.2)
    assert pytest_approx(shap["b"], 2.0, tol=0.2)
    assert pytest_approx(shap["c"], 3.0, tol=0.2)


def pytest_approx(x, y, tol=0.1):
    return abs(x - y) <= tol
