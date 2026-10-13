"""Model tests: the exact tree math must match brute-force enumeration exactly."""

import itertools

import networkx as nx
import numpy as np
import pytest

from qtr.model import (
    TreeModel,
    exact_crash_count_distribution,
    exact_marginals,
    exact_pairwise,
    fit_parameters,
    fit_tree_model,
    joint_probability,
    max_spanning_tree,
    mutual_information,
    orient_tree,
    random_tree_model,
    sample,
)


def all_patterns(n: int) -> np.ndarray:
    """Every possible 0/1 crash pattern for n assets: shape (2^n, n)."""
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.uint8)


def brute_force(model: TreeModel):
    x = all_patterns(model.n)
    p = joint_probability(model, x)
    return x, p


# --- the answer key vs brute force --------------------------------------------

@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (3, 2), (5, 3), (8, 4), (10, 5), (12, 6), (12, 7)])
def test_joint_sums_to_one(n, seed):
    _, p = brute_force(random_tree_model(n, np.random.default_rng(seed)))
    assert p.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (4, 2), (7, 3), (10, 4), (12, 5), (12, 6)])
def test_crash_count_dp_matches_brute_force(n, seed):
    model = random_tree_model(n, np.random.default_rng(seed))
    x, p = brute_force(model)
    expected = np.bincount(x.sum(axis=1), weights=p, minlength=n + 1)
    np.testing.assert_allclose(exact_crash_count_distribution(model), expected, atol=1e-12)


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (3, 2), (6, 3), (9, 4), (10, 5)])
def test_marginals_and_pairs_match_brute_force(n, seed):
    model = random_tree_model(n, np.random.default_rng(seed))
    x, p = brute_force(model)
    expected_marg = x.T.astype(float) @ p
    expected_pairs = (x.T * p).astype(float) @ x
    np.testing.assert_allclose(exact_marginals(model), expected_marg, atol=1e-12)
    np.testing.assert_allclose(exact_pairwise(model), expected_pairs, atol=1e-12)


def test_sampler_matches_exact_marginals_within_3_standard_errors():
    model = random_tree_model(12, np.random.default_rng(42))
    shots = 1_000_000
    draws = sample(model, shots, np.random.default_rng(7))
    exact = exact_marginals(model)
    se = np.sqrt(exact * (1 - exact) / shots)
    assert np.all(np.abs(draws.mean(axis=0) - exact) <= 3 * se)


def test_sampler_is_reproducible_with_seed():
    model = random_tree_model(6, np.random.default_rng(0))
    a = sample(model, 1000, np.random.default_rng(1))
    b = sample(model, 1000, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


# --- mutual information ---------------------------------------------------------

def test_mi_zero_for_independent_and_large_for_identical():
    rng = np.random.default_rng(0)
    a = (rng.random(100_000) < 0.05).astype(np.uint8)
    b = (rng.random(100_000) < 0.05).astype(np.uint8)
    mi = mutual_information(np.column_stack([a, b, a]), alpha=1.0)
    assert mi[0, 1] < 1e-3                      # independent: ~0
    p = a.mean()
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert mi[0, 2] == pytest.approx(entropy, rel=1e-3)  # identical: MI = the asset's own entropy
    assert np.all(np.diag(mi) == 0)
    np.testing.assert_allclose(mi, mi.T)


def test_mi_hand_computed_with_laplace():
    # 4 days: both crash on day 0, only first on day 1, neither on days 2-3.
    c = np.array([[1, 1], [1, 0], [0, 0], [0, 0]], dtype=np.uint8)
    # Smoothed counts n00=3, n01=1, n10=2, n11=2 out of 8
    p = {"00": 3 / 8, "01": 1 / 8, "10": 2 / 8, "11": 2 / 8}
    pi1, pj1 = 4 / 8, 3 / 8
    expected = (
        p["00"] * np.log(p["00"] / ((1 - pi1) * (1 - pj1)))
        + p["01"] * np.log(p["01"] / ((1 - pi1) * pj1))
        + p["10"] * np.log(p["10"] / (pi1 * (1 - pj1)))
        + p["11"] * np.log(p["11"] / (pi1 * pj1))
    )
    assert mutual_information(c, alpha=1.0)[0, 1] == pytest.approx(expected)


# --- tree building -----------------------------------------------------------------

def test_kruskal_matches_networkx_total_weight():
    rng = np.random.default_rng(3)
    mi = rng.random((15, 15))
    mi = (mi + mi.T) / 2
    np.fill_diagonal(mi, 0)
    edges, log = max_spanning_tree(mi)
    graph = nx.Graph()
    graph.add_weighted_edges_from((i, j, mi[i, j]) for i in range(15) for j in range(i + 1, 15))
    nx_total = nx.maximum_spanning_tree(graph).size(weight="weight")
    assert sum(mi[e] for e in edges) == pytest.approx(nx_total)
    assert log == []


def test_degree_cap_respected_and_still_a_tree():
    rng = np.random.default_rng(4)
    mi = rng.random((20, 20))
    mi[0, :] += 5          # one asset that would be a hub with 19 neighbors
    mi = (mi + mi.T) / 2
    np.fill_diagonal(mi, 0)
    edges, _ = max_spanning_tree(mi, degree_cap=3)
    assert len(edges) == 19
    degrees = np.bincount(np.array(edges).ravel(), minlength=20)
    assert degrees.max() <= 3
    assert nx.is_tree(nx.Graph(edges))


def test_orientation_parents_before_children():
    model = random_tree_model(15, np.random.default_rng(5))
    position = {node: k for k, node in enumerate(model.bfs_order)}
    assert model.parent[model.root] == -1 and model.bfs_order[0] == model.root
    for a, c in model.edges:
        assert position[a] < position[c]


def test_disconnected_edges_rejected():
    with pytest.raises(ValueError, match="connected tree"):
        orient_tree(4, [(0, 1), (2, 3)], root=0)


# --- parameters: smoothing and clipping ----------------------------------------------

def test_laplace_parameters_hand_computed():
    # parent = asset 0, child = asset 1. 10 days.
    parent_col = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    child_col = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    c = np.column_stack([parent_col, child_col]).astype(np.uint8)
    cpt = fit_parameters(c, parent=(-1, 0), alpha=1.0, clip=1e-4)
    assert cpt[0, 0] == pytest.approx((4 + 1) / (10 + 2))   # root P(crash)
    assert cpt[1, 1] == pytest.approx((3 + 1) / (4 + 2))    # child | parent crashed
    assert cpt[1, 0] == pytest.approx((1 + 1) / (6 + 2))    # child | parent fine


def test_probabilities_are_clipped():
    c = np.zeros((10, 2), dtype=np.uint8)
    cpt = fit_parameters(c, parent=(-1, 0), alpha=0.0, clip=1e-4)  # no smoothing -> raw zeros
    assert cpt.min() == pytest.approx(1e-4)


# --- full fit and serialization -----------------------------------------------------

def test_fit_recovers_planted_tree():
    """Generate data from a known tree; Chow-Liu should find that tree's links."""
    true = random_tree_model(8, np.random.default_rng(11))
    true = TreeModel(true.tickers, true.root, true.parent, true.bfs_order,
                     np.array([[0.03, 0.85]] * 8), true.mi, "random", "x")
    true.cpt[true.root] = (0.1, 0.1)
    data = sample(true, 200_000, np.random.default_rng(12))
    fitted = fit_tree_model(data, true.tickers, data_hash="x")
    as_set = lambda m: {frozenset(e) for e in m.edges}
    assert as_set(fitted) == as_set(true)


def test_json_round_trip_and_tamper_check():
    model = fit_tree_model(
        sample(random_tree_model(6, np.random.default_rng(1)), 5000, np.random.default_rng(2)),
        tuple("ABCDEF"), data_hash="h",
    )
    again = TreeModel.from_json(model.to_json())
    assert again.model_hash() == model.model_hash()
    np.testing.assert_allclose(again.cpt, model.cpt)
    tampered = model.to_json().replace(model.tickers[0], "ZZZ", 1)
    with pytest.raises(ValueError, match="corrupted"):
        TreeModel.from_json(tampered)
