"""Circuit tests: the quantum circuit must reproduce the tree's probabilities exactly."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from qtr.circuits import (
    add_interference,
    array_to_bitstring,
    bitstring_to_array,
    build_tree_circuit,
    counts_to_samples,
    crash_angle,
    finalize,
    readout_calibration_circuits,
    two_qubit_gate_count,
)
from qtr.model import exact_marginals, joint_probability, random_tree_model


def statevector_pattern_probabilities(qc: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    """All 2^n patterns (asset order) and their probabilities from the exact quantum state.

    Statevector index k has qubit i's value in bit i of k (same little-endian convention).
    """
    n = qc.num_qubits
    probs = Statevector(qc).probabilities()
    idx = np.arange(2**n)
    patterns = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)
    return patterns, probs


# --- bit order -----------------------------------------------------------------

def test_bit_order_round_trip():
    rng = np.random.default_rng(0)
    for n in (1, 3, 8, 17):
        x = rng.integers(0, 2, n).astype(np.uint8)
        np.testing.assert_array_equal(bitstring_to_array(array_to_bitstring(x), n), x)


def test_bit_order_matches_real_qiskit_output():
    """Flip ONLY qubit 0 of 3 and check Qiskit reports '001', which decodes to asset 0 crashed."""
    qc = QuantumCircuit(3)
    qc.x(0)
    counts = StatevectorSampler(seed=1).run([finalize(qc)], shots=10).result()[0].data.meas.get_counts()
    assert counts == {"001": 10}
    np.testing.assert_array_equal(bitstring_to_array("001", 3), [1, 0, 0])


@pytest.mark.parametrize("bad", ["01", "0101", "01 1", "0a1"])
def test_bad_bitstrings_rejected(bad):
    with pytest.raises(ValueError):
        bitstring_to_array(bad, 3)


def test_counts_to_samples():
    samples = counts_to_samples({"01": 2, "10": 1}, 2)
    # "01" -> asset 0 crashed; "10" -> asset 1 crashed; sorted key order
    np.testing.assert_array_equal(samples, [[1, 0], [1, 0], [0, 1]])


# --- the circuit equals the tree --------------------------------------------------

def test_crash_angle_gives_right_probability():
    for p in (1e-4, 0.05, 0.5, 0.9):
        assert np.sin(crash_angle(p) / 2) ** 2 == pytest.approx(p)


@pytest.mark.parametrize(
    "n,seed", [(1, 0), (2, 1), (3, 2), (5, 3), (8, 4), (10, 5), (12, 6), (12, 7), (12, 8)]
)
def test_statevector_equals_tree_for_every_pattern(n, seed):
    """The M3 gate: every one of the 2^n crash patterns, to 1e-10."""
    model = random_tree_model(n, np.random.default_rng(seed))
    patterns, probs = statevector_pattern_probabilities(build_tree_circuit(model))
    np.testing.assert_allclose(probs, joint_probability(model, patterns), atol=1e-10)


def test_one_controlled_rotation_per_link():
    model = random_tree_model(9, np.random.default_rng(1))
    qc = build_tree_circuit(model)
    assert qc.count_ops().get("cry", 0) == model.n - 1
    assert two_qubit_gate_count(qc) == model.n - 1  # becomes 2 CZs each on hardware (checked in M5)


def test_sampler_shots_match_exact_marginals():
    model = random_tree_model(6, np.random.default_rng(3))
    shots = 20_000
    result = StatevectorSampler(seed=11).run([finalize(build_tree_circuit(model))], shots=shots).result()
    samples = counts_to_samples(result[0].data.meas.get_counts(), model.n)
    exact = exact_marginals(model)
    se = np.sqrt(exact * (1 - exact) / shots)
    assert samples.shape == (shots, model.n)
    assert np.all(np.abs(samples.mean(axis=0) - exact) <= 4 * se)


# --- interference and calibration ------------------------------------------------------

def test_interference_at_zero_changes_nothing():
    model = random_tree_model(6, np.random.default_rng(2))
    base = build_tree_circuit(model)
    _, p0 = statevector_pattern_probabilities(base)
    _, p1 = statevector_pattern_probabilities(add_interference(base, model.edges, 0.0))
    np.testing.assert_allclose(p0, p1, atol=1e-12)


def test_interference_changes_distribution_and_keeps_input():
    model = random_tree_model(6, np.random.default_rng(2))
    base = build_tree_circuit(model)
    n_ops = len(base.data)
    _, p0 = statevector_pattern_probabilities(base)
    _, p1 = statevector_pattern_probabilities(add_interference(base, model.edges, 0.2))
    assert np.abs(p0 - p1).max() > 1e-3
    assert len(base.data) == n_ops  # original untouched


def test_calibration_circuits_read_all_zeros_and_all_ones():
    zeros, ones = readout_calibration_circuits(4)
    result = StatevectorSampler(seed=0).run([zeros, ones], shots=50).result()
    assert result[0].data.meas.get_counts() == {"0000": 50}
    assert result[1].data.meas.get_counts() == {"1111": 50}
