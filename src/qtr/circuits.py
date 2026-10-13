"""Turn the crash tree into a quantum circuit, and decode measured bitstrings.

The core trick (spec section 3.3), in plain words:

- One qubit per asset. Measuring a qubit gives 0 ("fine") or 1 ("crashed").
- A qubit starts at 0. The RY(theta) gate rotates it so that measuring gives 1 with
  probability sin^2(theta / 2). To get probability p, use theta = 2 * arcsin(sqrt(p)).
  Think of tilting a dial: no tilt = always 0, a quarter turn = 50/50, half turn = always 1.
- For a child c with parent a, we want two different tilts: theta_0 when the parent
  is 0 and theta_1 when the parent is 1. So:
      1. RY(theta_0) on c                       (the tilt for "parent fine")
      2. controlled-RY(theta_1 - theta_0) from a to c
         (the EXTRA tilt, applied only if the parent qubit is 1)
  Rotations about the same axis add up, so c ends at theta_0 or theta_1 depending
  on its parent. That builds "c follows a" directly into the physics.
- Do this root first, then down the tree in BFS order (parents before children).
  Measuring every qubit then produces crash patterns with EXACTLY the tree's
  probabilities. No training, no trial-and-error: the angles come straight from the model.

Bit order warning: Qiskit writes measured bitstrings "little-endian": the RIGHTMOST
character is qubit 0. So "001" means qubit 0 = 1, qubits 1 and 2 = 0.
`bitstring_to_array` is the ONE place in this project that decodes that.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from qtr.model import TreeModel


def crash_angle(p: float | np.ndarray) -> float | np.ndarray:
    """RY angle that makes a qubit read 1 with probability p: theta = 2 arcsin(sqrt(p))."""
    return 2 * np.arcsin(np.sqrt(p))


# ---------------------------------------------------------------------------
# Building circuits
# ---------------------------------------------------------------------------


def build_tree_circuit(model: TreeModel) -> QuantumCircuit:
    """The UNMEASURED circuit encoding the tree. Qubit i = asset i.

    Unmeasured on purpose (DECISIONS D10): the interference layer must go before
    measurement, and tests read exact probabilities from the unmeasured state.
    Call `finalize` to add measurements.
    """
    qc = QuantumCircuit(model.n, name=f"crash_tree_{model.n}")
    for node in model.bfs_order:
        a = model.parent[node]
        theta_0 = float(crash_angle(model.cpt[node, 0]))
        if a < 0:
            qc.ry(theta_0, node)              # root: just tilt to P(root crashes)
            continue
        theta_1 = float(crash_angle(model.cpt[node, 1]))
        qc.ry(theta_0, node)                  # tilt for "parent fine"
        qc.cry(theta_1 - theta_0, a, node)    # extra tilt only when parent = 1
    return qc


def add_interference(qc: QuantumCircuit, edges: list[tuple[int, int]], lam: float) -> QuantumCircuit:
    """EXPLORATORY (spec 3.7, no financial meaning): one layer of quantum interference.

    RZZ(2 lam) on every tree link adds a phase that depends on whether the two
    linked qubits agree; RX(2 lam) on every qubit then mixes 0 and 1 so those
    phases change the measured probabilities. At lam = 0 both gates do nothing.
    Returns a new circuit; the input is not modified.
    """
    out = qc.copy(name=f"{qc.name}_lam{lam:g}")
    for a, c in edges:
        out.rzz(2 * lam, a, c)
    for q in range(qc.num_qubits):
        out.rx(2 * lam, q)
    return out


def finalize(qc: QuantumCircuit) -> QuantumCircuit:
    """Add measurements of every qubit into a classical register called 'meas'.

    This happens on the LOGICAL circuit, before transpiling to a chip, so bit i of
    the result is always asset i, no matter which physical qubit asset i lands on.
    """
    out = qc.copy()
    out.measure_all()
    return out


def readout_calibration_circuits(n: int) -> list[QuantumCircuit]:
    """Two circuits for measuring readout error: all qubits prepared 0, and all prepared 1.

    On perfect hardware the first always reads 00...0 and the second 11...1. Any
    other outcome is the machine misreading qubits, which is exactly what we need
    to estimate (and later correct) in post-processing.
    """
    zeros = QuantumCircuit(n, name=f"cal_zeros_{n}")
    ones = QuantumCircuit(n, name=f"cal_ones_{n}")
    ones.x(range(n))
    return [finalize(zeros), finalize(ones)]


# ---------------------------------------------------------------------------
# Decoding results: the ONLY place bit order is handled
# ---------------------------------------------------------------------------


def bitstring_to_array(key: str, n: int) -> np.ndarray:
    """Qiskit count key -> 0/1 array in asset order. key[-1 - i] is qubit (asset) i."""
    if len(key) != n or set(key) - {"0", "1"}:
        raise ValueError(
            f"Expected a {n}-character bitstring of 0s and 1s, got {key!r}. "
            "(A space would mean multiple classical registers; we always measure into one.)"
        )
    return np.array([int(ch) for ch in reversed(key)], dtype=np.uint8)


def array_to_bitstring(x: np.ndarray) -> str:
    """Inverse of bitstring_to_array: asset-order 0/1 array -> Qiskit-style key."""
    return "".join(str(int(b)) for b in reversed(np.asarray(x).ravel()))


def counts_to_samples(counts: dict[str, int], n: int) -> np.ndarray:
    """Expand {bitstring: count} into a (shots, n) array of 0/1 rows in asset order.

    Rows are grouped by bitstring in sorted-key order, so the result is
    deterministic. Shot order carries no information (Qiskit doesn't keep it in counts).
    """
    keys = sorted(counts)
    if not keys:
        return np.zeros((0, n), dtype=np.uint8)
    patterns = np.stack([bitstring_to_array(k, n) for k in keys])
    return np.repeat(patterns, [counts[k] for k in keys], axis=0)


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    """Number of gates acting on exactly two qubits (the noisy, expensive ones on hardware)."""
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
