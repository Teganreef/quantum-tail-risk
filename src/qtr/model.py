"""The crash-dependency model: a Chow-Liu tree Bayesian network over crash indicators.

Plain-English version of what happens here:

1. Crash-buddy scores. For every pair of assets we measure mutual information (MI):
   how much knowing "A crashed today" tells you about "B crashed today". Zero means
   unrelated; bigger means they tend to crash together (or, rarely, apart).

2. Keep a tree. With n assets there are n(n-1)/2 pairs, far too many links to
   estimate from ~1,000 days. We keep the n-1 strongest links that connect every
   asset without forming a loop: a maximum spanning tree (Kruskal's algorithm).
   Chow & Liu (1968) proved this is the best tree-shaped approximation of the data.

3. Give it a direction. Pick a root (the asset most connected to its neighbors).
   Every other asset gets exactly one parent: its neighbor on the path to the root.

4. Two numbers per link. For each child c with parent a we estimate
   P(c crashes | a fine) and P(c crashes | a crashed). The root gets P(root crashes).
   That's the whole model: 1 + 2(n-1) numbers.

5. Answer key. Because it's a tree, we can compute some things EXACTLY with short
   loops instead of simulation: each asset's crash probability, every pair's
   probability of crashing together, and the full distribution of "how many assets
   crash on the same day". That exactness is what lets us grade quantum hardware later.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# The model object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeModel:
    """A tree Bayesian network over n binary crash indicators.

    Asset i is column i everywhere (and later, logical qubit i).

    cpt[i] = (P(i crashes | parent fine), P(i crashes | parent crashed)).
    For the root, which has no parent, both entries are P(root crashes), so every
    asset can be handled by the same code.
    """

    tickers: tuple[str, ...]
    root: int
    parent: tuple[int, ...]       # parent[i] = index of i's parent; -1 for the root
    bfs_order: tuple[int, ...]    # root first; every parent comes before its children
    cpt: np.ndarray               # shape (n, 2)
    mi: np.ndarray                # shape (n, n) mutual information matrix used to build the tree
    layout_mode: str              # "chow_liu", "logical" (degree-capped) or "native" (chip-shaped)
    data_hash: str

    @property
    def n(self) -> int:
        return len(self.tickers)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """(parent, child) pairs in BFS order."""
        return [(self.parent[c], c) for c in self.bfs_order if self.parent[c] >= 0]

    @property
    def p_root(self) -> float:
        return float(self.cpt[self.root, 0])

    def children(self, node: int) -> list[int]:
        return [c for c in self.bfs_order if self.parent[c] == node]

    def total_mi(self) -> float:
        """Sum of MI along the tree's links: how much dependence the tree captures."""
        return float(sum(self.mi[a, c] for a, c in self.edges))

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "root": self.root,
            "parent": list(self.parent),
            "bfs_order": list(self.bfs_order),
            "cpt": [[float(x) for x in row] for row in self.cpt],
            "mi": [[float(x) for x in row] for row in self.mi],
            "layout_mode": self.layout_mode,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TreeModel":
        d = json.loads(text)
        model = cls(
            tickers=tuple(d["tickers"]),
            root=int(d["root"]),
            parent=tuple(int(p) for p in d["parent"]),
            bfs_order=tuple(int(i) for i in d["bfs_order"]),
            cpt=np.array(d["cpt"], dtype=float),
            mi=np.array(d["mi"], dtype=float),
            layout_mode=d["layout_mode"],
            data_hash=d["data_hash"],
        )
        if "model_hash" in d and d["model_hash"] != model.model_hash():
            raise ValueError("Model JSON is corrupted: stored model_hash doesn't match its contents")
        return model

    def model_hash(self) -> str:
        """Fingerprint of everything that defines the probability distribution."""
        core = {
            "tickers": list(self.tickers),
            "parent": list(self.parent),
            "cpt": [[round(float(x), 12) for x in row] for row in self.cpt],
        }
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Mutual information
# ---------------------------------------------------------------------------


def pair_counts(crashes: np.ndarray) -> tuple[np.ndarray, ...]:
    """2x2 contingency counts for every pair at once.

    For assets i, j: n11 = days both crashed, n10 = i crashed but j didn't, etc.
    One matrix product (crashes.T @ crashes) gives n11 for all pairs simultaneously.
    """
    c = crashes.astype(np.int64)
    days = c.shape[0]
    n1 = c.sum(axis=0)                      # crash days per asset
    n11 = c.T @ c                           # both crashed
    n10 = n1[:, None] - n11                 # i crashed, j didn't
    n01 = n1[None, :] - n11                 # j crashed, i didn't
    n00 = days - n11 - n10 - n01            # neither crashed
    return n00, n01, n10, n11


def mutual_information(crashes: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Pairwise MI (in nats) between crash indicators, with Laplace smoothing.

    Laplace smoothing adds `alpha` imaginary days to each of the 4 cells of every
    2x2 table. Why: with ~50 crash days per asset, some pairs have 0 or 1 shared
    crash days, and a raw zero count would claim "these can NEVER crash together",
    which is overconfident. The pseudo-counts pull estimates gently toward "unrelated".

    MI = sum over the 4 cells of p(a,b) * log( p(a,b) / (p(a) p(b)) ).
    """
    cells = [x + alpha for x in pair_counts(crashes)]   # n00, n01, n10, n11, smoothed
    total = sum(cells)
    p00, p01, p10, p11 = (x / total for x in cells)
    p_i1 = p10 + p11      # P(i crashed)
    p_j1 = p01 + p11      # P(j crashed)
    p_i0, p_j0 = 1 - p_i1, 1 - p_j1
    mi = (
        p00 * np.log(p00 / (p_i0 * p_j0))
        + p01 * np.log(p01 / (p_i0 * p_j1))
        + p10 * np.log(p10 / (p_i1 * p_j0))
        + p11 * np.log(p11 / (p_i1 * p_j1))
    )
    np.fill_diagonal(mi, 0.0)  # an asset's link to itself isn't a candidate tree edge
    return mi


# ---------------------------------------------------------------------------
# 2. Maximum spanning tree (Kruskal), optionally with a degree cap
# ---------------------------------------------------------------------------


def max_spanning_tree(
    mi: np.ndarray, degree_cap: int | None = None
) -> tuple[list[tuple[int, int]], list[str]]:
    """Kruskal's algorithm: take links from strongest to weakest, skipping any that
    would close a loop. Returns undirected edges (i < j) and a log of fallbacks.

    With `degree_cap`, a link is also skipped if either asset already has that many
    neighbors (used for the "logical" layout, since chips connect each qubit to <= 3
    others). Greedy capping can occasionally strand a group; then the best remaining
    links are added ignoring the cap, and that's logged.

    Ties are broken by (i, j) index order, so the result is deterministic.
    """
    n = mi.shape[0]
    candidates = sorted(
        ((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda e: (-mi[e], e)
    )
    group = list(range(n))  # union-find: group[i] points toward i's group leader

    def leader(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    degree = [0] * n
    edges: list[tuple[int, int]] = []
    fallback_log: list[str] = []

    for enforce_cap in (True, False):
        for i, j in candidates:
            if len(edges) == n - 1:
                break
            if leader(i) == leader(j):
                continue  # would close a loop
            if enforce_cap and degree_cap is not None and (degree[i] >= degree_cap or degree[j] >= degree_cap):
                continue
            if not enforce_cap:
                fallback_log.append(f"degree cap relaxed to connect assets {i} and {j}")
            group[leader(i)] = leader(j)
            degree[i] += 1
            degree[j] += 1
            edges.append((i, j))
        if degree_cap is None:
            break
    return edges, fallback_log


def choose_root(n: int, edges: list[tuple[int, int]], mi: np.ndarray, tickers: tuple[str, ...]) -> int:
    """Root = the asset with the largest total MI to its tree neighbors (ties: alphabetical ticker)."""
    strength = np.zeros(n)
    for i, j in edges:
        strength[i] += mi[i, j]
        strength[j] += mi[i, j]
    return min(range(n), key=lambda i: (-round(strength[i], 15), tickers[i]))


def orient_tree(n: int, edges: list[tuple[int, int]], root: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Breadth-first search from the root: gives each asset its parent and a parent-first order."""
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    parent = [-2] * n          # -2 = not visited yet
    parent[root] = -1
    order: list[int] = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in sorted(neighbors[node]):
            if parent[nb] == -2:
                parent[nb] = node
                queue.append(nb)
    if len(order) != n:
        raise ValueError(f"Edges don't form a connected tree: reached {len(order)} of {n} assets")
    return tuple(parent), tuple(order)


# ---------------------------------------------------------------------------
# 3. Parameters
# ---------------------------------------------------------------------------


def fit_parameters(
    crashes: np.ndarray, parent: tuple[int, ...], alpha: float = 1.0, clip: float = 1e-4
) -> np.ndarray:
    """Estimate cpt[i] = (P(i=1 | parent=0), P(i=1 | parent=1)) by counting days.

    Example: if ETH crashed on 50 days and SOL crashed on 27 of those, then
    P(SOL=1 | ETH=1) = (27 + 1) / (50 + 2) with Laplace alpha = 1.

    Probabilities are clipped to [clip, 1 - clip] so no outcome is ever "impossible"
    (which would also make the quantum rotation angles degenerate).
    """
    c = crashes.astype(np.int64)
    days, n = c.shape
    cpt = np.empty((n, 2))
    for i in range(n):
        a = parent[i]
        if a < 0:
            p = (c[:, i].sum() + alpha) / (days + 2 * alpha)
            cpt[i] = (p, p)
            continue
        for s in (0, 1):
            on_days = c[:, a] == s
            denominator = on_days.sum() + 2 * alpha
            if denominator == 0:
                # Parent never took state s (possible only with alpha = 0): nothing to
                # learn from, so fall back to the child's overall crash rate.
                cpt[i, s] = c[:, i].mean()
            else:
                cpt[i, s] = (c[on_days, i].sum() + alpha) / denominator
    return np.clip(cpt, clip, 1 - clip)


def fit_tree_model(
    crashes: np.ndarray,
    tickers: tuple[str, ...],
    data_hash: str,
    alpha: float = 1.0,
    clip: float = 1e-4,
    degree_cap: int | None = None,
    edges: list[tuple[int, int]] | None = None,
    layout_mode: str | None = None,
) -> TreeModel:
    """Full fit: MI -> tree -> root -> orientation -> parameters.

    Pass `edges` to fit parameters on a tree chosen elsewhere (the chip-shaped
    "native" tree in layout.py). The MI matrix is still computed and stored, so we
    can measure how much dependence that tree captures versus the best possible tree.
    """
    mi = mutual_information(crashes, alpha)
    if edges is None:
        edges, _ = max_spanning_tree(mi, degree_cap)
        mode = layout_mode or ("logical" if degree_cap else "chow_liu")
    else:
        mode = layout_mode or "native"
    root = choose_root(len(tickers), edges, mi, tickers)
    parent, order = orient_tree(len(tickers), edges, root)
    return TreeModel(
        tickers=tuple(tickers),
        root=root,
        parent=parent,
        bfs_order=order,
        cpt=fit_parameters(crashes, parent, alpha, clip),
        mi=mi,
        layout_mode=mode,
        data_hash=data_hash,
    )


# ---------------------------------------------------------------------------
# 4. Sampling (the classical way to generate simulated days)
# ---------------------------------------------------------------------------


def sample(model: TreeModel, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Ancestral sampling: decide the root first, then each child given its parent.

    Exactly mirrors what the quantum circuit does, but with a random number generator
    instead of qubits. Returns a (shots, n) array of 0/1 in asset order.
    """
    x = np.zeros((shots, model.n), dtype=np.uint8)
    for node in model.bfs_order:
        a = model.parent[node]
        p = model.cpt[node, 0] if a < 0 else model.cpt[node][x[:, a]]
        x[:, node] = rng.random(shots) < p
    return x


def joint_probability(model: TreeModel, x: np.ndarray) -> np.ndarray:
    """Exact probability of each full crash pattern. x: (k, n) array -> (k,) probabilities.

    It's the product along the tree: P(root) * product of P(child | parent).
    Used by tests (brute force) and to check the quantum circuit bit-for-bit.
    """
    x = np.atleast_2d(x).astype(np.int64)
    prob = np.ones(x.shape[0])
    for node in range(model.n):
        a = model.parent[node]
        p1 = np.full(x.shape[0], model.cpt[node, 0]) if a < 0 else model.cpt[node][x[:, a]]
        prob *= np.where(x[:, node] == 1, p1, 1 - p1)
    return prob


# ---------------------------------------------------------------------------
# 5. Exact answers (the "answer key")
# ---------------------------------------------------------------------------


def exact_marginals(model: TreeModel) -> np.ndarray:
    """P(asset i crashes), top-down: P(c=1) = P(a=1) P(c=1|a=1) + P(a=0) P(c=1|a=0)."""
    m = np.zeros(model.n)
    for node in model.bfs_order:
        a = model.parent[node]
        m[node] = model.cpt[node, 0] if a < 0 else m[a] * model.cpt[node, 1] + (1 - m[a]) * model.cpt[node, 0]
    return m


def exact_pairwise(model: TreeModel) -> np.ndarray:
    """P(i=1 and j=1) for every pair; the diagonal holds P(i=1).

    How: for each asset i, pretend we KNOW it crashed, and push that knowledge
    outward along the tree to get P(j=1 | i=1) for every j. Then
    P(i=1, j=1) = P(i=1) * P(j=1 | i=1).

    Pushing knowledge from u to its neighbor k needs P(k=1 | u=s):
      - k is u's child: read it straight from the table (cpt[k]).
      - k is u's parent: flip it with Bayes' rule,
        P(k=1 | u=s) = P(u=s | k=1) P(k=1) / P(u=s).
    In a tree there's only one path between any two assets, so this is exact.
    Cost: n passes over n assets, so ~n^2 steps; trivial even at n = 96.
    """
    n = model.n
    m = exact_marginals(model)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, c in model.edges:
        neighbors[a].append(c)
        neighbors[c].append(a)

    def p_k1_given_u(k: int, u: int, s: int) -> float:
        if model.parent[k] == u:                     # k is u's child
            return float(model.cpt[k, s])
        # k is u's parent: Bayes' rule
        p_u_given_k1 = model.cpt[u, 1] if s == 1 else 1 - model.cpt[u, 1]
        p_u = m[u] if s == 1 else 1 - m[u]
        return float(p_u_given_k1 * m[k] / p_u)

    joint = np.zeros((n, n))
    for i in range(n):
        cond = np.zeros(n)      # cond[j] = P(j=1 | i=1)
        cond[i] = 1.0
        seen = {i}
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for k in neighbors[u]:
                if k in seen:
                    continue
                seen.add(k)
                cond[k] = cond[u] * p_k1_given_u(k, u, 1) + (1 - cond[u]) * p_k1_given_u(k, u, 0)
                queue.append(k)
        joint[i] = m[i] * cond
    return (joint + joint.T) / 2  # symmetric by math; averaging removes rounding noise


def exact_crash_count_distribution(model: TreeModel) -> np.ndarray:
    """P(exactly k assets crash on the same day), for k = 0..n. Exact, via the tree.

    Idea (dynamic programming, bottom-up): for each asset v and each state s of v,
    D[v][s] = distribution of the number of crashes inside v's subtree, given v = s.
      - Leaf: D[v][s] is "1 crash with certainty" if s = 1, else "0 crashes".
      - Internal node: start from the leaf answer, then for each child c, the child's
        subtree adds a random number of crashes with distribution
        sum_t P(c=t | v=s) * D[c][t]. Adding independent counts = convolving their
        distributions.
    At the root: P(count = k) = sum_s P(root=s) * D[root][s][k].
    """
    n = model.n
    dist: dict[int, np.ndarray] = {}
    for v in reversed(model.bfs_order):        # children are always finished before parents
        d = np.zeros((2, n + 1))
        d[0, 0] = 1.0
        d[1, 1] = 1.0
        for c in model.children(v):
            for s in (0, 1):
                p1 = model.cpt[c, s]
                child_contrib = (1 - p1) * dist[c][0] + p1 * dist[c][1]
                d[s] = np.convolve(d[s], child_contrib)[: n + 1]
            del dist[c]  # no longer needed
        dist[v] = d
    p_root = model.p_root
    return (1 - p_root) * dist[model.root][0] + p_root * dist[model.root][1]


# ---------------------------------------------------------------------------
# Test helper: random models (also used by circuit tests later)
# ---------------------------------------------------------------------------


def random_tree_model(n: int, rng: np.random.Generator, data_hash: str = "random") -> TreeModel:
    """A random tree with random crash probabilities, for testing."""
    tickers = tuple(f"A{i:02d}" for i in range(n))
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]  # attach each node to an earlier one
    root = int(rng.integers(0, n))
    parent, order = orient_tree(n, edges, root)
    cpt = rng.uniform(0.02, 0.9, size=(n, 2))
    cpt[root, 1] = cpt[root, 0]
    mi = np.zeros((n, n))
    return TreeModel(tickers, root, parent, order, cpt, mi, "random", data_hash)
