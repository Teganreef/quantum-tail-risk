# BUILD SPEC: Quantum Tail-Risk Observatory

> Canonical spec, saved verbatim from the owner on 2026-10-12. The build plan lives in
> `docs/GAMEPLAN.md`; judgment calls where this spec is silent live in `docs/DECISIONS.md`.
> If this file and the gameplan disagree, this file wins unless DECISIONS.md records why.

You are a senior quantitative developer and quantum software engineer. You are building a complete, working project from this spec in one pass. Read the whole document before writing any code. Every section exists for a reason: the "why" explains intent so you can make good judgment calls where the spec is silent. Where the spec is explicit, follow it exactly. Where it is silent, choose the simplest correct option and note the choice in `docs/DECISIONS.md`.

The owner is a 20-year-old CS junior (AI minor), intermediate Python, active investor, learning quantum computing. He will run this on his own laptop and must be able to understand every file. Code clarity and honest results matter more than cleverness.

---

## 1. The idea in one paragraph

Markets don't crash one asset at a time. When tech sells off, a bunch of names drop together. This project learns, from real price history, *which assets tend to crash together*, encodes that crash-dependency network into a quantum circuit (one qubit per asset), and runs it on a real IBM quantum processor at scales up to ~100 qubits. Every measurement shot is one simulated trading day: a bitstring saying which assets crashed. Those scenarios are turned into portfolio losses so we can measure tail risk (worst-case days, odds of many assets crashing at once) for the owner's real portfolio and comparison portfolios. Because the model is built on a tree structure, the *ideal* answer stays exactly computable on a laptop even at 100 qubits. That lets us grade the real quantum hardware precisely at a scale where most quantum demos can't verify anything, and answer the research question: **does real quantum hardware make portfolio crash risk look safer or scarier than it really is, and how does that change as the circuit grows?**

---

## 2. Goals (ranked) and the honesty contract

### Goals, in priority order
1. **Resume-grade engineering and research rigor.** Reproducible, tested, benchmarked against fair classical baselines, every number traceable to raw data.
2. **Use the quantum hardware where it matters.** Real IBM QPU runs at large scale (50 to 100+ qubits), plus an exploratory experiment whose ideal output cannot be brute-force simulated.
3. **Useful to the owner.** Stress-test his actual portfolio, and reuse the saved scenarios all month without spending QPU time.
4. **Visual.** Screenshot-ready charts and one interactive dashboard.
5. **Teaches.** Plain-English documentation of how every piece works.

### Honesty contract (enforce this in README, reports, chart titles, and code comments)
Allowed claims:
- "Encoded a historical crash-dependency model into an N-qubit circuit and ran it on IBM quantum hardware."
- "Measured how real-device noise distorts tail-risk metrics versus an exact classical ground truth, across circuit sizes."
- "The interference experiment produces distributions beyond brute-force statevector simulation at N qubits; we tested whether tensor-network simulation could still reproduce them and report the result."

Forbidden claims (never write these anywhere):
- Quantum advantage, quantum speedup, or "a classical computer can't do this" for the calibrated risk model. (It can. That is the point: it is our ground truth.)
- "Quantum predicts crashes," "quantum finds hidden risk," "more accurate risk model."
- Anything that reads as a trade signal or investment advice. Every report and dashboard carries the line: *Research and education only. Not investment advice. Backward-looking data.*

If a result is ugly (hardware output is mostly noise at large N), report it plainly. A clean negative result is a valid outcome.

---

## 3. Core concepts you must understand before coding

### 3.1 Crash events
For each asset, a **crash day** is a day whose return is at or below that asset's own `q`-quantile over the fit window (default `q = 0.05`, so each asset crashes on about 5% of days by construction). This turns each asset's return series into a 0/1 series. Using per-asset quantiles means a "crash" for gold and a "crash" for SOL are equally rare events for each asset, which makes dependence comparable across assets.

### 3.2 Dependence as a tree (Chow-Liu)
Full joint dependence across 100 assets is impossible to estimate from ~1,000 days of data. The classic, principled approximation is the **Chow-Liu tree**: compute the mutual information (MI) between every pair of assets' crash indicators, then take the maximum-weight spanning tree. That tree is provably the best tree-shaped approximation (in KL divergence) of the true joint distribution. It is a **Bayesian network**: pick a root; every other asset has exactly one parent; the model is defined by `P(root crashes)` and, for each child, `P(child crashes | parent didn't)` and `P(child crashes | parent did)`.

### 3.3 Encoding the tree into qubits (exactly, no training)
One qubit per asset. A qubit measured as 1 means "crashed today."
- Root: apply `RY(theta_root)` with `theta = 2*arcsin(sqrt(p))`. Measuring gives 1 with probability `p`.
- Each child `c` with parent `a`, in parent-before-child (BFS) order: apply `RY(theta_c0)` on `c`, then a controlled `RY(theta_c1 - theta_c0)` with control `a`, target `c`. Because RY rotations about the same axis add, the child ends up rotated by `theta_c0` when the parent is 0 and `theta_c1` when the parent is 1. So `P(c=1 | a=0) = sin^2(theta_c0/2)` and `P(c=1 | a=1) = sin^2(theta_c1/2)`.
- Measuring all qubits yields bitstrings distributed **exactly** as the Bayesian network. Each edge costs one controlled-RY = 2 two-qubit gates. No variational training, no QPU time wasted on optimization loops.

### 3.4 Why the ideal answer stays computable at 100 qubits
A tree-structured Bayesian network can be sampled classically (ancestral sampling) and some quantities computed exactly (e.g., the distribution of the number of simultaneous crashes, via dynamic programming over the tree). So at any N we have an exact or high-precision ground truth. That is what lets us measure hardware error at 100 qubits, where generic quantum experiments can't check their answers. This is a feature. Do not hide it; explain it.

### 3.5 Three sources of error (the analysis framework)
Every comparison in this project decomposes into three layers. Keep them separate everywhere:
1. **Mapping error:** realized historical portfolio returns vs. historical crash patterns pushed through our bitstring-to-loss mapping.
2. **Model error:** historical crash patterns vs. the tree model (ideal). How much the tree approximation misses.
3. **Hardware error:** tree model ideal vs. what the real QPU sampled (raw and mitigated). This is the quantum research result.

### 3.6 Hardware-native layout (the key engineering idea)
IBM Heron chips use a heavy-hex connectivity: each qubit connects to at most 3 neighbors. A tree whose edges don't match the chip forces SWAP gates, which add noise and depth. So for hardware runs we **fit the model to the chip's shape**:
1. Pick a connected set of N good physical qubits and a spanning tree over them using real calibration data (lowest two-qubit and readout error).
2. Assign assets to those physical qubits to maximize the total MI along the chip's tree edges (a quadratic assignment problem; solve with greedy init plus simulated annealing over swap moves).
3. Fit the Bayesian network parameters on *that* tree.
Result: zero SWAPs, minimal depth. Report **native model efficiency** = (total MI captured by the native tree) / (total MI of the unconstrained Chow-Liu tree). The gap is the price of fitting the hardware, and it's an interesting number in itself.

### 3.7 The interference dial (exploratory experiment)
A tree Bayesian network is "classical" in the sense that its distribution is easy to sample. To push into territory where brute force fails, add one interference layer after the tree preparation, controlled by a dial `lam`:
- For each tree edge `(i, j)`: `RZZ(2*lam)` on `(i, j)` (native-adjacent, so no SWAPs).
- Then for every qubit: `RX(2*lam)`.
At `lam = 0` the circuit is exactly the calibrated model. For `lam > 0` the phases interfere and the output distribution changes in ways that, for N in the 40+ range, can't be computed by full statevector simulation on a laptop. This experiment has **no financial justification**; it is labeled exploratory everywhere. Its purpose: (a) quantify how quantum interference deforms the scenario distribution and its tail metrics, and (b) run a **classical-reproducibility check** using Aer's matrix-product-state simulator and report honestly whether a smarter classical method can still reproduce the hardware's ideal target.

---

## 4. Environment and accounts

- Python 3.11+. Use a venv. snake_case everywhere.
- Dependencies (pin minimum versions in `requirements.txt`): `qiskit>=2.0`, `qiskit-ibm-runtime`, `qiskit-aer`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`, `yfinance`, `plotly`, `matplotlib`, `pyarrow`, `pytest`, `tomli` only if Python < 3.11. Optional: `mthree` (readout mitigation; use only if it installs and works with the installed Qiskit, otherwise skip and use the fallback in section 9).
- Do **not** use `qiskit-finance`, `qiskit-optimization`, or `qiskit-algorithms`. Not needed.
- IBM account: write `scripts/setup_ibm.py` that prompts for the API key and instance, then calls `QiskitRuntimeService.save_account(...)`. Check the installed `qiskit-ibm-runtime` docs for the correct `channel` value for the current IBM Quantum Platform (recent versions use `"ibm_quantum_platform"`) and whether an instance CRN is required. Never write tokens into code, config, logs, or git. Add `.env`, `runs/`, `data_cache/`, `library/` to `.gitignore` (but see section 15 on committing a sample run).
- Budget facts: the IBM Open Plan allows about **10 minutes of QPU time per 28-day rolling window**. Use **job mode** (pass the backend as `mode`). Do not rely on sessions. Pack many circuits into one Sampler job as multiple PUBs to pay the queue wait once.

---

## 5. Data spec (`data.py`)

- Source: `yfinance`, auto-adjusted daily closes. Cache every download to `data_cache/` as parquet with a content hash; record the hash in every run.
- Default fit window: **4 years** (some tickers like CEG have shorter histories; 4y keeps them in).
- Alignment: crypto trades 7 days a week, stocks don't. Align on dates where **all** selected assets have a price (effectively stock trading days); compute simple returns between consecutive aligned dates (crypto weekend moves fold into Monday). Document this.
- Validation: drop any ticker with more than 2% missing aligned data or insufficient history, log it, and continue. Never silently fill gaps with zeros.
- Holdout: reserve the most recent 12 months as an out-of-sample period (section 12.5). Fit on the earlier window.
- If downloads fail and there's no cache, allow a clearly labeled `SYNTHETIC` mode for tests only. **Refuse to submit hardware jobs on synthetic data.**

### Universe tiers (must be nested: each tier contains the previous one)
- `n6`: `NOW, GOOG, CEG, GLDM, BTC-USD, SOL-USD`
- `n12`: n6 + `ISRG, AMZN, VST, HOOD, ETH-USD, SPY`
- `n24`: n12 + `QQQ, IWM, TLT, GLD, XLE, XLU, XLV, XLF, NVDA, MSFT, AAPL, META`
- `n48`: n24 + 24 liquid large caps and ETFs spanning sectors, e.g. `JPM, XOM, LLY, UNH, COST, WMT, KO, PG, JNJ, CVX, AVGO, TSLA, AMD, CRM, ORCL, NFLX, GS, CAT, LMT, NEE, SLV, USO, XLI, XLP`
- `n96`: n48 + 48 more liquid large caps, ETFs, and major crypto with 4+ years of history (choose from the S&P 100 plus sector/commodity ETFs; e.g. `MA, V, BAC, WFC, MS, BLK, AXP, PFE, MRK, ABBV, TMO, ABT, AMGN, GILD, HD, LOW, MCD, SBUX, NKE, DIS, T, VZ, TMUS, PEP, BA, GE, HON, DE, UPS, RTX, INTC, CSCO, QCOM, TXN, IBM, ADBE, INTU, AMAT, MU, PLTR, COIN, MSTR, SOFI, DUK, SO, NRG, XLK, XLY`). If a ticker fails validation, replace it with another liquid name so the tier still has 96 assets, and log the swap.
- Store tiers in `config/universe.toml`. The user may edit them.

### Portfolios (`config/portfolios.toml`)
- `my_portfolio`: equal weight across `GOOG, NOW, ISRG, AMZN, GLDM, VST, CEG, HOOD, BTC-USD, ETH-USD, SOL-USD` with a comment telling the owner to replace with his real weights.
- `fyp_composite`: an illustrative "typical finance-TikTok portfolio" (e.g., `NVDA, TSLA, PLTR, SOFI, HOOD, AMD, BTC-USD`) labeled everywhere as an illustrative composite, not any real person.
- `equal_weight_tier`: equal weight across the whole active tier.
- `sixty_forty`: 60% SPY, 40% TLT.
Weights are normalized at load. Tickers not in the active tier are dropped with a warning shown in the report.

---

## 6. Model spec (`model.py`)

1. Crash indicators: `C[t, i] = 1 if r[t, i] <= quantile_i(q)` over the fit window.
2. Pairwise MI on binary indicators with Laplace smoothing (`alpha = 1` pseudo-count on each 2x2 cell).
3. Chow-Liu tree: maximum spanning tree on MI (networkx). Root = asset with the largest total MI to its neighbors (ties broken by ticker order). Also support a **degree cap** (default 3) using a greedy constrained maximum spanning tree, for the "logical" layout mode.
4. Parameters with Laplace smoothing: `P(root=1)`, `P(c=1|a=0)`, `P(c=1|a=1)` for each edge. Clip probabilities to `[1e-4, 1 - 1e-4]`.
5. Classical ancestral sampler (vectorized numpy, seeded) producing an `(shots, n)` 0/1 array in **asset order**.
6. **Exact crash-count distribution via tree DP.** For node `v` and its state `s`, let `D_v[s]` be the distribution (vector over counts) of the number of crashes in `v`'s subtree given `v = s`. Leaf: `D_v[s] = delta(s)`. Internal: start from `delta(s)`, and for each child `c` convolve with `sum_t P(c=t | v=s) * D_c[t]`. Root total: `sum_s P(root=s) * D_root[s]`. Output `P(count = k)` for `k = 0..n`. Unit test against brute force for `n <= 12`.
7. Exact marginals `P(i=1)` and exact pairwise `P(i=1, j=1)` for all pairs (via tree propagation; for pairs, condition along the tree path). Unit test against brute force for `n <= 10`.
8. Serialize the fitted model (tree edges, root, BFS order, parameters, MI matrix, data hash) to JSON.

---

## 7. Circuit spec (`circuits.py`)

- `build_tree_circuit(model) -> QuantumCircuit`: `n` qubits, qubit `i` = asset `i` in model order. Gates in BFS order exactly as section 3.3. Use `RYGate(theta).control(1)` or `cry`; verify the transpiled two-qubit count equals `2 * (n - 1)` under native layout.
- `add_interference(circuit, edges, lam)`: section 3.7.
- `readout_calibration_circuits(n)`: two circuits, all-zeros (measure only) and all-ones (`X` on every qubit, then measure).
- Always end with `measure_all()` on the **logical** circuit before transpiling. The classical register `meas` then maps bit `i` to logical qubit `i` regardless of physical placement.
- **Bit order:** Qiskit bitstrings are little-endian. Character `-1-i` of a count key is qubit `i`. Implement one helper `bitstring_to_array(key, n)` and use it everywhere. Unit test it.

### Layout (`layout.py`)
- `logical` mode: degree-capped Chow-Liu tree; let the transpiler route (`optimization_level=3`). Use only for `n <= 24` comparisons.
- `native` mode (default for hardware):
  1. From `backend.target` and the coupling map, build an undirected device graph. Edge weight = two-qubit gate error; node weight = readout error. Exclude qubits/edges flagged faulty or with error above a configurable threshold.
  2. Grow a connected set of `n` physical qubits from the best-scoring qubit, Prim-style, always adding the lowest-error frontier edge. The added edges form the device tree `T_dev`.
  3. Assign assets to `T_dev` nodes: initialize by matching BFS orders of the Chow-Liu tree and `T_dev`, then simulated annealing over pairwise swaps maximizing `sum MI` along `T_dev` edges (fixed seed, configurable iterations). Fit Bayesian network parameters on this tree.
  4. Transpile with `initial_layout` fixed to that assignment and assert: no `swap` ops, two-qubit gate count equals `2 * (n - 1)` (plus interference gates if used).
  5. Report native model efficiency (section 3.6).
- For the noisy simulator, use the same procedure against a fake backend (`qiskit_ibm_runtime.fake_provider`, default `FakeFez` or `FakeTorino`).

---

## 8. Execution spec (`execution.py`)

Four modes behind one interface. Every mode returns per-PUB counts in asset order plus metadata.
- `exact`: classical ancestral sampler (and exact DP quantities). Any `n`.
- `ideal`: `StatevectorSampler`, `n <= 24`. Must match `exact` statistically (and exactly in probabilities, see tests).
- `noisy`: `qiskit_ibm_runtime.SamplerV2(mode=FakeBackend())` on the transpiled ISA circuit. Use Aer `method="matrix_product_state"` for `n > 28` if it runs in reasonable time; otherwise cap noisy runs at the largest size that finishes and say so.
- `ibm`: `QiskitRuntimeService().backend(name)` or `least_busy(operational=True, simulator=False, min_num_qubits=n)`. Transpile with `generate_preset_pass_manager(backend=..., optimization_level=3, initial_layout=...)`. `SamplerV2(mode=backend)`, `options.default_shots`, and for the "mitigated" variant `options.dynamical_decoupling.enable = True`, `options.twirling.enable_gates = True`, `options.twirling.enable_measure = True`. The "raw" variant sets these off. Results: `result[k].data.meas.get_counts()`.

### QPU budget guard (non-negotiable)
- `runs/qpu_ledger.json` records every hardware job: timestamp, job id, backend, PUB count, shots, estimated seconds, **actual quantum seconds** (from `job.usage()` or `job.metrics()["usage"]["quantum_seconds"]`, whichever the installed version supports).
- Before submitting, compute an estimate: `shots * n_pubs * (circuit_duration + rep_delay) * 1.3 + 3 s per PUB`, using durations from `backend.target` and the backend's default rep delay. After the first real job, also show a calibrated estimate using measured seconds per shot from the ledger.
- Hard cap: refuse to submit if (spent in the last 28 days + estimate) exceeds **9.0 minutes** (configurable), leaving margin.
- Hardware submission requires both a `--yes` flag and an interactive confirmation that prints: backend, qubits used, two-qubit gate count, depth, PUBs, shots, estimate, remaining budget.
- Never submit hardware jobs from tests, imports, or default commands.

### Persistence (every run, every mode)
`runs/<UTC timestamp>_<mode>_<experiment>/` containing: config snapshot, data hash, model JSON, circuits (QPY), transpiled ISA circuits (QPY), physical layout, backend name, a calibration snapshot (per-qubit readout error, T1/T2 if available, per-edge two-qubit error for used qubits), raw counts per PUB (JSON), job ids, actual QPU seconds, and computed metrics JSON. A `replay` command recomputes all analysis from saved counts with zero QPU usage. **QPU data is never paid for twice.**

---

## 9. Mitigation spec (`mitigation.py`)

- Error *suppression* happens at runtime (dynamical decoupling, gate and measurement twirling).
- Readout *correction* in post-processing:
  - Preferred: `mthree` if available and compatible; apply to the counts for distribution-level quantities.
  - Fallback (always implement): tensored per-qubit correction from the two calibration PUBs. Per qubit, `e01 = P(read 1 | prepared 0)`, `e10 = P(read 0 | prepared 1)`. Corrected marginal `p = (p_meas - e01) / (1 - e01 - e10)`, clipped to `[0, 1]`. For pairs, invert the 2x2 tensor-product confusion matrix on the 4 joint outcomes and clip/renormalize.
- Report raw and corrected side by side. Never present corrected numbers without the raw ones.
- Expected physics worth explaining in the docs: readout errors of ~1% on 100 qubits create roughly one false "crash" per shot, inflating crash counts. Hypothesis: raw hardware makes risk look **scarier**. Measure it; don't assume it.

---

## 10. Scenarios and loss mapping (`scenarios.py`)

Each scenario is one bitstring = one simulated day.
- Per asset, from the fit window: `m_crash_i` = mean return on crash days, `m_normal_i` = mean return on non-crash days.
- **Deterministic mapping (default, for clean attribution):** portfolio return `R(x) = sum_i w_i * (x_i * m_crash_i + (1 - x_i) * m_normal_i)`; loss `L = -R`. Loss depends only on the bitstring, so differences between sources isolate the scenario distribution.
- **Bootstrap mapping (realism mode):** for each scenario and asset, draw a return uniformly from that asset's empirical crash-day set or non-crash set (seeded). Label clearly.
- **Scenario library (the reusable asset):** append all scenarios from every source to `library/scenarios.parquet` with columns: `run_id, source (historical|independent|gaussian_copula|tree_exact|ideal|noisy|qpu_raw|qpu_mitigated|interference_lam_x), tier, backend, date, bitstring (packed), shot_index`. Store the per-asset mapping tables per run. The `stress` command evaluates **any** portfolio file against **any** saved source with no QPU usage. This is what makes the 10 minutes keep paying off all month, including for Discord members' portfolios later.

---

## 11. Baselines (`baselines.py`)

All produce bitstrings in asset order so they flow through the same mapping and metrics:
1. **Historical:** the actual crash patterns from the fit window (and, separately, realized portfolio returns for mapping-error measurement).
2. **Independent:** each asset crashes independently with its own marginal. Shows how much dependence matters for tails.
3. **Gaussian copula:** Ledoit-Wolf correlation of normal scores of returns; simulate 1,000,000 correlated normals; crash if `z_i <= Phi^-1(q)`.
4. **Tree exact:** ancestral sampling from the fitted model (1,000,000 samples, seeded) plus exact DP quantities. This is the ground truth for hardware error.

---

## 12. Metrics (`metrics.py`)

Compute for every source vs. tree exact (and vs. historical for model error):
1. **Marginals:** per-asset crash probability; mean absolute error.
2. **Pairs:** `P(i=1, j=1)` for all pairs; MAE on tree edges and on all pairs. **Lift matrix** `P(i,j) / (P(i) P(j))` for heatmaps.
3. **Crash-count distribution:** `P(count >= k)` for `k = 0..n`; Wasserstein-1 distance and KS statistic vs. tree exact (exact DP).
4. **Tail risk per portfolio:** VaR and CVaR at 95% and 99% of 1-day loss; probability of a portfolio loss worse than -3%, -5%, -10%.
5. **Out-of-sample check:** on the 12-month holdout, compare each source's predicted VaR exceedance rate to the realized rate (Kupiec proportion-of-failures test). Applies to historical, independent, copula, and tree models; the QPU sample is evaluated through the same tree-model lens.
6. **Uncertainty:** bootstrap over shots (1,000 resamples) for every QPU/simulator metric. Any hardware-vs-ideal difference inside the 95% CI is reported as "not distinguishable from shot noise."
7. **Native model efficiency** (section 3.6) per tier.
8. **Hardware context:** two-qubit gate count, depth, mean readout error and mean two-qubit error of the qubits used.

Headline research metric: **crash-count Wasserstein distance and VaR error vs. N, raw vs. mitigated, with CI bands.**

---

## 13. Experiments and QPU budget plan (`experiments.py`)

Run everything locally first (`exact`, `ideal`, `noisy`) and only then spend QPU. Pack each experiment into as few jobs as possible.

| Experiment | What | Hardware job(s) | Budget cap |
|---|---|---|---|
| E0 smoke | n6 tree, 4,000 shots, mitigated, plus 2 readout-calibration PUBs | 1 | 0:30 |
| E1 scale ladder | n6, n12, n24, n48, n96 native trees, 8,000 shots each, plus readout-calibration PUBs per size. One job mitigated, one job raw | 2 | 3:00 |
| E2 interference (exploratory) | n48 native tree with `lam` in {0, 0.05, 0.1, 0.2}, 8,000 shots each, mitigated | 1 | 1:30 |
| Reserve | failures, reruns | | 5:00 |

These are caps, not predictions. E0 calibrates the estimator; re-plan E1/E2 with measured seconds-per-shot before submitting. If n96 doesn't fit the chosen backend's good qubits, use the largest native tree that does and report it.

**E2 classical-reproducibility check:** compute the ideal E2 distributions with statevector where possible (`n <= 28`) and with Aer MPS at n48. Report whether MPS reproduces them (bond dimension needed, runtime) and compare all three: ideal (if computable), MPS, QPU. Report the result honestly either way.

**Monthly refresh:** `qtr monthly` refits on fresh data, reruns E1 within budget, appends to the library, and adds a point to a hardware-drift timeline (same circuits, different months).

---

## 14. Visual outputs (`viz.py`, `report.py`)

One interactive `reports/dashboard.html` (Plotly, single self-contained file) plus PNG exports (matplotlib) at 1080x1350 for social posts. Light background `#fcfcfb`, text `#0b0b0b`, secondary text `#52514e`. Fixed colors per source, never cycled: historical `#6b6a65` (gray reference), tree exact `#2a78d6` (blue), QPU raw `#eb6834` (orange), QPU mitigated `#1baf7a` (aqua), Gaussian copula `#6250d6` (violet), independent `#eda100` (yellow, dashed), noisy simulator `#e87ba4` (magenta). Every chart: a title stating the takeaway, axis units, legend for 2+ series, source footnote (backend, date, shots, data window), and the not-investment-advice line. No dual y-axes. If any data is synthetic, watermark "SYNTHETIC DATA" across the chart.

Required figures:
1. **Crash network on the chip:** the device's heavy-hex layout; used qubits labeled with tickers and colored by sector; tree edges highlighted with width proportional to MI; toggle to color nodes by marginal error. The signature image of the project.
2. **Crash network (abstract):** the Chow-Liu tree as a graph with tickers.
3. **"How many crash together":** `P(count >= k)` vs `k`, log y-axis; historical, independent, copula, tree exact, QPU raw, QPU mitigated (noisy sim available as a dashboard toggle).
4. **Fidelity vs. scale (headline):** x = qubits (6, 12, 24, 48, 96); y = crash-count Wasserstein distance vs. exact; raw vs. mitigated with CI bands. Second panel (separate chart, not dual-axis): 99% VaR error for `my_portfolio`.
5. **The crash matrix:** lift-matrix heatmaps for tree exact, QPU, and their difference (diverging palette, gray midpoint), tickers ordered by tree BFS so clusters are visible. Plotly 3D bar/surface toggle of the same matrix.
6. **Portfolio tail panel:** VaR/CVaR 95/99 for each portfolio under each source, with CIs.
7. **Interference dial (exploratory):** tail metrics vs. `lam`, labeled exploratory.
8. **QPU ledger panel:** seconds spent per run, remaining budget this window.

`reports/summary.md`: auto-generated plain-English findings with every number linked to its run folder, including the three-layer error decomposition (mapping, model, hardware).

---

## 15. Repository layout and CLI

```
quantum-tail-risk/
  README.md                  # what, why, results, honest claims, how to run
  requirements.txt
  config/default.toml        # q, window, holdout, shots, budget cap, backend prefs, seeds
  config/universe.toml
  config/portfolios.toml
  docs/HOW_IT_WORKS.md       # plain-English walkthrough of every concept in section 3
  docs/DECISIONS.md          # judgment calls you made where the spec was silent
  scripts/setup_ibm.py
  src/qtr/
    __init__.py
    config.py  data.py  model.py  layout.py  circuits.py  execution.py
    mitigation.py  scenarios.py  baselines.py  metrics.py  experiments.py
    viz.py  report.py  ledger.py  cli.py
  tests/
  runs/  data_cache/  library/  reports/
```

CLI (`python -m qtr <command>`):
- `fetch --tier n12` download/validate/cache.
- `fit --tier n12 [--layout logical|native --backend fake_fez]` fit model, print tree, MI, efficiency.
- `simulate --mode exact|ideal|noisy --tiers n6,n12,n24` local runs.
- `plan --experiment E1 --backend <name>` transpile and print gate counts, depth, layout, estimate, remaining budget. Zero QPU usage.
- `run-ibm --experiment E0|E1|E2 --yes` budget-guarded hardware submission.
- `replay --run <id>` recompute analysis from saved counts.
- `stress --portfolio my_portfolio [--source qpu_mitigated --run <id>]` reuse library, no QPU.
- `report` build dashboard, PNGs, summary.
- `monthly` refresh routine.
- `ledger` show QPU spend.

Commit one completed example run (small, with counts) under `examples/` so the repo shows real results without needing a token.

---

## 16. Build order with acceptance criteria

Do these in order. Don't start a milestone until the previous one's tests pass.

1. **M1 Data:** fetch, cache, align, crash indicators. *Done when:* each asset's crash rate on the fit window is within 0.5 percentage points of `q`; alignment and drop logs are printed.
2. **M2 Model:** MI, Chow-Liu, parameters, sampler, exact DP, exact pairs. *Done when:* DP matches brute-force enumeration exactly for n <= 12; ancestral sampler matches exact marginals within 3 standard errors at 1M samples.
3. **M3 Circuits:** tree circuit, bit-order helper. *Done when:* for n <= 12, the statevector probabilities of the circuit equal the Bayesian network's exact joint probabilities to 1e-10 for **every** bitstring.
4. **M4 Scenarios, baselines, metrics.** *Done when:* end-to-end `simulate --mode exact,ideal` on n6 and n12 produces all metrics; ideal-vs-exact differences are inside shot-noise CIs.
5. **M5 Native layout + noisy sim.** *Done when:* on a fake backend, native transpilation has zero SWAPs and exactly `2(n-1)` two-qubit gates for n6 through n48 (and n96 if the fake device allows); noisy runs complete for at least n6 to n24.
6. **M6 Hardware path + ledger + replay.** *Done when:* `plan` works for E0/E1/E2 against a fake backend and (if credentials exist) a real backend name; `run-ibm` refuses without `--yes`, refuses past the budget cap, and is fully exercised in tests against a fake backend with the network call mocked; `replay` reproduces metrics byte-identically from saved counts.
7. **M7 Visuals + report.** *Done when:* all eight figures render from the exact/ideal/noisy runs, PNGs export, summary.md generates.
8. **M8 Docs.** README, HOW_IT_WORKS.md, DECISIONS.md complete.
9. **M9 Final local dress rehearsal:** run E0/E1/E2 fully in `noisy` mode as a stand-in for hardware and generate the full report. Hand off to the owner with exact commands for the real runs.

### Required tests (`pytest -q`, no network, no QPU)
- bit-order round trip
- tree circuit statevector == Bayesian network exact joint (n <= 12, several random models)
- DP crash-count == brute force; exact pairs == brute force
- Laplace smoothing and probability clipping
- native layout: zero swaps, correct two-qubit count
- readout correction recovers known injected errors on synthetic counts
- deterministic loss mapping on a hand-computed example
- budget guard: refuses over cap, refuses without `--yes`, ledger math over a 28-day rolling window
- replay determinism
- synthetic-data guard blocks hardware submission

---

## 17. Coding standards

- Small, pure functions; type hints; docstrings that explain **why**, not just what. Comments at the level of a smart student learning the topic.
- All randomness seeded and seeds recorded in run metadata.
- No hidden global state; config flows from TOML into explicit arguments.
- Fail loudly with clear messages (e.g., which ticker failed, which qubit was excluded and why).
- Use current Qiskit 2.x APIs only (`generate_preset_pass_manager`, primitives V2, PUBs). No deprecated `execute`, `QuantumInstance`, `BackendV1`, or V1 primitives.
- Keep plotting separate from computation so metrics are testable without matplotlib.

---

## 18. Pitfalls checklist (verify each before handing off)

- [ ] Bitstrings decoded little-endian through the single helper everywhere.
- [ ] Logical `measure_all()` before transpile, so physical layout never scrambles asset order.
- [ ] Parameters fit on the **native** tree for hardware runs (not the unconstrained Chow-Liu tree).
- [ ] Crash thresholds computed on the fit window only, never including the holdout.
- [ ] Crypto/stock calendar alignment documented.
- [ ] Raw and mitigated always reported together; CIs on every hardware number.
- [ ] Interference results labeled exploratory; no finance claims from them.
- [ ] No tokens in code, config, logs, or git history.
- [ ] Hardware never runs on synthetic data, never without `--yes`, never past the budget cap.
- [ ] Every chart has source, date, backend, shots, and the not-investment-advice line.

---

## 19. Definition of done

- `pytest -q` passes with no network.
- `python -m qtr simulate --mode exact,ideal,noisy` and `python -m qtr report` produce the full dashboard from local runs.
- `python -m qtr plan --experiment E1` prints a credible budget plan.
- The owner can run E0, then E1, then E2 on real hardware with three commands, and the report rebuilds automatically from the saved counts.
- README states the research question, the honest claims, the headline results (once real data exists), and how to reproduce everything.

---

## 20. Summary

We are building a **Quantum Tail-Risk Observatory**: a pipeline that learns which assets crash together from real price data, encodes that crash network into a quantum circuit exactly (one qubit per asset, no training), and samples it on a real IBM quantum computer at up to ~100 qubits. The circuit is laid out to match the chip's physical wiring so it runs with no extra gates. Because the model is tree-shaped, the perfect answer is still computable on a laptop, which lets us measure exactly how much real hardware noise distorts portfolio tail risk as circuits grow, something most quantum demos can't verify. An exploratory interference experiment pushes beyond brute-force simulation and honestly tests whether smarter classical methods can still keep up. Every QPU shot is saved into a reusable scenario library, so any portfolio (the owner's, a TikTok-style composite, or a Discord member's) can be stress-tested all month without spending more quantum time. Outputs: a chip-map visualization of the crash network, a fidelity-vs-scale research chart, crash-matrix heatmaps, portfolio tail-risk panels, and a plain-English report. The headline question: **does real quantum hardware make crash risk look safer or scarier than it really is?**
