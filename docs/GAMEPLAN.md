# Gameplan: Quantum Tail-Risk Observatory

Written 2026-10-12, before any code. This is the execution plan for `docs/SPEC.md`.
Judgment calls live in `docs/DECISIONS.md` (D-numbers below point there).
Update the **Status board** as milestones land. Don't rewrite history; add to it.

---

## 0. Status board

| Milestone | State | Exit gate |
|---|---|---|
| M0 Scaffold | **done 2026-10-13** (11 tests) | venv + `pip install -e .` + `pytest -q` (empty) green, git initialized, `.gitignore` committed first |
| M1 Data | **done 2026-10-13** (1,254 aligned days, all 96 pass, max gap 0.09pp) | crash rate per asset within 0.5pp of q; alignment/drop log printed |
| M2 Model | **done 2026-10-13** (DP, marginals, pairs == brute force to 1e-12; sampler within 3 SE) | DP == brute force (n<=12); pairs == brute force (n<=10); sampler within 3 SE at 1M |
| M3 Circuits | **done 2026-10-13** (statevector == tree for every pattern, n<=12, 1e-10) | statevector == BN joint to 1e-10, every bitstring, several random models, n<=12 |
| M4 Scenarios/baselines/metrics | not started | `simulate --mode exact,ideal` on n6, n12; ideal vs exact inside shot-noise CI |
| M5 Native layout + noisy | not started | FakeFez: 0 swaps, 2q count == 2(n-1), n6..n96; noisy completes n6..n24 (MPS beyond if feasible) |
| M6 Hardware path/ledger/replay | not started | `plan` E0/E1/E2; `run-ibm` guard tests with mocked network; replay byte-identical |
| M7 Visuals + report | not started | 8 figures + PNGs + summary.md from local runs |
| M8 Docs | not started | README, HOW_IT_WORKS, DECISIONS complete |
| M9 Dress rehearsal | not started | E0/E1/E2 in noisy mode, full report, hand-off commands verified |
| — Real QPU (owner) | blocked on M9 | E0 → recalibrate estimator → E1 → E2 |

---

## 1. Non-negotiables

These hold in every session, every file. They're repeated in `CLAUDE.md` so they load automatically.

1. **Honesty contract** (spec §2). No advantage, speedup, "predicts crashes," or trade-signal language, ever. Every report, chart, and the dashboard carries *Research and education only. Not investment advice. Backward-looking data.*
2. **QPU safety.** No hardware submission from tests, imports, or default commands. `run-ibm` needs `--yes` *and* an interactive confirmation, refuses past the 9.0-minute rolling cap, and refuses synthetic data. The network call is mocked in every test.
3. **One bit-order helper.** `bitstring_to_array` is the only place Qiskit's little-endian keys get decoded. Grep for `[::-1]` on count keys during review; any hit outside the helper is a bug.
4. **Ground truth is always the model that was actually encoded.** For hardware or native runs that's the *native-tree* model, not the unconstrained Chow-Liu tree (D7).
5. **Raw next to mitigated, CI on every hardware number.** A difference inside the 95% bootstrap CI is reported as "not distinguishable from shot noise."
6. **No tokens anywhere** in code, config, logs, run folders, or git.
7. **Milestone gates are real.** Don't start M(k+1) until M(k)'s tests pass.
8. **Clarity first.** The owner must be able to read every file. Prefer a 15-line loop with a comment over a clever one-liner.

---

## 2. Verified environment (2026-10-12)

- Machine Python: **3.12.5** (meets 3.11+, so no `tomli`; use stdlib `tomllib`).
- Latest on PyPI today: `qiskit 2.5.2`, `qiskit-ibm-runtime 0.50.0`, `qiskit-aer 0.17.2`, `mthree 3.0.0`.
- Already installed globally (the venv will reinstall them): numpy 2.3.3, pandas 2.3.2, networkx 3.6.1, yfinance 0.2.65.
- **Check at M0 against the installed packages, not from memory:**
  - the `QiskitRuntimeService.save_account` `channel` value (expected `"ibm_quantum_platform"`) and whether `instance` (a CRN) is required;
  - how to read quantum seconds: `job.usage()` vs `job.metrics()["usage"]["quantum_seconds"]`;
  - that `qiskit_ibm_runtime.fake_provider` still ships `FakeFez` (156q) and `FakeTorino` (133q);
  - whether `mthree 3.0` imports and runs against qiskit 2.5. If not, use the tensored fallback only (D15).

---

## 3. Architecture

### 3.1 Module dependency graph (arrows = "imports")

```
config ─┬─> data ──> model ──> layout ──> circuits ──> execution ──> experiments ──> cli
        │              │          │                       │   ▲            │
        │              ├─> baselines                      │   └─ ledger    ├─> report ──> viz
        │              └─> scenarios ──> metrics <────────┘                │
        │                                   ▲                              │
        └─────────────> mitigation ─────────┘                              │
                                  (report reads run folders only) <────────┘
```

Rules: `metrics` never imports `viz`, `matplotlib`, or `plotly` (spec §17). `viz` takes plain dicts or DataFrames and computes nothing statistical. `report` reads **only** run folders and the library, never live objects, so `replay` and `report` share one code path.

### 3.2 Core data contracts (frozen dataclasses; serialize to JSON)

| Object | Fields (essentials) | Owner module |
|---|---|---|
| `PriceData` | `prices: DataFrame`, `returns_fit`, `returns_holdout`, `tickers`, `data_hash`, `is_synthetic`, `drop_log` | data |
| `CrashData` | `indicators_fit: uint8[T,n]`, `thresholds: float[n]`, `q` | data |
| `TreeModel` | `tickers`, `edges [(parent, child)]`, `root`, `bfs_order`, `p_root`, `p_c_given_0[n]`, `p_c_given_1[n]`, `mi: float[n,n]`, `layout_mode`, `data_hash`, `model_hash` | model |
| `NativeLayout` | `physical_qubits[n]` (asset i → phys), `device_edges`, `efficiency`, `excluded {qubit: reason}`, `calibration_snapshot` | layout |
| `PubResult` | `label`, `samples: uint8[shots,n]` in **asset order**, `raw_counts` (Qiskit keys, kept for audit), `shots` | execution |
| `RunRecord` | `run_id`, `mode`, `experiment`, `backend`, `pubs [PubResult]`, `job_ids`, `quantum_seconds`, `seeds`, `config_snapshot` | execution |
| `MappingTable` | per asset `m_crash`, `m_normal`, crash-day and normal-day return arrays (bootstrap mode) | scenarios |

`model_hash` = sha256 of the canonical JSON of the model's parameters. Runs, library rows, and baseline caches key on it.

### 3.3 Run folder layout (spec §8, made concrete)

```
runs/20261012T221500Z_noisy_E1/
  meta.json              # run_id, mode, experiment, backend, seeds, versions, data_hash, model_hashes, job_ids, quantum_seconds
  config_snapshot.toml
  models/<tier>.json     # TreeModel per tier (native and/or logical)
  layout/<tier>.json     # NativeLayout + calibration snapshot
  circuits/<tier>.qpy    # logical circuits
  isa/<tier>.qpy         # transpiled circuits
  counts/<pub_label>.json# raw Qiskit counts, untouched
  metrics.json           # everything computed, canonical JSON
```

Byte-identical replay (D17): metrics JSON written with `sort_keys=True`, floats rounded to 12 significant digits, bootstrap RNG seeded from `(run seed, metric name)`. `replay` rewrites `metrics.json` to a temp file and the test asserts the bytes match.

---

## 4. Milestones in detail

Each milestone lists: files touched → key work → tests → traps.

### M0: Scaffold
- **Files:** `pyproject.toml` (src layout, so `pip install -e .` makes `python -m qtr` work; D1), `requirements.txt` (minimum pins), `.gitignore`, `config/*.toml`, `src/qtr/__init__.py`, `src/qtr/__main__.py`, `src/qtr/config.py`, `tests/conftest.py`, `CLAUDE.md`.
- **Work:** `git init`; **commit `.gitignore` before anything else** so `runs/`, `data_cache/`, `library/`, `.env` can never slip in. `config.py` loads the TOML files into frozen dataclasses, with no module-level globals. Runs the API checks from §2.
- **Tests:** config loads; tier nesting assertion (each tier ⊇ the previous one); portfolio weights normalize.
- **Trap:** the spec's tree is `quantum-tail-risk/` inside `~/Quant`. The repo root is `~/Quant/quantum-tail-risk/` (D2).

### M1: Data (`data.py`)
- Download each ticker once, cached as `data_cache/<ticker>_<start>_<end>.parquet`. `data_hash` = sha256 of the **aligned price frame rendered to canonical CSV** (fixed float format), not of the parquet bytes, which vary across pyarrow versions (D4).
- **Windows (D3, owner-decided):** pull 5 calendar years. Holdout = last 12 months. Fit = the 4 years before that (~1,000 aligned days). CEG is dropped (not enough history); NRG replaces it in `n6`.
- Align on the intersection of dates, then simple returns. Validation: >2% missing in aligned span, or first price after window start, means drop and log. Tier backfill (n96 only, per spec) pulls from a ranked `replacements` list in `universe.toml` and logs every swap.
- Crash indicators: `thresholds = np.quantile(r_fit, q, axis=0, method="lower")`, crash = `r <= threshold`. Computed on the **fit window only**; the holdout reuses those thresholds.
- SYNTHETIC mode: generates returns from a known tree BN plus Gaussian noise; `data_hash = "SYNTHETIC-" + sha`; `is_synthetic=True` travels into every run's meta.
- **Tests (offline):** alignment on a hand-built weekday/weekend frame (crypto Saturday and Sunday fold into Monday); a ticker with gaps gets dropped and logged; the crash rate gate; thresholds never touch holdout rows; the synthetic flag propagates.
- **Trap:** with ~750 days, 5% is ~37 crash days per asset. Ties or `method` choice can push the rate a bit over q. The 0.5pp gate catches it; `method="lower"` plus `<=` is the most predictable pairing.

### M2: Model (`model.py`)
- MI with alpha=1 Laplace on the 2×2 table, vectorized: joint counts = `C.T @ C` and friends, which handles all pairs at once even at n96.
- Chow-Liu: `nx.maximum_spanning_tree`. Degree-capped variant: Kruskal on MI-sorted edges, skipping any edge that would push a node past degree 3 (union-find); fall back to the best remaining edge if it gets stuck (log it).
- Root = largest total MI to tree neighbors, ties broken by ticker order. BFS order from the root, ties broken by ticker order, so the tree is deterministic.
- Parameters: Laplace counts, clipped to [1e-4, 1-1e-4].
- `fit_on_tree(crash_data, edges)` is a separate function, because the native layout supplies its own tree (D7).
- Sampler: draw the root, then each child in BFS order with one vectorized `rng.random(shots) < p[parent_state]`.
- DP crash count: post-order recursion → iterative over reversed BFS; `np.convolve` per child. O(n²) total; n96 is trivial.
- Exact marginals: top-down propagation. Exact pairs P(i=1, j=1): for each i, condition on i=1 and propagate a "message" outward across the whole tree (re-rooting at i) in O(n), so all pairs cost O(n²). Simpler and easier to test than path-walking (D8).
- **Tests:** brute-force enumeration over 2^n (n<=12) for the DP; n<=10 for pairs; random trees generated from random Prüfer sequences with random probabilities; smoothing and clipping on hand-computed tables; sampler vs. exact marginals within 3 SE at 1M samples (mark this one `slow` but keep it in the default run if it's under ~5 s).

### M3: Circuits (`circuits.py`)
- `build_tree_circuit(model)`: `ry` on the root; per child in BFS order, `ry(theta0)` then `cry(theta1 - theta0, parent, child)`. Then `measure_all()`.
- `bitstring_to_array(key, n)` → `uint8[n]`, plus `counts_to_samples(counts, n)` → `uint8[shots, n]`. Both live here; every other module imports them.
- `add_interference(qc_without_measure, edges, lam)` → RZZ(2λ) on edges, RX(2λ) on all. **Must be inserted before the measurements.** So the builder returns the unmeasured circuit, and `finalize(qc)` adds `measure_all` (D10).
- `readout_calibration_circuits(n)`.
- **Tests:** bit-order round trip (including n=1 and keys with spaces from multiple registers, which shouldn't happen but should be rejected loudly); `Statevector(qc_unmeasured).probabilities()` vs. BN joint for every bitstring, 1e-10, n in {1, 2, 5, 8, 12}, several seeds; λ=0 interference == plain tree.

### M4: Scenarios, baselines, metrics
- `scenarios.py`: `MappingTable` from the fit window; `deterministic_losses(samples, weights, table)` is one matrix product: `L = -(samples @ (w*(mc-mn)) + w@mn)`. `bootstrap_losses` is seeded. Library writer (D12).
- `baselines.py`: historical (the fit-window indicators themselves), independent, Gaussian copula (`sklearn.covariance.LedoitWolf` on normal scores via rank → `norm.ppf`, then Cholesky, then simulate 1M in 100k-row chunks), tree exact (1M samples + DP). Baseline samples are cached per `model_hash` in `library/baselines/` and not recomputed per run (D13).
- `metrics.py`: marginals and MAE; pair matrix via `S.T @ S / shots`; lift; tail curve `P(count >= k)`; W1 between count distributions (`scipy.stats.wasserstein_distance` with weights, or the CDF-difference sum, which is exact for integer support); KS; VaR/CVaR 95/99 (D14 for the quantile convention); loss-threshold probabilities; Kupiec POF; bootstrap CIs.
- **Bootstrap speed (D16):** resample via `rng.multinomial(shots, p_unique)` over unique bitstrings. QPU samples at n≥24 are almost all unique, so for pairs use resampled index arrays plus `S.T @ S`. 1,000 × (8000×96 matmul) is a few seconds in numpy. Fine.
- **Mapping choice (D31, owner rule):** on the fit window, push the historical crash patterns through BOTH the deterministic and bootstrap mappings, for EVERY portfolio, and compare with the realized portfolio returns (VaR/CVaR 95/99 and loss-threshold probabilities). Headline mapping = lowest total error across all portfolios. Bootstrap draws use one shared seed for every source. Log the numbers and the choice in DECISIONS.
- Every portfolio result carries its tier's dropped weight / proxy note (D30).
- `experiments.py` v1 + `cli simulate`.
- **Tests:** hand-computed 3-asset loss mapping; VaR/CVaR on a known discrete distribution; Kupiec against a textbook value; W1 == 0 on identical inputs; ideal-vs-exact differences inside CI on n6.

### M5: Native layout + noisy sim (`layout.py`, `execution.py` noisy)
- Device graph from `backend.target`: the 2q gate name is discovered from the target (CZ on Heron), error from `target[gate][(a,b)].error`, readout from `target["measure"][(q,)].error`. Exclude any qubit or edge with error `None`, ≥ the threshold, or missing. Log each exclusion with its reason.
- **Prim growth once, at the largest size (D5).** Grow 96 (or the max feasible) from the best seed. The first k qubits added by Prim always form a connected subtree, so **every tier's device tree is a prefix of one growth**. Payoffs: nested chip regions across the scale ladder (cleaner science, since a bigger N only adds qubits), and the readout-calibration PUBs at the largest size cover every tier (D6). That roughly halves E1's PUB count.
- Seed qubit choice: try the top-5 readout-error qubits as seeds and keep the growth with the lowest summed edge+node error.
- Asset assignment: BFS-order matching for the initial guess, then SA over swaps; objective = Σ MI over T_dev edges. Incremental Δ per swap is O(degree), so 200k iterations is fast. Geometric cooling, fixed seed. Then `fit_on_tree`.
- Transpile: `generate_preset_pass_manager(optimization_level=3, backend=..., initial_layout=[phys for asset i], approximation_degree=1.0)` (D9). After transpiling, assert:
  1. no `swap` in `count_ops()`;
  2. 2q count == 2(n-1) for the tree, 4(n-1) with interference (RZZ → 2 CZ unless fractional gates are enabled, which we leave off);
  3. `isa.layout.final_index_layout()[:n] == physical_qubits` (catches VF2PostLayout or a routing pass moving us).
  If (2) comes out *lower* (resynthesis merged something), log it, don't fail, and record the actual count. The equality is the target, but the honest number is what matters.
- Noisy execution: `AerSimulator.from_backend(FakeFez(), method=...)` passed as `SamplerV2(mode=...)`. Use statevector for n<=28 and `matrix_product_state` above. Time n48 and n96 at 8,000 shots; if a size takes more than ~20 min on the laptop, cap it and log the cap in DECISIONS.
- **Tests:** layout on FakeFez: connectivity, nested prefixes, 0 swaps, 2q count, layout preserved, n6..n48 in the default suite, n96 marked `slow`; SA never lowers the objective below the initial guess; efficiency ≤ 1.

### M6: Hardware path, ledger, replay
- `ledger.py`: append-only JSON list; `spent_in_window(now, days=28)`; estimate formula from spec §8 with `circuit_duration` from the target's instruction durations (sum along the critical path; `qc.estimate_duration(target)` if available in the installed qiskit, else a manual critical-path sum) and `backend.default_rep_delay` (fall back to 250 µs if it's missing, and log that). Calibrated estimate = median measured seconds-per-shot from past ledger entries on the same backend family.
- `execution.run_ibm(...)` order of checks, each with a precise refusal message:
  1. `is_synthetic` → refuse. Portfolio commitment missing or stale (D32) → refuse.
  2. no `--yes` → refuse.
  3. spent + estimate > cap → refuse, printing all three numbers.
  4. print the full confirmation block; `input()` must return exactly `yes`.
  5. submit; write `job_id` to the ledger immediately (status `submitted`) so a crash mid-wait can't lose a paid job.
  6. wait; fetch results; write counts; update the ledger with actual quantum seconds.
- `qtr recover --job <id>` (D18): re-fetch a finished job's results into a run folder if the laptop died while waiting. This is what "never pay twice" needs in practice.
- `plan`: the full transpile + assertions + estimate + remaining budget, printed as a table. Works against fake backends offline, and against a real name if credentials exist.
- Mitigated vs. raw options exactly per spec §8. Twirling is left at its defaults otherwise; the actual `num_randomizations`/`shots_per_randomization` get recorded in meta.
- **Tests (all offline, network mocked with `unittest.mock` around the service and `SamplerV2`):** refuses without `--yes`; refuses over cap; refuses synthetic; the 28-day window boundary (an entry 27d23h ago counts, 28d1h doesn't); a mocked submit writes the ledger before results; replay is byte-identical; a guard test that `import qtr.*` never constructs `QiskitRuntimeService`.

### M7: Visuals + report
- `viz.py`: one function per figure, each taking plain data and returning a Plotly figure *or* a matplotlib figure. A shared `style.py`-like section holds the palette, footnote builder, disclaimer line, and SYNTHETIC watermark.
- **Chip map (figure 1, the signature image):** qubit coordinates. First try qiskit's built-in gate-map coordinates for the backend size. Fallback: derive heavy-hex row/column coordinates from the coupling map (degree-3 qubits form the rows; degree-2 "bridge" qubits sit between rows). Verify at M5 which one works for 156q (risk R4).
- Lift heatmaps are ordered by tree BFS; diverging palette with a gray midpoint at lift=1 (difference panel at 0).
- `report.py`: assembles a self-contained `dashboard.html` (`include_plotlyjs="inline"`), PNGs at 1080×1350 via matplotlib, and `summary.md` built from templates where every number is a link to `runs/<id>/metrics.json`.
- The three-layer error table (mapping / model / hardware) is the centerpiece of summary.md.

### M8: Docs
- README (research question, honest claims, results placeholder until real data, reproduce-everything commands), HOW_IT_WORKS (spec §3 in plain English with tiny worked examples, e.g. a 3-asset tree computed by hand), DECISIONS (finalize the seeded list).

### M9: Dress rehearsal
- `experiments` run E0/E1/E2 with `mode=noisy` on FakeFez, using the exact PUB packing the real runs will use. Full report. Copy one small run (n6/n12 E0-style) into `examples/` and commit it.
- Write the **hand-off card** (§8 below) into README with verified commands.

---

## 5. QPU budget plan (numbers to verify with E0)

Estimator (spec §8): `shots × PUBs × (dur + rep_delay) × 1.3 + 3 s/PUB`. With Heron's rep_delay ≈ 250 µs and dur ≈ a few µs, the per-shot cost is ≈ 0.33 ms.

| Job | PUBs | Shots/PUB | Spec estimate | Cap |
|---|---|---|---|---|
| E0 (n6 tree + 2 cal), mitigated | 3 | 4,000 | ~13 s | 0:30 |
| E1 mitigated (5 tiers + 2 shared cal, D6) | 7 | 8,000 | ~40 s | 1:30 |
| E1 raw (same) | 7 | 8,000 | ~40 s | 1:30 |
| E2 (4 λ + 2 cal), mitigated | 6 | 8,000 | ~34 s | 1:30 |
| **Planned total** | | | **~2.1 min** | 9.0 cap / 10 plan |

Without D6 (per-tier calibration, as the spec literally reads), E1 is 15 PUBs per job, ≈ 84 s per job: still under cap but much tighter. Shared calibration keeps the reserve large. **Fallback:** if the owner wants per-tier calibration anyway, it fits.

**Free extra analysis, zero QPU:** the raw and mitigated jobs plus post-hoc readout correction give a 2×2 grid: {no suppression, suppression} × {no correction, correction}. That separates what DD/twirling buys from what readout correction buys. Report all four; "QPU raw" = top-left and "QPU mitigated" = bottom-right per spec.

Sequence for the owner: `plan E0` → `run-ibm E0 --yes` → the ledger now has measured s/shot → `plan E1` (shows spec and calibrated estimates) → `run-ibm E1 --yes` → `plan E2` → `run-ibm E2 --yes` → `report`.

---

## 6. Pre-registered hypotheses

Written down **before** any QPU data so the results can't quietly reshape the story. The report checks each one and says confirmed, refuted, or inconclusive.

- **H1 (scarier at scale):** raw hardware inflates P(count ≥ k) for large k, and the inflation grows with N. Reason: about n × e_readout false crashes per shot, plus gate errors, all with a base rate around 5%, so false 1s outnumber false 0s. At n96 with ~1% readout error that's ~1 spurious crash per shot.
- **H2 (mitigation helps marginals more than tails):** readout correction cuts marginal MAE a lot, but crash-count W1 at n96 stays well above the shot-noise floor, because tensored correction can't undo correlated and gate-level errors.
- **H3 (VaR is more robust than count tails):** `anchor_n6` VaR error grows more slowly with N than count-tail error does. Reason: only its 6 qubits enter its loss, and they exist at every tier, so its VaR error isolates *how the same assets' qubits degrade inside a bigger circuit*. That's a clean probe of crosstalk and context effects (D29).
- **H4 (MPS keeps up, honestly):** Aer MPS reproduces the E2 ideal distributions at n48 with a small bond dimension. Reason: the tree state has Schmidt rank ≤ 2 across any single tree-edge cut, and one RZZ+RX layer at most squares that per cut. So "beyond brute-force statevector" is true (2^48 amplitudes ≈ 4.5 PB), but "beyond classical" almost certainly is not. The report states this up front. It's the honest framing the spec asks for, and it's a stronger resume line than an overclaim.
- **H5 (native price is modest):** native model efficiency is ≥ 0.85 at n6–n24 and drops at n96. A guess; recorded so we can be wrong in public.

---

## 7. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | yfinance rate limits / outages | med | per-ticker cache, retry with backoff, `fetch` is resumable; tests never hit the network |
| R2 | 2q count ≠ 2(n-1) after level-3 resynthesis (near-zero CRY angle on a near-independent edge) | med | `approximation_degree=1.0`; assert no-swap and layout strictly; record the actual count if it's lower (D9) |
| R3 | VF2PostLayout or a layout pass moves our qubits | low-med | post-transpile layout assertion; if it fires, drop to a custom pass manager stage with layout fixed |
| R4 | No built-in heavy-hex coordinates for 156q | med | algorithmic coordinates from the coupling map; kamada-kawai as a last resort |
| R5 | Noisy MPS too slow at n96 × 8,000 shots | med | try fewer shots for the noisy stand-in only, else cap and document (the spec allows this) |
| R6 | mthree incompatible with qiskit 2.5 | med | tensored fallback is always implemented anyway |
| R7 | Not enough good qubits for a 96-node tree on the chosen device | low on Heron r2 | use the largest feasible N, report it (the spec allows this) |
| R8 | Runtime API drift (channel, usage, options names) | med | §2 checks at M0; thin adapter functions in `execution.py` so drift is a one-line fix |
| R9 | Holdout Kupiec has low power at 99% (~2.5 expected exceedances in 250 days) | certain | report it plainly; lead with the 95% results |
| R10 | Laptop dies during a hardware job wait | low | ledger writes the job id at submit time; `recover --job` (D18) |
| R11 | Byte-identical replay breaks from float/library nondeterminism | med | canonical JSON (D17); BLAS thread count pinned to 1 in replay test |

---

## 8. Hand-off card (finalized at M9)

```bash
cd ~/Quant/quantum-tail-risk && source .venv/bin/activate
python scripts/setup_ibm.py                     # once; stores the key outside the repo
python -m qtr fetch --tier n96
python -m qtr plan --experiment E0 --backend <name>
python -m qtr run-ibm --experiment E0 --yes
python -m qtr plan --experiment E1 --backend <name>   # now shows the calibrated estimate
python -m qtr run-ibm --experiment E1 --yes
python -m qtr run-ibm --experiment E2 --yes
python -m qtr report
```

---

## 9. Owner answers (2026-10-13)

1. **Fit window:** 4y fit + 1y holdout; CEG dropped, NRG in its place (D3).
2. **Portfolio weights:** owner is sending them; they go in the gitignored `config/portfolios.local.toml` (D19).
3. **IBM account:** not created yet. Not needed until after M9.
4. **Shared readout calibration:** no answer yet; building with the default (shared, D6). Easy to flip in `config/default.toml` before any hardware run.
5. **GitHub:** yes, a public repo under the owner's account (`gh` is logged in as Teganreef). Secrets stay out (D20). Push after M0 once `.gitignore` and the secret-scan test are in.
