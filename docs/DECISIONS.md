# Decisions

Judgment calls where `docs/SPEC.md` is silent or ambiguous. Each one: the choice, why, and what would change it.
Seeded during planning (2026-10-12); add new ones as the build surfaces them.

| ID | Decision | Why | Revisit if |
|---|---|---|---|
| D1 | `pyproject.toml` with src layout + `pip install -e .`, alongside the required `requirements.txt` | `python -m qtr` must find `src/qtr` without PYTHONPATH hacks | never |
| D2 | Repo root is `~/Quant/quantum-tail-risk/` | matches the spec's tree; keeps `~/Quant` free for other projects | owner prefers `~/Quant` as root |
| D3 | **Owner decision 2026-10-13:** 5 calendar years total: last 12 months = holdout, previous 4 years = fit (~1,000 aligned days). CEG (spun off Feb 2022) lacks the history and is dropped; NRG (similar power-producer exposure, long history) replaces it in `n6`, and n96 backfills from the replacements list | more fit data beats keeping one ticker; owner OK'd losing CEG. CEG in `my_portfolio` gets dropped with a warning | owner wants CEG back |
| D4 | `data_hash` = sha256 of the aligned price frame as canonical CSV (fixed float format), not of parquet bytes | parquet bytes vary with pyarrow version and metadata; the hash must reflect only the data | never |
| D5 | Native layout: one Prim growth at the largest tier; each tier's device tree is the first-k prefix | Prim prefixes are always connected, so tiers sit on nested chip regions (bigger N only adds qubits) | prefix regions turn out much worse than per-tier growth |
| D6 | One pair of readout-calibration PUBs per job, on the largest tier's qubits, shared by all tiers in that job | covers every tier's qubits thanks to D5; cuts E1 from 15 to 7 PUBs per job. Caveat: readout crosstalk with 96 qubits measured vs. 6 differs slightly; documented | owner wants per-tier calibration (fits the budget either way) |
| D7 | Hardware/native runs are graded against the *native-tree* model; the unconstrained Chow-Liu model is used only for model-error analysis and native efficiency | hardware error must compare the circuit to the distribution it actually encodes | never (spec §18) |
| D8 | Exact pairwise P(i=1, j=1) by re-rooted message propagation per i (O(n²) total) instead of path-walking | simpler to code and test, same result | never |
| D9 | Transpile with `approximation_degree=1.0`; assert no swaps and preserved layout strictly; if resynthesis gives *fewer* than 2(n-1) two-qubit gates, record the real count instead of failing | near-independent edges give near-zero CRY angles the optimizer may simplify; the honest number matters more than the formula | never |
| D10 | Circuit builder returns an unmeasured circuit; `finalize()` adds `measure_all()` | interference must go before measurement; also makes statevector tests trivial | never |
| D11 | Crash threshold: `np.quantile(..., method="lower")` with `r <= threshold` | most predictable empirical rate close to q | M1 gate fails |
| D12 | Library stored as one parquet file per run under `library/scenarios/` and read as a pyarrow dataset (logically "scenarios.parquet") | append-only without rewriting a growing file; a crash can't corrupt history | never |
| D13 | Baseline samples (1M tree-exact / copula) cached per `model_hash` in `library/baselines/`, referenced by runs instead of duplicated | avoids ~30 MB per run per tier | never |
| D14 | VaR_α = the α-quantile of the loss distribution (losses positive); CVaR_α = mean loss at or above VaR_α (empirical, inclusive) | standard; documented so numbers are reproducible by hand | never |
| D15 | mthree used only if it imports and passes a smoke test against installed qiskit; tensored correction always computed and always reported | the spec makes mthree optional | n/a |
| D16 | Bootstrap via resampled index arrays (pairs) / multinomial over unique bitstrings (counts), 1,000 resamples, seeded per metric | fast enough at n96 × 8,000 shots | never |
| D17 | Canonical metrics JSON: sorted keys, floats at 12 significant digits, per-metric seeded RNG | makes replay byte-identical across runs on the same machine | never |
| D18 | Extra CLI command `recover --job <id>`; ledger writes the job id at submit time | a paid job must be recoverable if the laptop dies mid-wait ("never pay twice") | never |
| D19 | Real portfolio weights live in `config/portfolios.local.toml` (gitignored), which overrides the committed `config/portfolios.toml` placeholders | repo is public; the owner's real holdings must not be | never |
| D20 | Public-repo hygiene: IBM key only in qiskit's own account file (`~/.qiskit/`, outside the repo); `.gitignore` committed first; a pytest that scans tracked files for key-like strings; `examples/` run reviewed before commit (job IDs are fine, no account/instance info) | owner wants a public GitHub repo with secrets kept private | never |
