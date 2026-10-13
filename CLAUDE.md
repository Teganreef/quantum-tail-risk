# Quantum Tail-Risk Observatory

Read in this order before doing anything:
1. `docs/SPEC.md`: the canonical spec (wins on conflicts unless DECISIONS.md says otherwise).
2. `docs/GAMEPLAN.md`: the execution plan + status board. Update the status board when a milestone gate passes.
3. `docs/DECISIONS.md`: judgment calls. Add a row for every new one.

## Non-negotiables
- Build milestones in order (M0 → M9). Don't start the next until the current gate's tests pass.
- Honesty contract: never claim quantum advantage, speedup, crash prediction, or anything resembling investment advice. Every report/chart/dashboard carries: *Research and education only. Not investment advice. Backward-looking data.*
- Never submit to real IBM hardware from tests, imports, or default commands. `run-ibm` requires `--yes` + interactive `yes`, refuses synthetic data, refuses past the 9.0-min rolling-28-day cap. Mock the network in tests.
- The only Qiskit bitstring decoder is `circuits.bitstring_to_array`. No ad-hoc `[::-1]` on count keys.
- Hardware runs are graded against the native-tree model they actually encode.
- Raw beside mitigated, bootstrap CI on every hardware number.
- No tokens in code, config, logs, run folders, or git.
- Qiskit 2.x APIs only (primitives V2, PUBs, `generate_preset_pass_manager`).
- The owner is a CS junior learning quantum: clear code, docstrings that explain *why*, no cleverness for its own sake.

## Commands
- Tests: `pytest -q` (offline, no QPU)
- CLI: `python -m qtr <command>` (see spec §15)
