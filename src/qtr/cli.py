"""Command-line interface: `python -m qtr <command>`.

Commands are added milestone by milestone. Nothing here ever submits to real
quantum hardware unless the user runs `run-ibm --yes` and confirms interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qtr import DISCLAIMER
from qtr.commitments import verify_commitment, write_commitment
from qtr.config import ConfigError, load_config, resolve_portfolio
from qtr.data import build_price_data, compute_crash_data
from qtr.circuits import build_tree_circuit, counts_to_samples, finalize, two_qubit_gate_count
from qtr.model import TreeModel, exact_crash_count_distribution, exact_marginals, fit_tree_model


def cmd_check(args: argparse.Namespace) -> int:
    """Load every config file and print a summary, so you can confirm what the code sees."""
    config = load_config(args.config_dir) if args.config_dir else load_config()
    print(f"Config loaded from {config.config_dir}\n")
    print(f"Crash definition: worst {config.data.q:.0%} of days per asset")
    print(f"History: {config.data.history_years} years, last {config.data.holdout_months} months held out\n")

    print("Tiers:")
    for name, tickers in config.universe.tiers.items():
        preview = ", ".join(tickers[:8]) + (", ..." if len(tickers) > 8 else "")
        print(f"  {name:>4}: {len(tickers):>3} assets  ({preview})")

    print("\nPortfolios: share of each portfolio's weight that is modeled at each tier")
    tiers = list(config.universe.tiers)
    print(f"  {'portfolio':<24}{'kind':<14}" + "".join(f"{t:>8}" for t in tiers))
    notes = []
    for portfolio in config.portfolios.values():
        cells = []
        for tier in tiers:
            try:
                r = resolve_portfolio(portfolio, config.universe.tickers(tier))
                cells.append(f"{1 - r.dropped_weight:>8.0%}")
                if tier == tiers[-1] and r.coverage_note() != "fully modeled":
                    notes.append(f"  {portfolio.name} at {tier}: {r.coverage_note()}")
            except ConfigError:
                cells.append(f"{'none':>8}")
        label = portfolio.name + (" (private)" if portfolio.from_local_file else "")
        print(f"  {label:<24}{portfolio.kind:<14}" + "".join(cells))
    print("  (weights are renormalized over the modeled part; dropped weight is always reported)")
    for line in notes:
        print(line)
    ok, msg = verify_commitment(config.config_dir)
    print(f"\nPortfolio commitment: {'OK' if ok else 'NOT READY'} - {msg}")
    print(f"\nQPU budget cap: {config.budget.cap_minutes} min per {config.budget.window_days} days")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download/validate/cache prices, then print alignment, drops, and per-asset crash rates."""
    config = load_config(args.config_dir) if args.config_dir else load_config()
    n_assets = len(config.universe.tickers(args.tier))
    print(f"Fetching the full universe ({len(next(reversed(config.universe.tiers.values())))} assets; "
          f"tier {args.tier} = first {n_assets}). Cached files are reused.\n")
    full = build_price_data(config.universe, config.data)
    data = full.for_tier(n_assets)

    print("Validation log:")
    for line in full.drop_log or ("all tickers passed",):
        print(f"  {line}")
    prices = full.prices
    print(f"\nAligned days (every asset has a price): {len(prices)}, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print("  (crypto weekend moves are folded into the next stock trading day)")
    print(f"Fit window:     {len(data.returns_fit)} days, "
          f"{data.returns_fit.index[0].date()} to {data.returns_fit.index[-1].date()}")
    print(f"Holdout window: {len(data.returns_holdout)} days, "
          f"{data.returns_holdout.index[0].date()} to {data.returns_holdout.index[-1].date()}")
    print(f"Data fingerprint: {full.data_hash[:16]}...")

    crashes = compute_crash_data(data, config.data.q)
    q = config.data.q
    print(f"\nCrash days per asset (fit window; target {q:.1%}, must be within 0.5 points):")
    print(f"  {'ticker':<9}{'crash cutoff':>13}{'crash rate':>12}  check")
    worst_gap = 0.0
    for ticker, cutoff, rate in zip(crashes.tickers, crashes.thresholds, crashes.crash_rate_fit):
        gap = abs(rate - q)
        worst_gap = max(worst_gap, gap)
        print(f"  {ticker:<9}{cutoff:>12.2%}{rate:>12.2%}  {'ok' if gap <= 0.005 else 'FAIL'}")
    passed = worst_gap <= 0.005
    print(f"\nM1 gate: {'PASSED' if passed else 'FAILED'} (largest gap from {q:.0%}: {worst_gap * 100:.2f} points)")
    print(f"\n{DISCLAIMER}")
    return 0 if passed else 1


def _print_tree(model: TreeModel, mark: set[str]) -> None:
    """Draw the tree as an indented family tree, root at the top."""
    m = exact_marginals(model)

    def show(node: int, prefix: str, is_last: bool, is_root: bool) -> None:
        name = model.tickers[node] + (" *" if model.tickers[node] in mark else "")
        if is_root:
            print(f"  {name}  (root: crashes {m[node]:.1%} of days)")
            child_prefix = "  "
        else:
            a = model.tickers[model.parent[node]]
            p0, p1 = model.cpt[node]
            branch = "└─ " if is_last else "├─ "
            print(f"{prefix}{branch}{name:<10} {p0:5.1%} if {a} fine  |  {p1:5.1%} if {a} crashed")
            child_prefix = prefix + ("   " if is_last else "│  ")
        kids = model.children(node)
        for k, child in enumerate(kids):
            show(child, child_prefix, k == len(kids) - 1, False)

    show(model.root, "", True, True)


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the crash tree for a tier and print it, plus how well it matches history."""
    config = load_config(args.config_dir) if args.config_dir else load_config()
    if args.layout == "native":
        print("The chip-shaped (native) layout arrives in milestone M5.")
        return 2
    tier_tickers = config.universe.tickers(args.tier)
    data = build_price_data(config.universe, config.data).for_tier(len(tier_tickers))
    crashes = compute_crash_data(data, config.data.q)
    fit = lambda cap: fit_tree_model(
        crashes.indicators_fit, crashes.tickers, data.data_hash,
        alpha=config.model.laplace_alpha, clip=config.model.prob_clip, degree_cap=cap,
    )
    best = fit(None)
    model = best if args.layout == "chow_liu" else fit(config.model.degree_cap)

    held = set(config.portfolios["my_portfolio"].weights)
    print(f"Crash tree for {args.tier} ({model.n} assets, layout: {model.layout_mode})")
    print("Each line: child, its crash chance when its parent is fine | when its parent crashed.")
    print("* = in my_portfolio\n")
    _print_tree(model, held)

    print(f"\nDependence captured (sum of link strengths): {model.total_mi():.3f}")
    if model is not best:
        print(f"Best possible tree captures {best.total_mi():.3f}; "
              f"this one keeps {model.total_mi() / best.total_mi():.1%} of it.")

    exact = exact_crash_count_distribution(model)
    hist = np.bincount(crashes.indicators_fit.sum(axis=1), minlength=model.n + 1) / len(crashes.indicators_fit)
    print("\nHow many assets crash on the same day: history vs. the tree (exact):")
    print(f"  {'k or more':>9}{'history':>10}{'tree':>10}")
    for k in range(1, model.n + 1):
        h, t = hist[k:].sum(), exact[k:].sum()
        if h == 0 and t < 1e-4:
            break
        print(f"  {k:>9}{h:>10.2%}{t:>10.2%}")
    if args.save:
        path = Path(args.save)
        path.write_text(model.to_json())
        print(f"\nSaved model to {path}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_circuit(args: argparse.Namespace) -> int:
    """Show the real crash-tree circuit and run it on a perfect (noise-free) simulator."""
    from qiskit.primitives import StatevectorSampler

    config = load_config(args.config_dir) if args.config_dir else load_config()
    tier_tickers = config.universe.tickers(args.tier)
    if len(tier_tickers) > config.execution.ideal_max_qubits:
        print(f"Perfect simulation is limited to {config.execution.ideal_max_qubits} qubits on a laptop.")
        return 2
    data = build_price_data(config.universe, config.data).for_tier(len(tier_tickers))
    crashes = compute_crash_data(data, config.data.q)
    model = fit_tree_model(crashes.indicators_fit, crashes.tickers, data.data_hash,
                           alpha=config.model.laplace_alpha, clip=config.model.prob_clip)
    qc = build_tree_circuit(model)
    print(f"Crash-tree circuit for {args.tier}: {qc.num_qubits} qubits, "
          f"{two_qubit_gate_count(qc)} controlled rotations (one per tree link)\n")
    print("Qubit = asset:  " + ", ".join(f"q{i}={t}" for i, t in enumerate(model.tickers)) + "\n")
    if qc.num_qubits <= 8:
        print(qc.draw("text", fold=110))

    shots = args.shots
    result = StatevectorSampler(seed=config.seed).run([finalize(qc)], shots=shots).result()
    counts = result[0].data.meas.get_counts()
    samples = counts_to_samples(counts, model.n)
    exact = exact_marginals(model)
    print(f"\nRan {shots:,} shots on a perfect simulated quantum computer (each shot = one made-up day).")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    for key, c in top:
        crashed = [model.tickers[i] for i, b in enumerate(reversed(key)) if b == "1"]
        print(f"  {key}  x{c:<6} = {', '.join(crashed) if crashed else 'nobody crashed'}")
    print(f"\n  {'asset':<9}{'quantum':>9}{'exact':>9}{'within noise?':>15}")
    for i, t in enumerate(model.tickers):
        se = np.sqrt(exact[i] * (1 - exact[i]) / shots)
        ok = abs(samples[:, i].mean() - exact[i]) <= 3 * se
        print(f"  {t:<9}{samples[:, i].mean():>9.2%}{exact[i]:>9.2%}{'yes' if ok else 'NO':>15}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_commit_portfolios(args: argparse.Namespace) -> int:
    """Write the public pre-registration file for the portfolios (see commitments.py)."""
    config = load_config(args.config_dir) if args.config_dir else load_config()
    path = write_commitment(config.config_dir)
    print(f"Wrote {path.relative_to(config.config_dir.parent)}. Commit it to git before any hardware run.")
    print("Your private salt is in config/portfolio_salt.local.toml (gitignored). Keep a backup of it.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtr", description="Quantum Tail-Risk Observatory")
    parser.add_argument("--config-dir", default=None, help="folder with the TOML files (default: ./config)")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="load and summarize the config files")
    check.set_defaults(func=cmd_check)

    fetch = sub.add_parser("fetch", help="download, validate and cache prices; show crash rates")
    fetch.add_argument("--tier", default="n12", help="which tier to report on (default n12)")
    fetch.set_defaults(func=cmd_fetch)

    fit = sub.add_parser("fit", help="fit the crash tree for a tier and print it")
    fit.add_argument("--tier", default="n12")
    fit.add_argument("--layout", default="chow_liu", choices=["chow_liu", "logical", "native"],
                     help="chow_liu = best tree; logical = max 3 links per asset; native = chip-shaped (M5)")
    fit.add_argument("--save", default=None, help="write the fitted model JSON to this path")
    fit.set_defaults(func=cmd_fit)

    circ = sub.add_parser("circuit", help="show the crash-tree circuit and run it on a perfect simulator")
    circ.add_argument("--tier", default="n6")
    circ.add_argument("--shots", type=int, default=8000)
    circ.set_defaults(func=cmd_circuit)

    commit = sub.add_parser("commit-portfolios", help="pre-register portfolios (salted hash of private weights)")
    commit.set_defaults(func=cmd_commit_portfolios)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
