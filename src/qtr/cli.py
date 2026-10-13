"""Command-line interface: `python -m qtr <command>`.

Commands are added milestone by milestone. Nothing here ever submits to real
quantum hardware unless the user runs `run-ibm --yes` and confirms interactively.
"""

from __future__ import annotations

import argparse

from qtr import DISCLAIMER
from qtr.config import load_config, resolve_portfolio
from qtr.data import build_price_data, compute_crash_data


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

    top_tier = next(reversed(config.universe.tiers.values()))
    print("\nPortfolios (resolved against the top tier):")
    for portfolio in config.portfolios.values():
        resolved = resolve_portfolio(portfolio, top_tier)
        source = "local file (private)" if portfolio.from_local_file else "portfolios.toml"
        print(f"  {portfolio.name}: {len(resolved.weights)} holdings, from {source}")
        if resolved.dropped:
            print(f"    WARNING: not in universe, dropped: {', '.join(resolved.dropped)}")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtr", description="Quantum Tail-Risk Observatory")
    parser.add_argument("--config-dir", default=None, help="folder with the TOML files (default: ./config)")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="load and summarize the config files")
    check.set_defaults(func=cmd_check)

    fetch = sub.add_parser("fetch", help="download, validate and cache prices; show crash rates")
    fetch.add_argument("--tier", default="n12", help="which tier to report on (default n12)")
    fetch.set_defaults(func=cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
