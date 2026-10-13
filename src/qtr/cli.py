"""Command-line interface: `python -m qtr <command>`.

Commands are added milestone by milestone. Nothing here ever submits to real
quantum hardware unless the user runs `run-ibm --yes` and confirms interactively.
"""

from __future__ import annotations

import argparse

from qtr import DISCLAIMER
from qtr.config import load_config, resolve_portfolio


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtr", description="Quantum Tail-Risk Observatory")
    parser.add_argument("--config-dir", default=None, help="folder with the TOML files (default: ./config)")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="load and summarize the config files")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
