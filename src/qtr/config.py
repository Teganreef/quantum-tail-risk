"""Load and validate the project's TOML settings.

Why this module exists: every number the pipeline uses (crash quantile, shots,
budget cap, seeds...) comes from `config/*.toml`, never from constants buried in
code. Each run saves a snapshot of these files, so any result can be traced back
to exactly the settings that produced it.

Everything here is a frozen dataclass: once loaded, settings can't be changed by
accident halfway through a run. Functions take the settings they need as explicit
arguments; there is no global "current config".
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# The repo root is two levels above this file: src/qtr/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or inconsistent."""


# ---------------------------------------------------------------------------
# Settings sections (one dataclass per [section] of default.toml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSettings:
    q: float
    history_years: int
    holdout_months: int
    max_missing_frac: float
    cache_dir: str


@dataclass(frozen=True)
class ModelSettings:
    laplace_alpha: float
    prob_clip: float
    degree_cap: int


@dataclass(frozen=True)
class LayoutSettings:
    mode: str
    fake_backend: str
    max_two_qubit_error: float
    max_readout_error: float
    anneal_iterations: int
    prim_seed_candidates: int


@dataclass(frozen=True)
class ExecutionSettings:
    shots: int
    baseline_samples: int
    ideal_max_qubits: int
    noisy_mps_above: int


@dataclass(frozen=True)
class MitigationSettings:
    shared_readout_calibration: bool
    use_mthree: bool


@dataclass(frozen=True)
class BudgetSettings:
    cap_minutes: float
    window_days: int
    overhead_factor: float
    per_pub_seconds: float
    fallback_rep_delay_s: float


@dataclass(frozen=True)
class MetricsSettings:
    bootstrap_resamples: int
    var_levels: tuple[float, ...]
    loss_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    tiers: tuple[str, ...]
    shots: int
    variants: tuple[str, ...]  # "mitigated" and/or "raw"
    cap_minutes: float
    lams: tuple[float, ...] = ()  # only E2 (interference dial) uses this


# ---------------------------------------------------------------------------
# Universe and portfolios
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    """Nested asset tiers. `tiers["n12"]` is the full 12-ticker list, in order."""

    tiers: dict[str, tuple[str, ...]]
    replacements: tuple[str, ...]
    sectors: dict[str, str]  # ticker -> sector name

    def tickers(self, tier: str) -> tuple[str, ...]:
        if tier not in self.tiers:
            raise ConfigError(f"Unknown tier '{tier}'. Known tiers: {list(self.tiers)}")
        return self.tiers[tier]

    def sector_of(self, ticker: str) -> str:
        return self.sectors.get(ticker, "Other")


PORTFOLIO_KINDS = ("personal", "benchmark", "anchor", "illustrative")


@dataclass(frozen=True)
class Portfolio:
    name: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)  # raw, un-normalized
    equal_weight_tier: bool = False
    kind: str = "personal"
    proxies: dict[str, str] = field(default_factory=dict)    # missing ticker -> labeled stand-in
    from_local_file: bool = False  # True if it came from portfolios.local.toml


@dataclass(frozen=True)
class ResolvedPortfolio:
    """A portfolio's weights restricted to one tier and normalized to sum to 1.

    Never use `weights` without also showing `dropped_weight` and `proxied`: the
    normalization hides how much of the real portfolio isn't modeled.
    """

    name: str
    kind: str
    weights: dict[str, float]
    dropped: tuple[str, ...]                 # tickers in the portfolio but not modeled in this tier
    dropped_weight: float                    # fraction (0-1) of the ORIGINAL weight that was dropped
    proxied: tuple[tuple[str, str], ...]     # (original, proxy) substitutions that were applied

    def coverage_note(self) -> str:
        """One-line, human-readable statement of what was dropped or substituted."""
        parts = []
        if self.dropped:
            parts.append(f"{self.dropped_weight:.1%} of weight dropped ({', '.join(self.dropped)})")
        parts += [f"{orig} -> {proxy} (proxy)" for orig, proxy in self.proxied]
        return "; ".join(parts) if parts else "fully modeled"


@dataclass(frozen=True)
class Config:
    data: DataSettings
    model: ModelSettings
    layout: LayoutSettings
    execution: ExecutionSettings
    mitigation: MitigationSettings
    budget: BudgetSettings
    metrics: MetricsSettings
    seed: int
    experiments: dict[str, ExperimentSpec]
    universe: Universe
    portfolios: dict[str, Portfolio]
    config_dir: Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_toml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"Could not parse {path.name}: {err}") from err


def _section(raw: dict, name: str, cls: type):
    """Build a settings dataclass from one [section], naming any missing/extra keys."""
    if name not in raw:
        raise ConfigError(f"default.toml is missing the [{name}] section")
    values = dict(raw[name])
    expected = set(cls.__dataclass_fields__)
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing or extra:
        raise ConfigError(
            f"[{name}] in default.toml: missing keys {sorted(missing)}, unknown keys {sorted(extra)}"
        )
    # TOML arrays come in as lists; store them as tuples so the dataclass stays immutable.
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**values)


def load_universe(path: Path) -> Universe:
    """Build cumulative tiers (each tier = previous tier + its `add` list) and validate them."""
    raw = _read_toml(path)
    tiers: dict[str, tuple[str, ...]] = {}
    running: list[str] = []
    for tier_name, spec in raw.get("tiers", {}).items():
        running = running + list(spec["add"])
        tiers[tier_name] = tuple(running)

    if not tiers:
        raise ConfigError("universe.toml defines no tiers")
    for tier_name, tickers in tiers.items():
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        if dupes:
            raise ConfigError(f"Tier {tier_name} lists these tickers twice: {dupes}")
        # Tier names encode their size (n6 -> 6 assets); catch edits that break that.
        if tier_name.startswith("n") and tier_name[1:].isdigit() and int(tier_name[1:]) != len(tickers):
            raise ConfigError(f"Tier {tier_name} should have {tier_name[1:]} tickers but has {len(tickers)}")

    sectors: dict[str, str] = {}
    for sector, members in raw.get("sectors", {}).items():
        for ticker in members:
            if ticker in sectors:
                raise ConfigError(f"{ticker} is listed under two sectors: {sectors[ticker]} and {sector}")
            sectors[ticker] = sector

    replacements = tuple(raw.get("replacements", {}).get("ordered", []))
    top = next(reversed(tiers.values()))
    clash = sorted(set(replacements) & set(top))
    if clash:
        raise ConfigError(f"Replacement tickers already in the universe: {clash}")
    return Universe(tiers=tiers, replacements=replacements, sectors=sectors)


def _parse_portfolios(raw: dict, from_local_file: bool) -> dict[str, Portfolio]:
    out: dict[str, Portfolio] = {}
    for name, spec in raw.get("portfolios", {}).items():
        weights = {str(k): float(v) for k, v in spec.get("weights", {}).items()}
        equal = bool(spec.get("equal_weight_tier", False))
        kind = spec.get("kind", "personal")
        if kind not in PORTFOLIO_KINDS:
            raise ConfigError(f"Portfolio '{name}' has kind '{kind}'; expected one of {PORTFOLIO_KINDS}")
        proxies = {str(k): str(v) for k, v in spec.get("proxies", {}).items()}
        stray = sorted(set(proxies) - set(weights))
        if stray:
            raise ConfigError(f"Portfolio '{name}' has proxies for tickers it doesn't hold: {stray}")
        if not equal and not weights:
            raise ConfigError(f"Portfolio '{name}' has no weights")
        bad = {t: w for t, w in weights.items() if w <= 0}
        if bad:
            raise ConfigError(f"Portfolio '{name}' has non-positive weights (long-only for now): {bad}")
        out[name] = Portfolio(
            name=name,
            description=spec.get("description", ""),
            weights=weights,
            equal_weight_tier=equal,
            kind=kind,
            proxies=proxies,
            from_local_file=from_local_file,
        )
    return out


def load_portfolios(config_dir: Path) -> dict[str, Portfolio]:
    """Public placeholders from portfolios.toml, overridden by the private local file if present."""
    portfolios = _parse_portfolios(_read_toml(config_dir / "portfolios.toml"), from_local_file=False)
    local_path = config_dir / "portfolios.local.toml"
    if local_path.exists():
        portfolios.update(_parse_portfolios(_read_toml(local_path), from_local_file=True))
    return portfolios


def load_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Config:
    """Load default.toml, universe.toml and portfolios(.local).toml into one Config."""
    config_dir = Path(config_dir)
    raw = _read_toml(config_dir / "default.toml")

    experiments: dict[str, ExperimentSpec] = {}
    for name, spec in raw.get("experiments", {}).items():
        experiments[name] = ExperimentSpec(
            name=name,
            description=spec["description"],
            tiers=tuple(spec["tiers"]),
            shots=int(spec["shots"]),
            variants=tuple(spec["variants"]),
            cap_minutes=float(spec["cap_minutes"]),
            lams=tuple(spec.get("lams", ())),
        )

    config = Config(
        data=_section(raw, "data", DataSettings),
        model=_section(raw, "model", ModelSettings),
        layout=_section(raw, "layout", LayoutSettings),
        execution=_section(raw, "execution", ExecutionSettings),
        mitigation=_section(raw, "mitigation", MitigationSettings),
        budget=_section(raw, "budget", BudgetSettings),
        metrics=_section(raw, "metrics", MetricsSettings),
        seed=int(raw["seeds"]["global"]),
        experiments=experiments,
        universe=load_universe(config_dir / "universe.toml"),
        portfolios=load_portfolios(config_dir),
        config_dir=config_dir,
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Cross-file checks that a single section can't do on its own."""
    if not 0 < config.data.q < 0.5:
        raise ConfigError(f"data.q must be between 0 and 0.5, got {config.data.q}")
    if config.layout.mode not in ("native", "logical"):
        raise ConfigError(f"layout.mode must be 'native' or 'logical', got '{config.layout.mode}'")
    for exp in config.experiments.values():
        unknown = [t for t in exp.tiers if t not in config.universe.tiers]
        if unknown:
            raise ConfigError(f"Experiment {exp.name} uses unknown tiers {unknown}")
        bad_variants = set(exp.variants) - {"mitigated", "raw"}
        if bad_variants:
            raise ConfigError(f"Experiment {exp.name} has unknown variants {sorted(bad_variants)}")


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def resolve_portfolio(portfolio: Portfolio, tier_tickers: tuple[str, ...]) -> ResolvedPortfolio:
    """Restrict a portfolio to the active tier's tickers and normalize weights to sum to 1.

    For each holding not modeled in this tier:
      - if the portfolio maps it to a proxy that IS in the tier, its weight moves to
        the proxy, and the substitution is recorded (reports label it);
      - otherwise it's dropped, and its share of the original weight is recorded.
    Dropping instead of failing lets the same portfolio be evaluated at every tier
    size; reporting `dropped_weight` keeps that from being silent.
    """
    if portfolio.equal_weight_tier:
        n = len(tier_tickers)
        return ResolvedPortfolio(portfolio.name, portfolio.kind, {t: 1.0 / n for t in tier_tickers},
                                 dropped=(), dropped_weight=0.0, proxied=())

    total_original = sum(portfolio.weights.values())
    kept: dict[str, float] = {}
    dropped: list[str] = []
    proxied: list[tuple[str, str]] = []
    for ticker, w in portfolio.weights.items():
        if ticker in tier_tickers:
            target = ticker
        elif portfolio.proxies.get(ticker) in tier_tickers:
            target = portfolio.proxies[ticker]
            proxied.append((ticker, target))
        else:
            dropped.append(ticker)
            continue
        kept[target] = kept.get(target, 0.0) + w
    if not kept:
        raise ConfigError(
            f"Portfolio '{portfolio.name}' has no tickers in this tier (it holds {list(portfolio.weights)})"
        )
    total_kept = sum(kept.values())
    # Keep tier order so weight vectors line up with asset columns everywhere else.
    weights = {t: kept[t] / total_kept for t in tier_tickers if t in kept}
    dropped_weight = sum(portfolio.weights[t] for t in dropped) / total_original
    return ResolvedPortfolio(portfolio.name, portfolio.kind, weights, tuple(dropped),
                             dropped_weight, tuple(proxied))
