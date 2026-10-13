"""Download, cache, align and validate prices; turn returns into crash indicators.

The pipeline in this module, in order:

1. Download each ticker's daily auto-adjusted closes from Yahoo Finance (or reuse
   the cached copy in data_cache/).
2. Validate: a ticker with too little history, or more than 2% of market days
   missing, is dropped and replaced by the next name on the replacement list.
3. Align: keep only dates where EVERY asset has a price. Crypto trades 7 days a
   week, stocks 5, so this keeps (roughly) stock trading days. A crypto move over
   the weekend lands in Monday's return, which is when a stock investor would feel it.
4. Returns: simple daily returns between consecutive aligned dates.
5. Split: the last 12 months become the holdout (an exam the model never sees);
   everything before is the fit window.
6. Crash indicators: for each asset, a day is a "crash" (1) if its return is at or
   below that asset's own 5% quantile, measured on the fit window only.

Nothing in this module fills a missing price with zero or with yesterday's
value. Missing data gets the ticker dropped, loudly.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from qtr.config import REPO_ROOT, DataSettings, Universe

log = logging.getLogger(__name__)

SYNTHETIC_PREFIX = "SYNTHETIC-"

# A downloader takes (ticker, start, end) and returns a Series of closes indexed by date
# (or an empty Series on failure). Passing it in as an argument lets tests run offline.
Downloader = Callable[[str, date, date], pd.Series]


@dataclass(frozen=True)
class PriceData:
    """Aligned prices and returns for the full top tier (smaller tiers = first k columns)."""

    prices: pd.DataFrame           # aligned closes, columns in universe order
    returns_fit: pd.DataFrame      # daily simple returns used to fit the model
    returns_holdout: pd.DataFrame  # most recent months, held out for the out-of-sample test
    data_hash: str                 # fingerprint of `prices`; recorded in every run
    is_synthetic: bool
    drop_log: tuple[str, ...] = field(default_factory=tuple)  # human-readable validation log

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.prices.columns)

    def for_tier(self, n_assets: int) -> "PriceData":
        """The same data restricted to the first n assets (a smaller nested tier)."""
        cols = list(self.prices.columns[:n_assets])
        return PriceData(
            prices=self.prices[cols],
            returns_fit=self.returns_fit[cols],
            returns_holdout=self.returns_holdout[cols],
            data_hash=self.data_hash,
            is_synthetic=self.is_synthetic,
            drop_log=self.drop_log,
        )


@dataclass(frozen=True)
class CrashData:
    """0/1 crash indicators. thresholds[i] is asset i's q-quantile return on the fit window."""

    tickers: tuple[str, ...]
    q: float
    thresholds: np.ndarray        # shape (n,)
    indicators_fit: np.ndarray    # shape (T_fit, n), uint8
    indicators_holdout: np.ndarray  # shape (T_holdout, n), uint8, using the FIT thresholds
    dates_fit: pd.DatetimeIndex
    dates_holdout: pd.DatetimeIndex

    @property
    def crash_rate_fit(self) -> np.ndarray:
        return self.indicators_fit.mean(axis=0)


# ---------------------------------------------------------------------------
# Downloading and caching
# ---------------------------------------------------------------------------

def yahoo_downloader(ticker: str, start: date, end: date) -> pd.Series:
    """Daily auto-adjusted closes from Yahoo Finance, indexed by (timezone-free) date.

    Auto-adjusted means splits and dividends are folded into the price history, so
    a 2-for-1 split doesn't look like a 50% crash.
    """
    import yfinance as yf  # imported here so tests and offline code never need it

    try:
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
    except Exception as err:  # yfinance raises many different error types
        log.warning("Download failed for %s: %s", ticker, err)
        return pd.Series(dtype=float, name=ticker)
    if hist.empty or "Close" not in hist:
        return pd.Series(dtype=float, name=ticker)
    closes = hist["Close"].dropna()
    # Yahoo stamps stocks in New York time and crypto in UTC. Dropping the timezone
    # and the time of day leaves the calendar date each price belongs to.
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes.rename(ticker)


def _cache_path(cache_dir: Path, ticker: str, start: date, end: date) -> Path:
    return cache_dir / f"{ticker}_{start.isoformat()}_{end.isoformat()}.parquet"


def load_or_download(
    ticker: str, start: date, end: date, cache_dir: Path, downloader: Downloader
) -> pd.Series:
    """Return a ticker's closes, from the cache if this exact window was fetched before."""
    path = _cache_path(cache_dir, ticker, start, end)
    if path.exists():
        return pd.read_parquet(path)[ticker]
    closes = downloader(ticker, start, end)
    if not closes.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        closes.to_frame(ticker).to_parquet(path)
    return closes


# ---------------------------------------------------------------------------
# Validation and alignment
# ---------------------------------------------------------------------------

def is_crypto(ticker: str) -> bool:
    return ticker.endswith("-USD")


def market_days(series_by_ticker: dict[str, pd.Series], start: date) -> pd.DatetimeIndex:
    """Dates when at least half of the non-crypto tickers traded (DECISIONS D24)."""
    stocks = [s for t, s in series_by_ticker.items() if not is_crypto(t) and not s.empty]
    if not stocks:
        raise ValueError("Need at least one non-crypto ticker to define market days")
    counts = pd.concat([s.notna().rename(i) for i, s in enumerate(stocks)], axis=1).sum(axis=1)
    days = counts.index[counts >= len(stocks) / 2]
    return pd.DatetimeIndex(days[days >= pd.Timestamp(start)]).sort_values()


def validate_ticker(
    ticker: str, closes: pd.Series, days: pd.DatetimeIndex, start: date, max_missing_frac: float
) -> str | None:
    """Return None if the ticker is usable, else a plain-English reason it isn't."""
    if closes.empty:
        return "no data downloaded"
    # Allow a week of slack: the window may start on a weekend or holiday.
    if closes.index.min() > pd.Timestamp(start + timedelta(days=7)):
        return f"history starts {closes.index.min().date()}, after the window start {start}"
    missing = 1.0 - days.isin(closes.index).mean()
    if missing > max_missing_frac:
        return f"{missing:.1%} of market days missing (limit {max_missing_frac:.0%})"
    return None


def select_valid_universe(
    wanted: tuple[str, ...],
    replacements: tuple[str, ...],
    fetch: Callable[[str], pd.Series],
    start: date,
    max_missing_frac: float,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Fetch and validate tickers, swapping failures for replacements in place (DECISIONS D23).

    Returns the kept series (in final universe order) and a log of every drop and swap.
    """
    series = {t: fetch(t) for t in wanted}
    days = market_days(series, start)
    log_lines: list[str] = []
    spare = list(replacements)
    final: dict[str, pd.Series] = {}

    for ticker in wanted:
        candidate = ticker
        reason = validate_ticker(candidate, series[candidate], days, start, max_missing_frac)
        while reason is not None:
            log_lines.append(f"DROPPED {candidate}: {reason}")
            if not spare:
                raise ValueError(f"Ran out of replacement tickers while replacing {ticker}; add more to universe.toml")
            candidate = spare.pop(0)
            series[candidate] = fetch(candidate)
            reason = validate_ticker(candidate, series[candidate], days, start, max_missing_frac)
            if reason is None:
                log_lines.append(f"REPLACED {ticker} -> {candidate}")
        final[candidate] = series[candidate]
    return final, log_lines


def align_prices(series_by_ticker: dict[str, pd.Series], start: date) -> pd.DataFrame:
    """Keep only dates where every asset has a price (see module docstring, step 3)."""
    frame = pd.concat(series_by_ticker, axis=1)
    frame = frame[frame.index >= pd.Timestamp(start)].sort_index()
    return frame.dropna(how="any")


def data_fingerprint(prices: pd.DataFrame) -> str:
    """sha256 of the aligned prices as fixed-format CSV (DECISIONS D4).

    Hashing the values (not the parquet bytes) means the fingerprint only changes
    when the data changes, not when a library version changes file metadata.
    """
    csv = prices.to_csv(float_format="%.10g", date_format="%Y-%m-%d")
    return hashlib.sha256(csv.encode()).hexdigest()


def split_returns(prices: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple returns, split into (fit, holdout) at `holdout_months` before the last date."""
    returns = prices.pct_change().iloc[1:]
    cutoff = returns.index[-1] - pd.DateOffset(months=holdout_months)
    return returns[returns.index <= cutoff], returns[returns.index > cutoff]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def window_start(end: date, history_years: int) -> date:
    try:
        return end.replace(year=end.year - history_years)
    except ValueError:  # Feb 29 -> Feb 28
        return end.replace(year=end.year - history_years, day=28)


def build_price_data(
    universe: Universe,
    settings: DataSettings,
    end: date | None = None,
    downloader: Downloader = yahoo_downloader,
    cache_dir: Path | None = None,
) -> PriceData:
    """Build the aligned dataset for the full top tier (DECISIONS D22)."""
    end = end or date.today()
    start = window_start(end, settings.history_years)
    cache_dir = cache_dir or (REPO_ROOT / settings.cache_dir)
    top_tier = next(reversed(universe.tiers.values()))

    def fetch(ticker: str) -> pd.Series:
        return load_or_download(ticker, start, end, cache_dir, downloader)

    series, drop_log = select_valid_universe(
        top_tier, universe.replacements, fetch, start, settings.max_missing_frac
    )
    prices = align_prices(series, start)
    fit, holdout = split_returns(prices, settings.holdout_months)
    return PriceData(
        prices=prices,
        returns_fit=fit,
        returns_holdout=holdout,
        data_hash=data_fingerprint(prices),
        is_synthetic=False,
        drop_log=tuple(drop_log),
    )


def compute_crash_data(price_data: PriceData, q: float) -> CrashData:
    """Mark crash days: return <= the asset's own q-quantile on the FIT window.

    `method="lower"` picks an actual observed return as the threshold, and `<=`
    then counts that day as a crash, which gives a crash rate very close to q
    (DECISIONS D11). The holdout reuses the fit thresholds: the model must not peek
    at the future to decide what a crash is.
    """
    fit = price_data.returns_fit.to_numpy()
    holdout = price_data.returns_holdout.to_numpy()
    thresholds = np.quantile(fit, q, axis=0, method="lower")
    return CrashData(
        tickers=price_data.tickers,
        q=q,
        thresholds=thresholds,
        indicators_fit=(fit <= thresholds).astype(np.uint8),
        indicators_holdout=(holdout <= thresholds).astype(np.uint8),
        dates_fit=price_data.returns_fit.index,
        dates_holdout=price_data.returns_holdout.index,
    )


def make_synthetic_price_data(
    tickers: tuple[str, ...], n_days: int = 1300, holdout_days: int = 250, seed: int = 0
) -> PriceData:
    """Fake but realistic-looking prices, for tests only. Clearly labeled SYNTHETIC.

    Returns come from one shared "market" factor plus noise, with fat tails, so
    assets crash together sometimes. Hardware submission refuses any run whose
    data_hash starts with SYNTHETIC- (spec section 5).
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    market = rng.standard_t(df=4, size=n_days) * 0.01
    loadings = rng.uniform(0.3, 1.5, size=n)
    noise = rng.standard_t(df=4, size=(n_days, n)) * 0.012
    returns = market[:, None] * loadings[None, :] + noise
    dates = pd.bdate_range("2020-01-01", periods=n_days + 1)
    prices = pd.DataFrame(
        100 * np.cumprod(np.vstack([np.ones(n), 1 + returns]), axis=0), index=dates, columns=list(tickers)
    )
    rets = prices.pct_change().iloc[1:]
    return PriceData(
        prices=prices,
        returns_fit=rets.iloc[:-holdout_days],
        returns_holdout=rets.iloc[-holdout_days:],
        data_hash=SYNTHETIC_PREFIX + data_fingerprint(prices),
        is_synthetic=True,
        drop_log=("SYNTHETIC DATA: generated for tests, not market data",),
    )
