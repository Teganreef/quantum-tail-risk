"""Data pipeline tests. All offline: a fake downloader stands in for Yahoo Finance."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtr.config import DataSettings, Universe
from qtr.data import (
    SYNTHETIC_PREFIX,
    align_prices,
    build_price_data,
    compute_crash_data,
    data_fingerprint,
    make_synthetic_price_data,
    split_returns,
)

START, END = date(2021, 1, 1), date(2026, 1, 1)


def _series(ticker: str, days: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))), index=days, name=ticker)


def fake_downloader(overrides: dict[str, pd.Series | None] | None = None):
    """Stocks trade on weekdays, crypto every day. `overrides` replaces specific tickers."""
    overrides = overrides or {}

    def download(ticker: str, start: date, end: date) -> pd.Series:
        if ticker in overrides:
            s = overrides[ticker]
            return pd.Series(dtype=float, name=ticker) if s is None else s
        freq = "D" if ticker.endswith("-USD") else "B"
        days = pd.date_range(start, end, freq=freq, inclusive="left")
        return _series(ticker, days, seed=sum(map(ord, ticker)))

    return download


def _settings(**kw) -> DataSettings:
    base = dict(q=0.05, history_years=5, holdout_months=12, max_missing_frac=0.02, cache_dir="unused")
    base.update(kw)
    return DataSettings(**base)


def _universe(tickers=("AAA", "BBB", "BTC-USD", "CCC"), replacements=("RRR", "SSS")) -> Universe:
    return Universe(tiers={"n4": tuple(tickers)}, replacements=tuple(replacements), sectors={})


def _build(tmp_path: Path, universe: Universe, downloader) -> object:
    return build_price_data(universe, _settings(), end=END, downloader=downloader, cache_dir=tmp_path)


# --- alignment ------------------------------------------------------------

def test_crypto_weekend_folds_into_monday():
    """Fri -> Mon stock return and BTC's Fri -> Mon return (including the weekend) share one row."""
    fri, sat, sun, mon = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"])
    stock = pd.Series([100.0, 110.0], index=[fri, mon])
    btc = pd.Series([100.0, 90.0, 80.0, 120.0], index=[fri, sat, sun, mon])
    aligned = align_prices({"STK": stock, "BTC-USD": btc}, date(2024, 1, 1))
    assert list(aligned.index) == [fri, mon]  # weekend rows gone
    returns = aligned.pct_change().iloc[1:]
    assert returns.loc[mon, "BTC-USD"] == pytest.approx(0.20)  # 100 -> 120 over the whole weekend
    assert returns.loc[mon, "STK"] == pytest.approx(0.10)


def test_aligned_data_has_no_gaps_or_zero_fills(tmp_path):
    data = _build(tmp_path, _universe(), fake_downloader())
    assert not data.prices.isna().any().any()
    assert data.prices.index.dayofweek.max() <= 4  # weekends dropped because stocks don't trade
    assert (data.prices > 0).all().all()


# --- validation and replacement ---------------------------------------------

def test_gappy_ticker_is_dropped_and_replaced_in_place(tmp_path):
    days = pd.bdate_range(START, END, inclusive="left")
    gappy = _series("BBB", days, 1).iloc[::2]  # half the days missing
    data = _build(tmp_path, _universe(), fake_downloader({"BBB": gappy}))
    assert data.tickers == ("AAA", "RRR", "BTC-USD", "CCC")  # same position, same size
    assert any("DROPPED BBB" in line and "missing" in line for line in data.drop_log)
    assert "REPLACED BBB -> RRR" in data.drop_log


def test_short_history_is_dropped(tmp_path):
    days = pd.bdate_range("2023-06-01", END, inclusive="left")
    data = _build(tmp_path, _universe(), fake_downloader({"CCC": _series("CCC", days, 2)}))
    assert "CCC" not in data.tickers
    assert any("DROPPED CCC: history starts 2023-06-01" in line for line in data.drop_log)


def test_failed_download_skips_to_next_good_replacement(tmp_path):
    data = _build(tmp_path, _universe(), fake_downloader({"AAA": None, "RRR": None}))
    assert data.tickers[0] == "SSS"
    assert "REPLACED AAA -> SSS" in data.drop_log


def test_running_out_of_replacements_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="Ran out of replacement"):
        _build(tmp_path, _universe(replacements=()), fake_downloader({"AAA": None}))


# --- caching and fingerprint --------------------------------------------------

def test_second_build_uses_cache(tmp_path):
    calls = []
    inner = fake_downloader()

    def counting(t, s, e):
        calls.append(t)
        return inner(t, s, e)

    first = _build(tmp_path, _universe(), counting)
    n_calls = len(calls)
    second = _build(tmp_path, _universe(), counting)
    assert len(calls) == n_calls  # nothing re-downloaded
    assert first.data_hash == second.data_hash


def test_fingerprint_changes_when_data_changes():
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    changed = prices.copy()
    changed.iloc[1, 0] = 2.0001
    assert data_fingerprint(prices) == data_fingerprint(prices.copy())
    assert data_fingerprint(prices) != data_fingerprint(changed)


# --- holdout split and crash indicators --------------------------------------

def test_holdout_is_last_12_months_and_disjoint(tmp_path):
    data = _build(tmp_path, _universe(), fake_downloader())
    assert data.returns_fit.index.max() < data.returns_holdout.index.min()
    span = data.returns_holdout.index.max() - data.returns_holdout.index.min()
    assert pd.Timedelta(days=330) < span <= pd.Timedelta(days=366)


def test_crash_rate_close_to_q(tmp_path):
    data = _build(tmp_path, _universe(), fake_downloader())
    crashes = compute_crash_data(data, q=0.05)
    assert np.all(np.abs(crashes.crash_rate_fit - 0.05) <= 0.005)


def test_thresholds_ignore_the_holdout():
    """Wrecking the holdout must not move the crash thresholds (no peeking at the future)."""
    data = make_synthetic_price_data(("A", "B", "C"), seed=3)
    before = compute_crash_data(data, 0.05).thresholds
    wrecked = type(data)(
        prices=data.prices, returns_fit=data.returns_fit, returns_holdout=data.returns_holdout - 0.5,
        data_hash=data.data_hash, is_synthetic=True,
    )
    after = compute_crash_data(wrecked, 0.05)
    np.testing.assert_array_equal(before, after.thresholds)
    assert after.indicators_holdout.all()  # every wrecked holdout day is now a crash


def test_for_tier_takes_first_k_columns():
    data = make_synthetic_price_data(("A", "B", "C", "D"))
    small = data.for_tier(2)
    assert small.tickers == ("A", "B")
    assert small.data_hash == data.data_hash


def test_split_returns_cutoff():
    days = pd.bdate_range("2020-01-01", "2024-12-31")
    prices = pd.DataFrame({"A": np.linspace(100, 200, len(days))}, index=days)
    fit, holdout = split_returns(prices, holdout_months=12)
    assert holdout.index.min() > pd.Timestamp("2023-12-31")
    assert len(fit) + len(holdout) == len(days) - 1


# --- synthetic guard -----------------------------------------------------------

def test_synthetic_data_is_labeled():
    data = make_synthetic_price_data(("A", "B"))
    assert data.is_synthetic
    assert data.data_hash.startswith(SYNTHETIC_PREFIX)
    assert any("SYNTHETIC" in line for line in data.drop_log)
