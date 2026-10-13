"""Config loading: the settings files parse, tiers nest, and portfolios normalize correctly."""

from pathlib import Path

import pytest

from qtr.config import ConfigError, Portfolio, load_config, resolve_portfolio


def test_real_config_loads(config_copy: Path):
    config = load_config(config_copy)
    assert config.data.q == 0.05
    assert set(config.experiments) == {"E0", "E1", "E2"}


def test_tiers_are_nested_and_sized(config_copy: Path):
    tiers = load_config(config_copy).universe.tiers
    names = list(tiers)
    assert names == ["n6", "n12", "n24", "n48", "n96"]
    for smaller, bigger in zip(names, names[1:]):
        # Nested: the bigger tier starts with exactly the smaller tier, in the same order.
        assert tiers[bigger][: len(tiers[smaller])] == tiers[smaller]
    for name, tickers in tiers.items():
        assert len(tickers) == int(name[1:])


def test_every_universe_ticker_has_a_sector(config_copy: Path):
    universe = load_config(config_copy).universe
    missing = [t for t in universe.tiers["n96"] + universe.replacements if t not in universe.sectors]
    assert missing == []


def test_tier_size_mismatch_is_caught(config_copy: Path):
    path = config_copy / "universe.toml"
    path.write_text(path.read_text().replace('"NOW", "GOOG", "NRG"', '"NOW", "GOOG"'))
    with pytest.raises(ConfigError, match="should have 6"):
        load_config(config_copy)


def test_weights_normalize_and_drop_outside_tier():
    p = Portfolio("p", "", weights={"A": 2.0, "B": 1.0, "Z": 5.0})
    resolved = resolve_portfolio(p, ("A", "B", "C"))
    assert resolved.weights == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert resolved.dropped == ("Z",)


def test_equal_weight_tier():
    p = Portfolio("eq", "", equal_weight_tier=True)
    resolved = resolve_portfolio(p, ("A", "B", "C", "D"))
    assert resolved.weights == pytest.approx({t: 0.25 for t in "ABCD"})


def test_portfolio_with_no_tickers_in_tier_fails_loudly():
    with pytest.raises(ConfigError, match="no tickers in this tier"):
        resolve_portfolio(Portfolio("p", "", weights={"Z": 1.0}), ("A", "B"))


def test_local_portfolio_file_overrides_public_one(config_copy: Path):
    (config_copy / "portfolios.local.toml").write_text(
        '[portfolios.my_portfolio]\ndescription = "real"\nweights = { GOOG = 3, NOW = 1 }\n'
    )
    mine = load_config(config_copy).portfolios["my_portfolio"]
    assert mine.from_local_file
    assert mine.weights == {"GOOG": 3.0, "NOW": 1.0}


def test_negative_weight_rejected(config_copy: Path):
    (config_copy / "portfolios.local.toml").write_text(
        '[portfolios.bad]\nweights = { GOOG = -1 }\n'
    )
    with pytest.raises(ConfigError, match="non-positive"):
        load_config(config_copy)
