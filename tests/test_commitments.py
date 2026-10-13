"""Portfolio pre-registration: salted commitment detects any change, reveals nothing."""

from pathlib import Path

from qtr.commitments import (
    COMMITMENT_FILE,
    SALT_FILE,
    load_or_create_salt,
    salted_sha256,
    verify_commitment,
    write_commitment,
)

LOCAL = '[portfolios.my_portfolio]\nweights = { GOOG = 60, NOW = 40 }\n'


def test_no_commitment_means_not_ready(config_copy: Path):
    ok, msg = verify_commitment(config_copy)
    assert not ok and "commit-portfolios" in msg


def test_commitment_matches_then_detects_changes(config_copy: Path):
    local = config_copy / "portfolios.local.toml"
    local.write_text(LOCAL)
    write_commitment(config_copy)
    assert verify_commitment(config_copy)[0]

    local.write_text(LOCAL.replace("60", "61"))           # tweak the private weights
    ok, msg = verify_commitment(config_copy)
    assert not ok and "portfolios.local.toml changed" in msg

    local.write_text(LOCAL)
    public = config_copy / "portfolios.toml"
    public.write_text(public.read_text() + "\n# edit\n")  # tweak the public list
    assert "portfolios.toml changed" in verify_commitment(config_copy)[1]


def test_commitment_file_reveals_no_weights(config_copy: Path):
    (config_copy / "portfolios.local.toml").write_text(LOCAL)
    text = (write_commitment(config_copy)).read_text()
    assert "GOOG" not in text and "60" not in text.split("salted_sha256")[-1][:5]
    assert "salt =" not in text


def test_salt_is_stable_and_actually_salts(config_copy: Path):
    local = config_copy / "portfolios.local.toml"
    local.write_text(LOCAL)
    salt = load_or_create_salt(config_copy)
    assert load_or_create_salt(config_copy) == salt          # never regenerated
    assert len(salt) == 64
    assert salted_sha256(local, salt) != salted_sha256(local, "00" * 32)
    assert (config_copy / SALT_FILE).exists() and (config_copy / COMMITMENT_FILE).exists() is False
