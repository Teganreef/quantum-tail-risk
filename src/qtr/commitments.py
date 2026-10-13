"""Pre-registration of portfolios: commit to them publicly before any hardware run.

Why: if the portfolios could change after seeing quantum results, someone could
(even unintentionally) pick the ones that make the results look interesting.
Committing first rules that out.

- The comparison list (config/portfolios.toml) is public, so we commit its sha256.
- The private weights (config/portfolios.local.toml) must stay secret, so we commit
  a SALTED hash: sha256(salt + file). The salt is 32 random bytes kept only on this
  laptop (config/portfolio_salt.local.toml, gitignored). Without the salt, nobody
  can test guesses of the weights against the public hash. With it, the owner can
  later prove the file is unchanged by revealing salt + file to whoever asks.

KEEP A BACKUP of the salt file: lose it and the commitment can't be verified.
"""

from __future__ import annotations

import hashlib
import secrets
import tomllib
from datetime import datetime, timezone
from pathlib import Path

COMMITMENT_FILE = "portfolio_commitment.toml"   # public, committed to git
SALT_FILE = "portfolio_salt.local.toml"          # private, gitignored (*.local.toml)
LOCAL_FILE = "portfolios.local.toml"
PUBLIC_FILE = "portfolios.toml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def salted_sha256(path: Path, salt_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(salt_hex) + path.read_bytes()).hexdigest()


def load_or_create_salt(config_dir: Path) -> str:
    """Return the private salt, creating it once (never overwritten, so old commitments stay checkable)."""
    path = config_dir / SALT_FILE
    if path.exists():
        with path.open("rb") as f:
            return tomllib.load(f)["salt"]
    salt = secrets.token_hex(32)
    path.write_text(
        "# PRIVATE (gitignored). Needed to prove the portfolio commitment later. BACK THIS UP.\n"
        f'salt = "{salt}"\n'
    )
    return salt


def current_fingerprints(config_dir: Path) -> dict:
    """What the commitment file would say right now."""
    public = config_dir / PUBLIC_FILE
    with public.open("rb") as f:
        names = sorted(tomllib.load(f).get("portfolios", {}))
    local = config_dir / LOCAL_FILE
    out = {
        "public_portfolios_sha256": sha256_file(public),
        "comparison_portfolios": names,
        "has_local_file": local.exists(),
    }
    if local.exists():
        out["local_portfolios_salted_sha256"] = salted_sha256(local, load_or_create_salt(config_dir))
    return out


def write_commitment(config_dir: Path, now: datetime | None = None) -> Path:
    """Write the public commitment file (to be committed to git before any hardware run)."""
    now = now or datetime.now(timezone.utc)
    fp = current_fingerprints(config_dir)
    lines = [
        "# Portfolio pre-registration. Written by `python -m qtr commit-portfolios`.",
        "# Commit this file BEFORE any hardware run. Hardware runs refuse if it no longer matches.",
        "# The local hash is salted with a private salt that never leaves the owner's laptop.",
        f'created_utc = "{now.strftime("%Y-%m-%dT%H:%M:%SZ")}"',
        f'public_portfolios_sha256 = "{fp["public_portfolios_sha256"]}"',
        "comparison_portfolios = [" + ", ".join(f'"{n}"' for n in fp["comparison_portfolios"]) + "]",
        f"has_local_file = {'true' if fp['has_local_file'] else 'false'}",
    ]
    if fp["has_local_file"]:
        lines.append(f'local_portfolios_salted_sha256 = "{fp["local_portfolios_salted_sha256"]}"')
    path = config_dir / COMMITMENT_FILE
    path.write_text("\n".join(lines) + "\n")
    return path


def verify_commitment(config_dir: Path) -> tuple[bool, str]:
    """Check the portfolio files still match the committed fingerprints."""
    path = config_dir / COMMITMENT_FILE
    if not path.exists():
        return False, "no portfolio commitment yet: run `python -m qtr commit-portfolios` and commit it"
    with path.open("rb") as f:
        committed = tomllib.load(f)
    now = current_fingerprints(config_dir)
    if committed["public_portfolios_sha256"] != now["public_portfolios_sha256"]:
        return False, "config/portfolios.toml changed since the commitment"
    if committed["has_local_file"] != now["has_local_file"]:
        return False, "the private portfolio file was added or removed since the commitment"
    if now["has_local_file"] and committed["local_portfolios_salted_sha256"] != now["local_portfolios_salted_sha256"]:
        return False, "config/portfolios.local.toml changed since the commitment"
    return True, f"portfolios match the commitment from {committed['created_utc']}"
