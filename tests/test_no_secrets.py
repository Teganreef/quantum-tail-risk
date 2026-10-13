"""Guard for the public repo: no file that git would publish may contain a secret.

Checks every tracked or about-to-be-tracked file (git ls-files, honoring .gitignore)
for things that look like API keys, IBM Cloud CRNs, or private portfolio files.
"""

import re
import subprocess
from pathlib import Path

import pytest

from qtr.config import REPO_ROOT

# Patterns for things that must never be public. Deliberately broad: a false alarm
# costs a minute, a leaked key costs an account.
SECRET_PATTERNS = {
    "IBM Cloud CRN (instance id)": re.compile(r"crn:v1:bluemix:[a-z]+:[a-z0-9-]+:[^\s\"']*:a/[0-9a-f]{32}"),
    "token assignment": re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]"),
    "long hex/base64 blob": re.compile(r"['\"][A-Za-z0-9_\-]{40,}['\"]"),
}
SKIP_SUFFIXES = {".qpy", ".parquet", ".png", ".jpg", ".ico"}
FORBIDDEN_FILES = {"portfolios.local.toml", "portfolio_salt.local.toml", ".env", "qiskit-ibm.json"}


def _publishable_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not inside a git repository")
    return [REPO_ROOT / line for line in out.splitlines() if line]


def test_no_private_files_are_publishable():
    leaked = [str(p.relative_to(REPO_ROOT)) for p in _publishable_files() if p.name in FORBIDDEN_FILES]
    assert leaked == [], f"These private files are not gitignored: {leaked}"


def test_no_secret_looking_strings():
    hits = []
    for path in _publishable_files():
        if path.suffix in SKIP_SUFFIXES or not path.is_file() or path.name == Path(__file__).name:
            continue
        text = path.read_text(errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                line_text = text.splitlines()[line - 1]
                if "sha256" in line_text or "_hash" in line_text:
                    continue  # labeled fingerprints (one-way hashes) are meant to be public
                hits.append(f"{path.relative_to(REPO_ROOT)}:{line} looks like a {label}")
    assert hits == [], "Possible secrets found:\n" + "\n".join(hits)
