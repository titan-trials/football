"""
Project-local `.env` loader.

WHY THIS EXISTS RATHER THAN ENVIRONMENT VARIABLES
-------------------------------------------------
`baseball_predictor` and this project both want an Odds API key, and each
key has its own credit quota. With process environment variables the two
projects share one namespace, so `ODDS_API_KEY` means "whichever project set
it last", and on this machine that is baseball's. That is not a
hypothetical: a football command resolved baseball's key three times in a
row, and `setx` does not affect the shell you type it in, so the obvious fix
appeared not to work.

A file inside the project root cannot have that problem. `football_props/.env`
is football's by construction. Nothing about the shell, the order things were
installed, or which terminal is open can change that.

PRECEDENCE, highest first:

    1. .env in this project root        <- football's, unambiguously
    2. NFL_ODDS_API_KEY env var         <- football-specific by name
    3. .odds_api_key file               <- legacy, still read
    4. ODDS_API_KEY env var             <- SHARED; refused for real spends

A key named `ODDS_API_KEY` *inside this repo's .env* is football's key, and
is treated as such. Only the process-level variable of that name is suspect,
because only that one is shared with another project.

FORMAT
------
    # comments and blank lines are ignored
    NFL_ODDS_API_KEY=abc123
    ODDS_API_KEY = "also fine, quotes stripped"
    export FOO=bar        # leading `export` is tolerated

Values are NOT expanded or interpolated -- what is after the `=` is the
value, so a key containing `$` or `#` mid-string survives intact.

ENCODING
--------
Decoded with the same BOM-tolerant reader the key file uses, because
PowerShell's `>` redirection writes UTF-16 and that has now broken two files
in this project's lineage (baseball's requirements.txt, and a key file).
Write it with:

    Set-Content -Path .env -Value "NFL_ODDS_API_KEY=yourkey" -Encoding utf8

ONEDRIVE
--------
This repo lives under OneDrive, so `.env` syncs to the cloud and to every
device on the account. It is gitignored, which keeps it off GitHub, but
gitignored is not the same as private. That is a deliberate trade for having
one obvious place the key lives; if it ever matters, move the repo out of
OneDrive rather than scattering the secret.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

ENV_FILE = ".env"
_CACHE: Dict[str, Dict[str, str]] = {}


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _decode(raw: bytes) -> str:
    """BOM- and UTF-16-tolerant decode. See the module docstring."""
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            text = raw.decode(enc)
            if "\x00" not in text:
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_env(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        # Strip one matching pair of surrounding quotes, nothing else.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def load_env(root: Optional[str] = None, refresh: bool = False) -> Dict[str, str]:
    """
    Read `<root>/.env` into a dict. Cached per root; `refresh=True` re-reads.

    Does NOT write into os.environ. Mutating the process environment is how
    one project's key leaks into another's, which is the whole problem this
    file exists to solve.
    """
    root = root or project_root()
    if not refresh and root in _CACHE:
        return _CACHE[root]
    path = os.path.join(root, ENV_FILE)
    values: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            values = parse_env(_decode(f.read()))
    _CACHE[root] = values
    return values


def env_get(name: str, root: Optional[str] = None) -> Optional[str]:
    """`.env` first, then the process environment."""
    value = load_env(root).get(name)
    if value:
        return value.strip()
    value = os.environ.get(name, "").strip()
    return value or None


def env_path(root: Optional[str] = None) -> str:
    return os.path.join(root or project_root(), ENV_FILE)


def env_exists(root: Optional[str] = None) -> bool:
    return os.path.exists(env_path(root))
