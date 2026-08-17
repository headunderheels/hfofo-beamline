"""Locate external reference-data files from the criggall/muon-cooling G4BL
checkout (initial.dat, ReferenceParticle_*.txt, etc.) -- none of this data
lives in this repo; it's used for calibration/validation against real G4BL
output and its location is inherently environment-specific.

Do not hardcode a single sandbox's path as the only place to look (an
earlier version of this project did exactly that in three different files
-- scripts/emittance_sandbox.py and two test files -- and it broke the
first time someone ran it outside that one sandbox). Use ``find_file``
here instead, with ``HFOFO_MUON_COOLING`` as the explicit override.
"""

from __future__ import annotations

import glob
import os


def find_file(filename: str, env_var: str = "HFOFO_MUON_COOLING") -> str:
    """Find ``filename`` (e.g. ``"initial.dat"``) somewhere under a
    muon-cooling checkout.

    Search order:
    1. ``env_var`` (default ``HFOFO_MUON_COOLING``), if set -- either the
       exact file path, or a directory to search under.
    2. A broad glob from the current working directory, its parents, and
       the user's home directory.

    Raises ``FileNotFoundError`` with an actionable message (how to set the
    env var) if nothing is found, rather than a bare "not found".
    """
    override = os.environ.get(env_var)
    if override:
        if os.path.isfile(override) and os.path.basename(override) == filename:
            return override
        if os.path.isdir(override) or os.path.isfile(override):
            search_root = override if os.path.isdir(override) else os.path.dirname(override)
            hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f"{env_var}={override!r} is set but no {filename!r} found there "
            f"(checked as an exact file, then as a directory to search under)."
        )

    search_roots = [".", "..", "../..", "../../..", os.path.expanduser("~")]
    for root in search_roots:
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"{filename!r} not found. This file lives in the criggall/muon-cooling "
        f"checkout, not in this repo -- point at it explicitly with, e.g.:\n"
        f"  {env_var}=/path/to/muon-cooling uv run python <script>\n"
        f"({env_var} can be the exact file path, or a directory to search under.)"
    )
