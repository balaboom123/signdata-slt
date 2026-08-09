"""Keep workstation-specific home paths out of repository documentation."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {".git", ".venv"}
LOCAL_HOME_RE = re.compile(r"/home/[A-Za-z0-9._-]+(?:/[^\s`)>\]]*)?")


def test_no_local_home_paths_in_docs() -> None:
    failures: list[str] = []
    for path in REPO_ROOT.rglob("*.md"):
        if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = LOCAL_HOME_RE.search(line)
            if match:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_number}: {match.group(0)}"
                )
    assert not failures, "Machine-local home paths found:\n" + "\n".join(failures)
