"""Ensure generated Markdown reference blocks match the code."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = ("configuration.md", "cli.md")


def test_generated_reference_is_current(tmp_path: Path) -> None:
    temporary_reference = tmp_path / "docs" / "reference"
    temporary_reference.mkdir(parents=True)

    for name in REFERENCE_FILES:
        shutil.copy2(REPO_ROOT / "docs" / "reference" / name, temporary_reference / name)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gen_docs_reference.py"),
            "--root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    stale = [
        name
        for name in REFERENCE_FILES
        if (temporary_reference / name).read_bytes()
        != (REPO_ROOT / "docs" / "reference" / name).read_bytes()
    ]
    assert not stale, (
        f"Generated reference files are stale: {', '.join(stale)}; "
        "run python scripts/gen_docs_reference.py"
    )
