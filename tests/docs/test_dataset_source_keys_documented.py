"""Pin shipped dataset.source keys to the manual configuration table."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dataset_source_keys_are_documented() -> None:
    configured: set[str] = set()
    for path in (REPO_ROOT / "configs").rglob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dataset = raw.get("dataset", {})
        if isinstance(dataset, dict) and isinstance(dataset.get("source"), dict):
            configured.update(dataset["source"])

    reference = (REPO_ROOT / "docs" / "reference" / "configuration.md").read_text(
        encoding="utf-8"
    )
    source_section = reference.split("## `dataset.source`", 1)[1].split(
        "## `processing`", 1
    )[0]
    table_fields = set(re.findall(r"^\| `([^`]+)`", source_section, re.MULTILINE))
    documented: set[str] = set()
    for field in table_fields:
        documented.update(part.strip(" `") for part in field.split("/"))

    missing = configured - documented
    assert not missing, (
        "dataset.source keys missing from docs/reference/configuration.md: "
        + ", ".join(sorted(missing))
    )
