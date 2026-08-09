"""Validate concrete configs and override keys in documented bash commands."""

from __future__ import annotations

import copy
import re
import shlex
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from signdata.config.experiment import load_experiment
from signdata.config.loader import _load_raw_config, _parse_value, _set_nested
from signdata.config.schema import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {".git", ".venv"}
BASH_BLOCK_RE = re.compile(r"^```bash\s*$\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
CONFIG_TOKEN_RE = re.compile(r"^configs/[A-Za-z0-9_./-]+\.ya?ml$")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
    )


def _logical_commands(block: str) -> list[str]:
    commands: list[str] = []
    current = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if current:
                commands.append(current)
                current = ""
            continue
        if line.endswith("\\"):
            current += line[:-1].rstrip() + " "
        else:
            current += line
            commands.append(current)
            current = ""
    if current:
        commands.append(current)
    return commands


def _contains_mapping(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return annotation is dict or origin is dict


def _models(annotation: Any) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        found.append(annotation)
    for argument in get_args(annotation):
        found.extend(_models(argument))
    return found


def _schema_accepts_override(key: str) -> bool:
    parts = key.split(".")
    models: list[type[BaseModel]] = [Config]
    for index, part in enumerate(parts):
        next_models: list[type[BaseModel]] = []
        mapping_accepts_remainder = False
        for model in models:
            field = model.model_fields.get(part)
            if field is None:
                continue
            if index == len(parts) - 1:
                return True
            mapping_accepts_remainder |= _contains_mapping(field.annotation)
            next_models.extend(_models(field.annotation))
        if mapping_accepts_remainder:
            return True
        if not next_models:
            return False
        models = next_models
    return False


def _override_tokens(tokens: list[str]) -> list[str]:
    if "--override" not in tokens:
        return []
    start = tokens.index("--override") + 1
    values: list[str] = []
    for token in tokens[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def test_documented_configs_and_overrides_validate() -> None:
    config_paths: set[Path] = set()
    override_count = 0

    for markdown in _markdown_files():
        text = markdown.read_text(encoding="utf-8")
        for block in BASH_BLOCK_RE.findall(text):
            for command in _logical_commands(block):
                tokens = shlex.split(command, comments=True, posix=True)
                paths = [
                    REPO_ROOT / token
                    for token in tokens
                    if CONFIG_TOKEN_RE.fullmatch(token)
                ]
                for path in paths:
                    assert path.exists(), (
                        f"{markdown.relative_to(REPO_ROOT)} references missing config "
                        f"{path.relative_to(REPO_ROOT)}"
                    )
                    config_paths.add(path)
                    if "configs/experiments" in path.as_posix():
                        load_experiment(str(path))
                    else:
                        Config(**_load_raw_config(str(path)))

                overrides = _override_tokens(tokens)
                override_count += len(overrides)
                for override in overrides:
                    assert "=" in override, (
                        f"{markdown.relative_to(REPO_ROOT)} has malformed override: "
                        f"{override}"
                    )
                    key, _ = override.split("=", 1)
                    assert _schema_accepts_override(key), (
                        f"{markdown.relative_to(REPO_ROOT)} uses unknown override key: "
                        f"{key}"
                    )

                job_paths = [
                    path for path in paths
                    if "configs/experiments" not in path.as_posix()
                ]
                if job_paths and overrides:
                    raw = copy.deepcopy(_load_raw_config(str(job_paths[0])))
                    for override in overrides:
                        key, value = override.split("=", 1)
                        _set_nested(raw, key, _parse_value(value))
                    Config(**raw)

    assert config_paths, "No concrete config paths found in documented bash blocks"
    assert override_count, "No overrides found in documented bash blocks"
