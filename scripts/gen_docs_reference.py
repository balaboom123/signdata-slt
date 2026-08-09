#!/usr/bin/env python3
"""Regenerate code-derived Markdown reference sections in place."""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from signdata import cli  # noqa: E402
from signdata.config import schema  # noqa: E402
from signdata.processors.pose import list_presets  # noqa: E402


SECTION_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("config", schema.Config),
    ("paths", schema.PathsConfig),
    ("dataset", schema.DatasetConfig),
    ("processing", schema.ProcessingConfig),
    ("processing.detection_config.yolo", schema.YOLODetectionConfig),
    ("processing.detection_config.mmdet", schema.MMDetDetectionConfig),
    ("processing.detection_config.mediapipe", schema.MediaPipeDetectionConfig),
    ("processing.pose_config.mediapipe", schema.MediaPipePoseConfig),
    ("processing.pose_config.mmpose", schema.MMPosePoseConfig),
    ("processing.video_config", schema.VideoProcessingConfig),
    ("processing.parts_config", schema.PartsProcessingConfig),
    ("post_processing", schema.PostProcessingConfig),
    ("post_processing.normalize", schema.NormalizeConfig),
    ("output", schema.OutputConfig),
)


def _models_in_annotation(annotation: Any) -> set[type[BaseModel]]:
    models: set[type[BaseModel]] = set()
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        models.add(annotation)
    for argument in get_args(annotation):
        models.update(_models_in_annotation(argument))
    return models


def _reachable_models(root: type[BaseModel]) -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()
    pending = [root]
    while pending:
        model = pending.pop()
        if model in found:
            continue
        found.add(model)
        for field in model.model_fields.values():
            pending.extend(_models_in_annotation(field.annotation) - found)
    return found


def _literal(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, BaseModel):
        if isinstance(value, schema.ProcessingConfig) and not value.enabled:
            return "ProcessingConfig(enabled=false)"
        return f"{type(value).__name__}()"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _format_type(annotation: Any) -> str:
    if annotation is Any:
        return "Any"
    if annotation is type(None):
        return "null"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return " | ".join(_literal(argument) for argument in arguments)
    if origin in (Union, types.UnionType):
        return " | ".join(_format_type(argument) for argument in arguments)
    if origin is not None:
        name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
        if arguments:
            return f"{name}[{', '.join(_format_type(arg) for arg in arguments)}]"
        return name
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _format_default(field) -> str:
    if field.is_required():
        return "required"
    value = field.get_default(call_default_factory=True)
    return _literal(value)


def _format_constraints(field) -> str:
    constraints: list[str] = []
    labels = (
        ("gt", ">"),
        ("ge", "≥"),
        ("lt", "<"),
        ("le", "≤"),
        ("multiple_of", "multiple of"),
        ("min_length", "minimum length"),
        ("max_length", "maximum length"),
        ("pattern", "pattern"),
    )
    for metadata in field.metadata:
        for attribute, label in labels:
            value = getattr(metadata, attribute, None)
            if value is not None:
                constraints.append(f"{label} {_literal(value)}")
    return ", ".join(constraints) if constraints else "—"


def _cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _render_model(model: type[BaseModel]) -> str:
    lines = [
        "| Field | Type | Default | Constraints | Description |",
        "|---|---|---|---|---|",
    ]
    for name, field in model.model_fields.items():
        if not field.description:
            raise ValueError(
                f"{model.__name__}.{name} has no Field(description=...)."
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_cell(name)}`",
                    f"`{_cell(_format_type(field.annotation))}`",
                    f"`{_cell(_format_default(field))}`",
                    _cell(_format_constraints(field)),
                    _cell(field.description),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _argument_label(action: argparse.Action) -> str:
    if action.option_strings:
        label = ", ".join(action.option_strings)
    else:
        label = str(action.metavar or action.dest.upper())

    if action.nargs == 0:
        return label

    metavar = str(action.metavar or action.dest.upper())
    if action.nargs == "*":
        value = f"[{metavar} ...]"
    elif action.nargs == "+":
        value = f"{metavar} [{metavar} ...]"
    elif action.nargs == "?":
        value = f"[{metavar}]"
    else:
        value = metavar
    return f"{label} {value}" if action.option_strings else value


def _argument_default(action: argparse.Action) -> str:
    if action.default == argparse.SUPPRESS or action.dest == "help":
        return "—"
    if action.required and not action.option_strings:
        return "—"
    return _literal(action.default)


def _render_cli() -> str:
    parser = cli.build_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command_help = {
        action.dest: action.help for action in subparsers_action._choices_actions
    }

    lines = [
        "## Commands",
        "",
        "| Command | Description |",
        "|---|---|",
    ]
    for name in subparsers_action.choices:
        lines.append(f"| `{name}` | {_cell(command_help.get(name, ''))} |")

    for name, subparser in subparsers_action.choices.items():
        lines.extend(
            [
                "",
                f"## `{name}`",
                "",
                "```text",
                subparser.format_usage().strip(),
                "```",
                "",
                "| Argument | Required | Default | Description |",
                "|---|---:|---|---|",
            ]
        )
        for action in subparser._actions:
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{_cell(_argument_label(action))}`",
                        "yes" if action.required else "no",
                        f"`{_cell(_argument_default(action))}`"
                        if _argument_default(action) != "—"
                        else "—",
                        _cell(action.help or ""),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Keypoint Presets",
            "",
            "| Name | Description |",
            "|---|---|",
        ]
    )
    for name, description in sorted(list_presets().items()):
        lines.append(f"| `{name}` | {_cell(description)} |")
    return "\n".join(lines)


def _replace_generated(path: Path, key: str, body: str) -> None:
    begin = f"<!-- BEGIN GENERATED: {key} -->"
    end = f"<!-- END GENERATED: {key} -->"
    text = path.read_text(encoding="utf-8")
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(
            f"{path}: expected exactly one marker pair for {key!r}"
        )
    start = text.index(begin)
    finish = text.index(end)
    if finish <= start:
        raise ValueError(f"{path}: malformed marker order for {key!r}")

    generated = f"{begin}\n{body.rstrip()}\n{end}"
    updated = text[:start] + generated + text[finish + len(end):]
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def generate(root: Path) -> None:
    mapped = {model for _, model in SECTION_MODELS}
    reachable = _reachable_models(schema.Config)
    if mapped != reachable:
        missing = sorted(model.__name__ for model in reachable - mapped)
        extra = sorted(model.__name__ for model in mapped - reachable)
        raise ValueError(
            f"SECTION_MODELS does not match Config reachability; "
            f"missing={missing}, extra={extra}"
        )

    config_path = root / "docs" / "reference" / "configuration.md"
    cli_path = root / "docs" / "reference" / "cli.md"
    for key, model in SECTION_MODELS:
        _replace_generated(config_path, key, _render_model(model))
    _replace_generated(cli_path, "cli", _render_cli())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing docs/reference (default: this checkout)",
    )
    args = parser.parse_args(argv)
    generate(args.root.resolve())


if __name__ == "__main__":
    main()
