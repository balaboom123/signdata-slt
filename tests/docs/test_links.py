"""Check local Markdown links and GitHub-style heading anchors."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {".git", ".venv"}
INLINE_LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)",
)
HTML_LINK_RE = re.compile(
    r"<(?:a|img)\b[^>]*?\b(?:href|src)=[\"'](?P<target>[^\"']+)[\"']",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
    )


def _outside_fences(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\s*(```+|~~~+)", line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def _github_slug(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading).lower()
    heading = "".join(
        char for char in heading
        if char.isalnum() or char in {" ", "-", "_"}
    )
    return re.sub(r" +", "-", heading.strip())


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in _outside_fences(path.read_text(encoding="utf-8")).splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _github_slug(match.group(2))
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def _link_targets(text: str) -> list[str]:
    text = _outside_fences(text)
    targets = [match.group("target") for match in INLINE_LINK_RE.finditer(text)]
    targets.extend(match.group("target") for match in HTML_LINK_RE.finditer(text))
    return targets


def test_relative_links_and_anchors_exist() -> None:
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in markdown_files():
        for raw_target in _link_targets(source.read_text(encoding="utf-8")):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]

            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue

            relative_path = unquote(parsed.path)
            destination = (
                source if not relative_path else source.parent / relative_path
            ).resolve()
            try:
                destination.relative_to(REPO_ROOT.resolve())
            except ValueError:
                errors.append(f"{source.relative_to(REPO_ROOT)}: outside repo: {target}")
                continue

            if not destination.exists():
                errors.append(f"{source.relative_to(REPO_ROOT)}: missing: {target}")
                continue

            if parsed.fragment and destination.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(destination, _anchors(destination))
                anchor = unquote(parsed.fragment).lower()
                if anchor not in anchors:
                    errors.append(
                        f"{source.relative_to(REPO_ROOT)}: missing anchor: {target}"
                    )

    assert not errors, "Broken local Markdown links:\n" + "\n".join(errors)
