"""CSL manifest building."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .._shared.availability import apply_availability_policy_paths
from .source import (
    CSLSourceConfig,
    DEFAULT_FPS,
    SPLIT_I_TEST_SIGNERS,
    SPLIT_I_TRAIN_SIGNERS,
    SPLIT_II_TEST_SENTENCES,
    SPLIT_II_TRAIN_SENTENCES,
    VIDEO_EXTENSIONS,
    resolve_corpus_file,
    resolve_release_dir,
    resolve_video_dir,
)


def build(config, source: CSLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from CSL corpus file and video directory."""
    manifest_path = config.paths.manifest

    release_dir = resolve_release_dir(source, config)
    if not release_dir.exists():
        raise FileNotFoundError(
            f"CSL release directory not found: {release_dir!r}. "
            f"Run the download stage first or set release_dir / paths.videos."
        )

    corpus_path = resolve_corpus_file(source, release_dir, log)
    if corpus_path is None:
        raise FileNotFoundError(
            f"CSL corpus file not found under {release_dir}. "
            f"Set dataset.source.corpus_file explicitly."
        )

    video_dir = resolve_video_dir(release_dir, source)

    corpus = _parse_corpus(corpus_path, log)
    if corpus.empty:
        raise RuntimeError(
            f"CSL corpus file produced no rows: {corpus_path}. "
            f"Check the file format."
        )
    log.info("Loaded CSL corpus: %d rows from %s", len(corpus), corpus_path)

    custom_splits: Optional[Dict[str, str]] = None
    if source.split_spec_file and Path(source.split_spec_file).exists():
        custom_splits = _load_split_spec(source.split_spec_file)
        log.info(
            "Loaded custom split spec from %s (%d entries)",
            source.split_spec_file, len(custom_splits),
        )

    video_index = _build_video_index(video_dir, log)
    log.info("Discovered %d video files under %s", len(video_index), video_dir)

    rows: List[Dict] = []
    missing_videos = 0

    for _, corpus_row in corpus.iterrows():
        sentence_id = int(corpus_row["sentence_id"])
        signer_id = int(corpus_row["signer_id"])
        text = str(corpus_row.get("text", ""))

        matches = _find_videos(video_index, signer_id, sentence_id, video_dir)
        if not matches:
            matches = [("", 0)]
            missing_videos += 1

        for rel_path, variation_id in matches:
            sample_id = f"{signer_id:03d}_{sentence_id:03d}_{variation_id:02d}"

            if custom_splits is not None:
                split_label = custom_splits.get(sample_id, "unknown")
            else:
                split_label = _assign_split(signer_id, sentence_id, source.protocol)

            rows.append({
                "SAMPLE_ID": sample_id,
                "VIDEO_ID": sample_id,
                "REL_PATH": rel_path,
                "SPLIT": split_label,
                "TEXT": text,
                "SIGNER_ID": str(signer_id),
                "LANGUAGE": "zh",
                "FPS": DEFAULT_FPS,
            })

    if missing_videos:
        log.warning(
            "%d corpus entries had no matching video file in %s.",
            missing_videos, video_dir,
        )

    if not rows:
        raise RuntimeError("CSL build_manifest produced no rows. Check corpus file and video directory.")

    df = pd.DataFrame(rows)

    if source.split != "all":
        before = len(df)
        df = df[df["SPLIT"] == source.split].reset_index(drop=True)
        log.info("Filtered to split='%s': %d -> %d rows", source.split, before, len(df))

    df = apply_availability_policy_paths(
        df,
        base_dir=release_dir,
        policy=source.availability_policy,
        rel_path_col="REL_PATH",
    )

    canonical_columns = [
        "SAMPLE_ID", "VIDEO_ID", "REL_PATH", "SPLIT",
        "TEXT", "SIGNER_ID", "LANGUAGE", "FPS",
    ]
    ordered = [c for c in canonical_columns if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    return df


def _parse_corpus(corpus_path: Path, log: logging.Logger) -> pd.DataFrame:
    raw_lines = corpus_path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = [l for l in raw_lines if l.strip() and not l.startswith("#")]
    if not data_lines:
        return pd.DataFrame()

    sample = data_lines[0]
    delimiter = "\t" if "\t" in sample else None

    rows = []
    skipped = 0
    for line in data_lines:
        parts = line.split(delimiter) if delimiter else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            skipped += 1
            continue
        try:
            sentence_id = int(parts[0])
        except ValueError:
            skipped += 1
            continue

        if len(parts) == 2:
            signer_id = 0
            text = parts[1]
        elif len(parts) >= 3:
            try:
                signer_id = int(parts[1])
                text = " ".join(parts[2:])
            except ValueError:
                signer_id = 0
                text = " ".join(parts[1:])
        else:
            skipped += 1
            continue

        rows.append({"sentence_id": sentence_id, "signer_id": signer_id, "text": text})

    if skipped:
        log.warning("Skipped %d malformed lines in corpus file %s.", skipped, corpus_path)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_video_index(
    video_dir: Path,
    log: logging.Logger,
) -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
    index: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    video_dir_parent = video_dir.parent

    for ext in VIDEO_EXTENSIONS:
        for vpath in video_dir.rglob(f"*{ext}"):
            stem = vpath.stem
            parts = stem.split("_")
            rel_path = str(vpath.relative_to(video_dir_parent))

            signer_id = sentence_id = None
            variation_id = 0

            if len(parts) >= 3:
                try:
                    signer_id = int(parts[0])
                    sentence_id = int(parts[1])
                    variation_id = int(parts[2])
                except ValueError:
                    signer_id = sentence_id = None

            if signer_id is None and len(parts) == 2:
                try:
                    signer_id = int(parts[0])
                    sentence_id = int(parts[1])
                except ValueError:
                    pass

            if signer_id is None and len(parts) == 1:
                try:
                    sentence_id = int(parts[0])
                    signer_id = 0
                except ValueError:
                    pass

            if signer_id is None or sentence_id is None:
                log.debug("Could not parse signer/sentence from filename: %s", vpath.name)
                continue

            index.setdefault((signer_id, sentence_id), []).append((rel_path, variation_id))

    return index


def _find_videos(
    video_index: Dict,
    signer_id: int,
    sentence_id: int,
    video_dir: Path,
) -> List[Tuple[str, int]]:
    if signer_id != 0:
        return video_index.get((signer_id, sentence_id), [])
    results = []
    for (s_id, sent_id), entries in video_index.items():
        if sent_id == sentence_id:
            results.extend(entries)
    return results


def _assign_split(signer_id: int, sentence_id: int, protocol: str) -> str:
    if protocol == "split_i":
        if signer_id in SPLIT_I_TRAIN_SIGNERS:
            return "train"
        if signer_id in SPLIT_I_TEST_SIGNERS:
            return "test"
        return "unknown"
    if protocol == "split_ii":
        if sentence_id in SPLIT_II_TRAIN_SENTENCES:
            return "train"
        if sentence_id in SPLIT_II_TEST_SENTENCES:
            return "test"
        return "unknown"
    raise ValueError(
        f"Unknown CSL split protocol: {protocol!r}. Valid options: 'split_i', 'split_ii'."
    )


def _load_split_spec(spec_file: str) -> Dict[str, str]:
    df = pd.read_csv(spec_file, sep="\t", header=None, names=["sample_id", "split"])
    return dict(zip(df["sample_id"].astype(str), df["split"].astype(str)))
