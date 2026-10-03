#!/usr/bin/env python3
"""Run one atomic training-CV-only CKD baseline stage."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from src.artifact_layout import ArtifactLayout


def _cv_splits(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cv splits must be an integer") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("cv splits must be at least 2")
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if parsed < 0 or parsed > 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2**32 - 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Required isolated artifact root beneath this project's artifacts directory",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input CKD .xlsx/.xls/.csv/.arff file beneath this project's data directory",
    )
    parser.add_argument(
        "--cv-splits",
        type=_cv_splits,
        default=5,
        help="Training-only shuffled stratified CV folds; non-5 values are diagnostic",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional exact model-name filter; any filter makes the run diagnostic",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help="Random seed for the 70/30 split, CV registries and model factories "
        "(default: src.data_preprocessing.RANDOM_STATE); recorded in config.json",
    )
    return parser.parse_args(argv)


def _find_data_file(data_dir: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.xlsx", "*.xls", "*.csv", "*.arff"):
        candidates.extend(sorted(data_dir.glob(suffix)))
    if not candidates:
        raise FileNotFoundError(f"no CKD data file found beneath {data_dir}")
    preferred = [
        path for path in candidates
        if "kidney" in path.name.lower() or "ckd" in path.name.lower()
    ]
    return (preferred or candidates)[0]


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Callable[..., Path] | None = None,
    project_root: Path | None = None,
) -> Path:
    args = parse_args(argv)
    root = (
        Path(project_root).expanduser().resolve(strict=True)
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    layout = ArtifactLayout(project_root=root, artifact_root=args.artifact_root)
    selected_data = args.data if args.data is not None else _find_data_file(layout.data_dir)
    data_path = layout.validate_data_path(selected_data)

    if run is None:
        from src.baseline_experiment import run_baseline_experiment

        run = run_baseline_experiment
    options = {} if args.seed is None else {"random_state": args.seed}
    result = run(
        layout=layout,
        data_path=data_path,
        cv_splits=args.cv_splits,
        model_names=args.models,
        **options,
    )
    print(result, flush=True)
    return result


if __name__ == "__main__":
    main()
