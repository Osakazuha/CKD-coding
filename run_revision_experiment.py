#!/usr/bin/env python3
"""Run one atomic revision stage on top of signed baseline and advanced stages.

The revision stage holds the post-freeze sensitivity analyses (nested
resampling of the selection procedure, multiple imputation and missingness
mechanism, feature-selection stability, diagnostic-predictor removal, uniform
class-weighting policies, naive versus leakage-controlled grids).  Every output
lands in ``<artifact-root>/revision/`` and is signed by ``revision_manifest.json``
only after all tables validated.  The main seed and the sensitivity seeds are
inherited from the signed stages; nothing here can change the frozen selection.

``--smoke`` runs a shortened, non-release-eligible version (two outer folds, two
imputations, few bootstrap resamples) for an end-to-end check; use it only in a
scratch artifact root, because the stage directory is immutable once written.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from src.artifact_layout import ArtifactLayout


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, required=True,
        help="Isolated artifact root beneath this project's artifacts directory; must hold signed baseline and advanced stages",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Input CKD workbook beneath this project's data directory (must match the signed stages)",
    )
    parser.add_argument(
        "--imputations", type=_positive, default=None,
        help="Number of multiple imputations per training fold (default 20; a release run requires at least 20)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Shortened non-release run: two outer folds in the nested analysis, two imputations, 50 bootstrap resamples",
    )
    return parser.parse_args(argv)


def _find_data_file(data_dir: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.xlsx", "*.xls", "*.csv", "*.arff"):
        candidates.extend(sorted(data_dir.glob(suffix)))
    if not candidates:
        raise FileNotFoundError(f"no CKD data file found beneath {data_dir}")
    preferred = [path for path in candidates if "kidney" in path.name.lower() or "ckd" in path.name.lower()]
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
        from src.revision_experiment import run_revision_experiment

        run = run_revision_experiment
    options: dict[str, object] = {}
    if args.smoke:
        options.update({"smoke": True, "nested_outer_limit": 2, "n_imputations": 2, "stability_resamples": 50})
    if args.imputations is not None:
        options["n_imputations"] = args.imputations
    result = run(layout=layout, data_path=data_path, **options)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    main()
