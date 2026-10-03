#!/usr/bin/env python3
"""Run one atomic advanced CKD stage on top of a signed baseline stage.

The advanced stage never writes to a project-level path: every output lands in
``<artifact-root>/advanced/`` and is signed by ``advanced_manifest.json`` only after
all outputs validated and the baseline stage was verified unchanged.  The main random
seed is inherited from the baseline ``config.json``; ``--sensitivity-seeds`` lists the
extra seeds for the from-scratch seed-sensitivity replication.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from src.artifact_layout import ArtifactLayout


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sensitivity seeds must be integers") from exc
    if parsed < 0 or parsed > 2**32 - 1:
        raise argparse.ArgumentTypeError("sensitivity seeds must be between 0 and 2**32 - 1")
    return parsed


def _bootstrap(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bootstrap resamples must be an integer") from exc
    if parsed < 200:
        raise argparse.ArgumentTypeError("use at least 200 bootstrap resamples")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Isolated artifact root beneath this project's artifacts directory; must hold a signed baseline stage",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input CKD workbook beneath this project's data directory (must match the baseline stage)",
    )
    parser.add_argument(
        "--bootstrap",
        type=_bootstrap,
        default=1000,
        help="Conditional stratified-bootstrap resamples for the frozen finalist holdout stage",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=_seed,
        nargs="+",
        default=None,
        metavar="SEED",
        help="Distinct extra seeds under which split, registries, the 81-candidate grid, selection and "
        "holdout evaluation are replicated from scratch (default: src.seed_sensitivity.DEFAULT_SENSITIVITY_SEEDS); "
        "the main seed itself is inherited from the signed baseline config.json",
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
        from src.advanced_experiment import run_advanced_experiment

        run = run_advanced_experiment
    options = {} if args.sensitivity_seeds is None else {"sensitivity_seeds": tuple(args.sensitivity_seeds)}
    result = run(layout=layout, data_path=data_path, n_bootstrap=args.bootstrap, **options)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    main()
