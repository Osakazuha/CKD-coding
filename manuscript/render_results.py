#!/usr/bin/env python3
"""Build the run-local publication stage: figures, evidence ledger, DOCX and PDF.

Everything is rendered from one artifact root's signed baseline and advanced
stages.  The only hand-written input is the Markdown manuscript source; every
numeric claim it makes is listed in the evidence ledger and must appear verbatim
in its declared section, otherwise the stage is not built.

    python manuscript/render_results.py --artifact-root artifacts/<run>            # build + sign
    python manuscript/render_results.py --artifact-root artifacts/<run> --check    # ledger check only

Both forms exit with status 1 (and print the offending claims) when any ledger value
is missing from its declared section, so shell guards such as ``--check || exit 1``
behave as expected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.paths import MANUSCRIPT_NAME, SOURCE  # noqa: E402
from manuscript.evidence import (  # noqa: E402
    LEDGER_FIELDS,
    build_evidence_ledger,
    check_manuscript_against_ledger,
    fill_ledger_tokens,
    write_evidence_files,
)
from manuscript.publication_figures import build_publication_figures  # noqa: E402
from src.artifact_layout import ArtifactLayout, sha256_file  # noqa: E402


FIGURE_FILES = [f"figure_{i}.{ext}" for i in (1, 2, 3, 4, 5, "s1") for ext in ("png", "pdf")]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage-name", default="publication", help="publication or publication-vN for a revision")
    parser.add_argument("--check", action="store_true", help="only check the manuscript against the ledger")
    parser.add_argument("--skip-pdf", action="store_true", help="development only: do not require LibreOffice")
    return parser.parse_args(argv)


def _report_mismatches(mismatches: list[dict[str, str]]) -> str:
    lines = [f"{len(mismatches)} ledger value(s) are not present in their manuscript section:"]
    for row in mismatches:
        lines.append(f"  - {row['claim_id']}: expected {row['value']!r} in section {row['manuscript_section']!r} ({row['problem']})")
    return "\n".join(lines)


def convert_to_pdf(docx: Path, output_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) is required to render the PDF")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                soffice, f"-env:UserInstallation=file://{profile}", "--headless",
                "--convert-to", "pdf", "--outdir", str(output_dir), str(docx),
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
    pdf = output_dir / f"{docx.stem}.pdf"
    if not pdf.is_file() or pdf.stat().st_size < 20_000:
        raise RuntimeError("PDF conversion did not produce a usable file")
    return pdf


def main(argv=None) -> Path | None:
    args = parse_args(argv)
    layout = ArtifactLayout(project_root=PROJECT_ROOT, artifact_root=args.artifact_root)
    layout.verify_stage("baseline")
    layout.verify_stage("advanced")
    rows = build_evidence_ledger(layout.artifact_root)
    text = fill_ledger_tokens(SOURCE.read_text(encoding="utf-8"), rows)
    if "{{" in text:
        raise SystemExit("manuscript still contains unfilled placeholders")
    mismatches = check_manuscript_against_ledger(text, rows)
    if args.check:
        if mismatches:
            print(_report_mismatches(mismatches), file=sys.stderr)
            raise SystemExit(1)
        print(f"all {len(rows)} ledger values present in their declared sections")
        return None
    if mismatches:
        raise SystemExit(_report_mismatches(mismatches))

    workspace = layout.begin_stage(args.stage_name)
    stage = workspace.partial_dir
    figures = build_publication_figures(layout.artifact_root, stage / "publication_figures")
    write_evidence_files(layout.artifact_root, stage)
    manuscript_md = stage / f"{MANUSCRIPT_NAME}.md"
    manuscript_md.write_text(text, encoding="utf-8")
    from manuscript.build_publication_style import build as build_docx

    docx = build_docx(manuscript_md, stage / f"{MANUSCRIPT_NAME}.docx")
    required = [
        f"{MANUSCRIPT_NAME}.md", f"{MANUSCRIPT_NAME}.docx", "evidence_ledger.csv", "source_hashes.json",
        *(f"publication_figures/{name}" for name in FIGURE_FILES),
    ]
    if not args.skip_pdf:
        convert_to_pdf(docx, stage)
        required.append(f"{MANUSCRIPT_NAME}.pdf")
    for path in figures:
        if path.relative_to(stage).as_posix() not in required:
            raise RuntimeError(f"unexpected figure output {path}")
    provenance = {
        "schema_version": "publication_config_v1",
        "stage": args.stage_name,
        "manuscript_source": SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "manuscript_source_sha256": sha256_file(PROJECT_ROOT, SOURCE.relative_to(PROJECT_ROOT).as_posix()),
        "baseline_manifest_sha256": sha256_file(layout.baseline_dir, "baseline_manifest.json"),
        "advanced_manifest_sha256": sha256_file(layout.advanced_dir, "advanced_manifest.json"),
        "ledger_claims": len(rows),
        "ledger_fields": list(LEDGER_FIELDS),
        "pdf_rendered": not args.skip_pdf,
        "rendered_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    final = workspace.complete(required_files=required, provenance=provenance)
    print(final)
    return final


if __name__ == "__main__":
    main()
