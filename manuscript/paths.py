"""Manuscript locations shared by the builder, renderer and tests (no docx import)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_NAME = "CKD_Leakage_Safe_Reproduction_Publication_Style"
SOURCE = ROOT / "manuscript" / f"{MANUSCRIPT_NAME}.md"
