# Manuscript rendering

`CKD_Leakage_Safe_Reproduction_Publication_Style.md` is the source of the publication-style report of the
main analysis. Its prose is written by hand; every numeric value is a `{{claim_id}}` placeholder that
`render_results.py` fills from the evidence ledger (`evidence.py`), which is derived from the signed baseline
and advanced stages and the selection manifest.

```bash
python manuscript/render_results.py --artifact-root artifacts/<run-id> --check   # ledger vs. sections
python manuscript/render_results.py --artifact-root artifacts/<run-id>           # figures + ledger + DOCX + PDF
```

Outputs are written only to `artifacts/<run-id>/publication/` (or `publication-vN/` for a re-rendering) and
signed by `publication_manifest.json`. `publication_figures.py` draws every figure from persisted CSV/JSON
tables and fits no model. `figure6_predictor_removal.py` draws Figure 6 of the journal manuscript from the
revision stage's `predictor_removal_summary.csv`.

The journal manuscript itself was finalised in a word processor: it contains the values of this report plus
the post hoc analyses of the revision stage, which were transcribed from the persisted revision tables.
