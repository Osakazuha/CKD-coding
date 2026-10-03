# Git release instructions

The analysis is deposited as one clean tree with a tag, and the manuscript's code-availability
statement points at that tag.  Run the following from inside this directory.

## 1. Initialise the release repository (once)

```bash
cd <release-tree>            # the directory that holds the files listed below
git init -b main             # name the default branch "main" (git 2.28 or newer)
git config user.name  "Lening Yang"
git config user.email "<your e-mail, or your GitHub noreply address>"
```

## 2. Stage the whitelist only — never `git add .` from another directory

```bash
git add -- \
  README.md GIT_RELEASE_INSTRUCTIONS.md LICENSE \
  requirements.txt requirements.lock.txt pytest.ini conftest.py .gitignore \
  run_experiment.py run_advanced_experiment.py run_revision_experiment.py \
  'src/*.py' 'src/models/*.py' \
  'tests/*.py' \
  manuscript/README.md manuscript/__init__.py manuscript/paths.py manuscript/references.md \
  manuscript/CKD_Leakage_Safe_Reproduction_Publication_Style.md \
  manuscript/build_publication_style.py manuscript/evidence.py \
  manuscript/publication_figures.py manuscript/render_results.py \
  manuscript/figure6_predictor_removal.py \
  'manuscript/tests/*.py' \
  data/chronic_kidney_disease_full.xlsx \
  tools/run_main.sh tools/run_revision.sh \
  artifacts/run_2026-09-28_seed12
git status --short        # confirm: nothing from logs/, scratch artifacts/ roots or editor files
# `git add -A` run inside this directory gives the same tree as this whitelist (see .gitignore).
```

`manuscript/paths.py` is imported by `manuscript/render_results.py` and by the manuscript tests;
omitting it makes the released tree fail with `ModuleNotFoundError: No module named 'manuscript.paths'`.

Included on purpose: `artifacts/run_2026-09-28_seed12/` (135 files, about 21 MB), the released run whose
persisted registries, fold scores, selection manifest, holdout evaluation, evidence ledger and stage manifests
the manuscript's availability statement refers to; `.gitignore` admits only this run.  Excluded on purpose:
scratch `artifacts/` roots, `logs/`, and the superseded first rendering of the publication stage
(`publication/`, replaced by `publication-v2/` after text-only corrections to the report source).

## 3. Commit, verify in a fresh clone, tag

```bash
git commit -m "CKD leakage-controlled reproduction: release"
git log --oneline -1
```

Before tagging, verify the committed tree in a fresh clone (a clone needs at least one commit):

```bash
git clone . /tmp/ckd-release-check && cd /tmp/ckd-release-check
python -m pip install -r requirements.lock.txt      # Python 3.9-3.12
python -c "import manuscript.render_results, src.advanced_experiment"
python -m pytest -q
python -m pytest -q --artifact-root artifacts/run_2026-09-28_seed12 --publication-stage publication-v2 -m release_artifact
cd -
```

```bash
git tag -a v1.0 -m "Run run_2026-09-28_seed12: baseline, advanced, revision and publication manifests verified"
git tag --list
```

Record the tag name and the run id (`artifacts/run_2026-09-28_seed12`) in the manuscript's Code and Data
Availability statement before submission.  The signed manifests inside the run (`baseline_manifest.json`,
`advanced_manifest.json`, `revision_manifest.json`, `publication-v2/publication_manifest.json`) tie every result
file to the source hashes recorded in each stage's `config.json`.

## 4. Publish

```bash
git remote add origin <repository-url>
git push -u origin main --tags
```
