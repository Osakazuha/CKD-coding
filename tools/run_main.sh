#!/bin/bash
# One complete, auditable run of the README procedure (steps 1-5):
#   baseline -> advanced -> ledger check -> publication render (DOCX + PDF) -> unit tests -> release tests
# Usage: tools/run_main.sh [RUN_ID] [PUBLICATION_STAGE]
#   RUN_ID             defaults to run_<today>_seed12 (suffixed _runN if it already exists)
#   PUBLICATION_STAGE  defaults to "publication"; pass publication-vN to render a revision of an existing run
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${1:-run_$(date +%Y-%m-%d)_seed12}"
if [ -z "${1:-}" ]; then
  n=1; base="$RUN_ID"
  while [ -e "artifacts/${RUN_ID}" ]; do n=$((n+1)); RUN_ID="${base}_run${n}"; done
fi
PUB_STAGE="${2:-publication}"
RUN="artifacts/${RUN_ID}"
LOGDIR="logs/${RUN_ID}"
mkdir -p "$LOGDIR"
# LibreOffice (soffice) is required for the PDF of the publication stage (README step 3b).
export PATH="/Applications/LibreOffice.app/Contents/MacOS:$PATH"   # macOS LibreOffice; harmless elsewhere
KEEP_AWAKE=""; command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE="caffeinate -i"   # macOS only: keep the machine awake

PY=""
for candidate in python3 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 "$HOME/.pyenv/shims/python3" "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" .venv/bin/python; do
  if "$candidate" -c "import sklearn, xgboost, shap, pandas, numpy, scipy, matplotlib, seaborn, openpyxl, docx, pytest" >/dev/null 2>&1; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no Python interpreter with scikit-learn, xgboost, shap, python-docx and pytest was found." | tee "$LOGDIR/ERROR.txt"
  echo "Tried: python3 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 pyenv miniconda anaconda .venv" | tee -a "$LOGDIR/ERROR.txt"
  exit 1
fi
{
  echo "python=$PY"
  "$PY" -c "import sys; print(sys.version)"
  "$PY" -c "import numpy, pandas, scipy, sklearn, xgboost, shap, matplotlib, seaborn, openpyxl; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'shap', shap.__version__, 'matplotlib', matplotlib.__version__)"
  which soffice || echo "soffice: not on PATH (the publication PDF cannot be rendered; install LibreOffice or run manuscript/render_results.py --skip-pdf for a development build)"
} 2>&1 | tee "$LOGDIR/environment.txt"
"$PY" -m pip freeze > "$LOGDIR/pip_freeze.txt" 2>/dev/null || true

if [ -e "$RUN/baseline/baseline_manifest.json" ] && [ -e "$RUN/advanced/advanced_manifest.json" ]; then
  echo "== 1/5 + 2/5 skipped: $RUN already holds signed baseline and advanced stages" | tee -a "$LOGDIR/progress.txt"
else
  echo "== 1/5 baseline -> $RUN/baseline" | tee -a "$LOGDIR/progress.txt"
  $KEEP_AWAKE "$PY" run_experiment.py --artifact-root "$RUN" > "$LOGDIR/run_baseline.log" 2>&1
  echo "baseline exit=$?" | tee -a "$LOGDIR/progress.txt"

  echo "== 2/5 advanced -> $RUN/advanced" | tee -a "$LOGDIR/progress.txt"
  $KEEP_AWAKE "$PY" run_advanced_experiment.py --artifact-root "$RUN" > "$LOGDIR/run_advanced.log" 2>&1
  echo "advanced exit=$?" | tee -a "$LOGDIR/progress.txt"
fi

echo "== 3/5 ledger check + publication render -> $RUN/$PUB_STAGE" | tee -a "$LOGDIR/progress.txt"
"$PY" manuscript/render_results.py --artifact-root "$RUN" --check > "$LOGDIR/ledger_check.log" 2>&1
CHECK_EXIT=$?
echo "ledger check exit=$CHECK_EXIT" | tee -a "$LOGDIR/progress.txt"
if [ "$CHECK_EXIT" -eq 0 ]; then
  "$PY" manuscript/render_results.py --artifact-root "$RUN" --stage-name "$PUB_STAGE" > "$LOGDIR/render_${PUB_STAGE}.log" 2>&1
  echo "render exit=$?" | tee -a "$LOGDIR/progress.txt"
else
  echo "render skipped: the manuscript source does not match the evidence ledger (see $LOGDIR/ledger_check.log)" | tee -a "$LOGDIR/progress.txt"
fi

echo "== 4/5 unit tests" | tee -a "$LOGDIR/progress.txt"
"$PY" -m pytest -q -p no:cacheprovider > "$LOGDIR/pytest_unit.log" 2>&1
echo "unit exit=$?" | tee -a "$LOGDIR/progress.txt"

echo "== 5/5 release-artifact tests on $RUN (publication stage: $PUB_STAGE)" | tee -a "$LOGDIR/progress.txt"
REVISION_FILTER=()
if [ ! -e "$RUN/revision/revision_manifest.json" ]; then
  REVISION_FILTER=(-k "not test_revision_results")   # the revision stage is run separately (tools/run_revision.sh)
fi
"$PY" -m pytest -q -p no:cacheprovider --artifact-root "$RUN" --publication-stage "$PUB_STAGE" -m release_artifact ${REVISION_FILTER[@]+"${REVISION_FILTER[@]}"} > "$LOGDIR/pytest_release.log" 2>&1
echo "release exit=$?" | tee -a "$LOGDIR/progress.txt"
echo "ALL DONE" | tee -a "$LOGDIR/progress.txt"
