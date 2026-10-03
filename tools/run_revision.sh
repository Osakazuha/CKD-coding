#!/bin/bash
# One complete run of the revision stage on an existing signed run (README, "Reproducing the analysis"):
#   revision -> unit tests -> release-artifact tests (baseline + advanced + revision; publication stage if present)
# Usage: tools/run_revision.sh [RUN_ID] [PUBLICATION_STAGE] [--smoke]
#   RUN_ID             defaults to run_2026-09-28_seed12, the released run (must hold signed baseline and
#                      advanced stages)
#   PUBLICATION_STAGE  defaults to "publication-v2", the publication stage of the released run; passed
#                      through to the release tests (use "publication" for a run rendered by run_main.sh)
#   --smoke            shortened, non-release run for an end-to-end check; use a scratch RUN_ID because the
#                      revision directory is immutable once written (--smoke may appear in any position)
set -uo pipefail
cd "$(dirname "$0")/.."
SMOKE=""
POSITIONAL=()
for arg in "$@"; do
  if [ "$arg" = "--smoke" ]; then SMOKE="--smoke"; else POSITIONAL+=("$arg"); fi
done
RUN_ID="${POSITIONAL[0]:-run_2026-09-28_seed12}"
PUB_STAGE="${POSITIONAL[1]:-publication-v2}"
RUN="artifacts/${RUN_ID}"
LOGDIR="logs/revision_${RUN_ID}"
KEEP_AWAKE=""; command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE="caffeinate -i"   # macOS only: keep the machine awake
mkdir -p "$LOGDIR"

PY=""
for candidate in python3 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 "$HOME/.pyenv/shims/python3" "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" .venv/bin/python; do
  if "$candidate" -c "import sklearn, xgboost, shap, pandas, numpy, scipy, matplotlib, seaborn, openpyxl, docx, pytest" >/dev/null 2>&1; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no Python interpreter with scikit-learn, xgboost, shap, python-docx and pytest was found." | tee "$LOGDIR/ERROR.txt"
  exit 1
fi
{
  echo "python=$PY"
  "$PY" -c "import sys; print(sys.version)"
  "$PY" -c "import numpy, pandas, scipy, sklearn, xgboost, shap, matplotlib; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'shap', shap.__version__, 'matplotlib', matplotlib.__version__)"
} 2>&1 | tee "$LOGDIR/environment.txt"

if [ ! -e "$RUN/baseline/baseline_manifest.json" ] || [ ! -e "$RUN/advanced/advanced_manifest.json" ]; then
  echo "ERROR: $RUN must hold signed baseline and advanced stages" | tee -a "$LOGDIR/ERROR.txt"
  exit 1
fi
if [ -e "$RUN/revision/revision_manifest.json" ]; then
  echo "== 1/3 skipped: $RUN already holds a signed revision stage" | tee -a "$LOGDIR/progress.txt"
else
  echo "== 1/3 revision -> $RUN/revision (this takes roughly 40 minutes on a laptop; do not sleep the machine)" | tee -a "$LOGDIR/progress.txt"
  $KEEP_AWAKE "$PY" run_revision_experiment.py --artifact-root "$RUN" $SMOKE > "$LOGDIR/run_revision.log" 2>&1
  echo "revision exit=$?" | tee -a "$LOGDIR/progress.txt"
fi

echo "== 2/3 unit tests" | tee -a "$LOGDIR/progress.txt"
"$PY" -m pytest -q -p no:cacheprovider > "$LOGDIR/pytest_unit.log" 2>&1
echo "unit exit=$?" | tee -a "$LOGDIR/progress.txt"

echo "== 3/3 release-artifact tests on $RUN (publication stage: $PUB_STAGE)" | tee -a "$LOGDIR/progress.txt"
"$PY" -m pytest -q -p no:cacheprovider --artifact-root "$RUN" --publication-stage "$PUB_STAGE" -m release_artifact > "$LOGDIR/pytest_release.log" 2>&1
echo "release exit=$?" | tee -a "$LOGDIR/progress.txt"
echo "ALL DONE" | tee -a "$LOGDIR/progress.txt"
