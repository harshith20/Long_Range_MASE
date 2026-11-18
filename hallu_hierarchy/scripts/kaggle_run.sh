#!/usr/bin/env bash
# Guarded Kaggle CLI runner (CLAUDE.md §12). One run per call; never retries, never cancels,
# never deletes. Claude Code may only call this after the user said yes in the conversation.
#
# Usage:
#   hallu_hierarchy/scripts/kaggle_run.sh --stages correctness --gpu off
#   hallu_hierarchy/scripts/kaggle_run.sh --stages subject_states --gpu on [--config configs/x.yaml]
# Options:
#   --stages LIST          required; comma-separated (e.g. data,generation,correctness)
#   --gpu on|off           required; must match the stages (configs/kaggle_run.json, rule 3)
#   --config PATH          run config, relative to hallu_hierarchy/ (default in configs/kaggle_run.json)
#   --allow-generation     required in addition when the stages include `generation`
#   --create-kernel        required for the very first push, when the kernel does not exist yet
#   --max-wait-min N       stop waiting after N minutes (the Kaggle run keeps going)
#   --run-timeout-sec N    hard time limit for the Kaggle session itself (kaggle kernels push -t)
#
# Exit codes: 0 report downloaded and commit verified | 2 refused by a guard |
#             3 stopped waiting (run still going) | 4 Kaggle run failed | 5 report problem

set -euo pipefail

HH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # hallu_hierarchy/
REPO="$(git -C "$HH" rev-parse --show-toplevel)"
KAGGLE="$HH/.venv/bin/kaggle"
PY="$HH/.venv/bin/python"
CONF="$HH/configs/kaggle_run.json"

die()  { echo "kaggle_run: REFUSED: $*" >&2; exit 2; }
info() { echo "kaggle_run: $*"; }
usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

[[ -x "$KAGGLE" ]] || die "Kaggle CLI not found at $KAGGLE (create the venv: see CLAUDE.md §12)"
[[ -f "$CONF" ]] || die "missing $CONF"

# ------------------------------------------------------------------ arguments
STAGES="" GPU="" CONFIG="" ALLOW_GEN=0 CREATE=0 MAX_WAIT="" RUN_TIMEOUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)           STAGES="${2:-}"; shift 2 ;;
    --gpu)              GPU="${2:-}"; shift 2 ;;
    --config)           CONFIG="${2:-}"; shift 2 ;;
    --allow-generation) ALLOW_GEN=1; shift ;;
    --create-kernel)    CREATE=1; shift ;;
    --max-wait-min)     MAX_WAIT="${2:-}"; shift 2 ;;
    --run-timeout-sec)  RUN_TIMEOUT="${2:-}"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "unknown argument: $1 (see --help)" ;;
  esac
done
[[ -n "$STAGES" ]] || die "--stages is required (no default)"
[[ "$GPU" == "on" || "$GPU" == "off" ]] || die "--gpu on|off is required (no default)"
[[ -z "$MAX_WAIT" || "$MAX_WAIT" =~ ^[0-9]+$ ]] || die "--max-wait-min must be a whole number"
[[ -z "$RUN_TIMEOUT" || "$RUN_TIMEOUT" =~ ^[0-9]+$ ]] || die "--run-timeout-sec must be a whole number"

# ------------------------------------------------------------------ stage / GPU guards (one config file)
# Prints: normalized stages, accelerator, default config, kernel-metadata path, notebook path,
#         poll seconds, max wait minutes — or an error message and exit 1.
GUARD_OUT="$("$PY" - "$CONF" "$STAGES" "$GPU" "$ALLOW_GEN" 2>&1 <<'PYEOF'
import json, sys
conf_path, stages_arg, gpu, allow_gen = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
conf = json.load(open(conf_path))
alias = {"generate": "generation"}
stages = [alias.get(s.strip(), s.strip()) for s in stages_arg.replace(" ", ",").split(",") if s.strip()]
gpu_st, cpu_st = set(conf["gpu_stages"]), set(conf["cpu_stages"])
unknown = [s for s in stages if s not in gpu_st | cpu_st]
if not stages:
    sys.exit("no stages given")
if unknown:
    sys.exit(f"unknown stage(s) {unknown}; known: {sorted(gpu_st | cpu_st)}")
needs_gpu = [s for s in stages if s in gpu_st]
if needs_gpu and gpu != "on":
    sys.exit(f"stages {needs_gpu} need the GPU: use --gpu on")
if not needs_gpu and gpu != "off":
    sys.exit(f"stages {stages} are CPU-only (rule 3): use --gpu off")
if "generation" in stages and not allow_gen:
    sys.exit("`generation` also needs --allow-generation (rule 2: only when the user asked for it by name)")
wait = conf["max_wait_minutes"]["gpu" if needs_gpu else "cpu"]
print(",".join(stages), conf["accelerator"], conf["default_config"], conf["kernel_metadata"],
      conf["notebook"], conf["poll_seconds"], wait)
PYEOF
)" || die "$GUARD_OUT"
read -r STAGES ACCEL DEFAULT_CONFIG META_REL NB_REL POLL CONF_WAIT <<<"$GUARD_OUT"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
MAX_WAIT="${MAX_WAIT:-$CONF_WAIT}"
[[ -f "$HH/$CONFIG" ]] || die "config not found: hallu_hierarchy/$CONFIG"
RUN_NAME="$(sed -n 's/^run_name:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$HH/$CONFIG" | head -n1)"
[[ -n "$RUN_NAME" ]] || die "no run_name in $CONFIG"

# ------------------------------------------------------------------ kernel metadata
META="$HH/$META_REL"
KERNEL="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META")"
if grep -q "REPLACE_ME" "$META"; then
  die "kaggle/kernel-metadata.json still has REPLACE_ME placeholders (kernel slug / outputs dataset)"
fi

# ------------------------------------------------------------------ git guards
DIRTY="$(git -C "$REPO" status --porcelain -- hallu_hierarchy .claude)"
[[ -z "$DIRTY" ]] || die "uncommitted changes (Kaggle clones from GitHub and would run old code):
$DIRTY"
git -C "$REPO" fetch -q origin || die "git fetch failed; cannot check that the commit is pushed"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
SHORT="${COMMIT:0:7}"
[[ -n "$(git -C "$REPO" branch -r --contains "$COMMIT")" ]] \
  || die "commit $SHORT is not on GitHub yet — git push first"

# ------------------------------------------------------------------ one run at a time
STATUS_OUT="$("$KAGGLE" kernels status "$KERNEL" 2>&1)" && KERNEL_EXISTS=1 || KERNEL_EXISTS=0
if [[ $KERNEL_EXISTS == 1 ]]; then
  if grep -qiE 'queued|running|cancel_?requested' <<<"$STATUS_OUT"; then
    die "kernel $KERNEL is still busy ($STATUS_OUT). One run at a time."
  fi
elif grep -qiE 'auth|credential|401|403' <<<"$STATUS_OUT"; then
  die "Kaggle CLI is not logged in (see CLAUDE.md §12)"
elif [[ $CREATE == 0 ]]; then
  die "kernel $KERNEL not found. If this is the first run, pass --create-kernel (creates a new private kernel)."
fi

# ------------------------------------------------------------------ build the push folder
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUILD="$HH/kaggle/.build/$STAMP"
mkdir -p "$BUILD"
"$PY" - "$META" "$HH/$NB_REL" "$BUILD" "$GPU" "$COMMIT" "$CONFIG" "$STAGES" "$STAMP" <<'PYEOF'
import json, sys, pathlib
meta_p, nb_p, build, gpu, commit, config, stages, stamp = sys.argv[1:]
meta = json.load(open(meta_p))
meta["enable_gpu"] = "true" if gpu == "on" else "false"
meta["code_file"] = "run.ipynb"
json.dump(meta, open(pathlib.Path(build, "kernel-metadata.json"), "w"), indent=2)
request = {"commit": commit, "config": config, "stages": stages, "gpu": gpu == "on", "requested_utc": stamp}
nb = json.load(open(nb_p))
marker = "RUN_REQUEST = None"
hits = 0
for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "code" and src.startswith(marker):
        cell["source"] = "RUN_REQUEST = " + json.dumps(request) + src[src.index("\n"):]
        hits += 1
if hits != 1:
    sys.exit(f"expected exactly one '{marker}' line in the notebook, found {hits}")
json.dump(nb, open(pathlib.Path(build, "run.ipynb"), "w"), indent=1)
json.dump(request, open(pathlib.Path(build, "run_request.json"), "w"), indent=2)
PYEOF

# ------------------------------------------------------------------ summary, then push
ACC_ARGS=()
[[ "$GPU" == "on" ]] && ACC_ARGS=(--accelerator "$ACCEL")
TIMEOUT_ARGS=()
[[ -n "$RUN_TIMEOUT" ]] && TIMEOUT_ARGS=(-t "$RUN_TIMEOUT")
cat <<EOF
================ kaggle_run ================
kernel     : $KERNEL$( [[ $KERNEL_EXISTS == 0 ]] && echo "  (NEW — will be created)" )
stages     : $STAGES
config     : $CONFIG  (run_name $RUN_NAME)
GPU        : $GPU$( [[ "$GPU" == "on" ]] && echo "  ($ACCEL)" )
commit     : $SHORT  (pushed)
max wait   : ${MAX_WAIT} min (polling every ${POLL}s)
============================================
EOF
PUSH_OUT="$("$KAGGLE" kernels push -p "$BUILD" "${ACC_ARGS[@]}" "${TIMEOUT_ARGS[@]}" 2>&1)" || {
  echo "$PUSH_OUT"; echo "kaggle_run: push failed (not retried)" >&2; exit 4; }
echo "$PUSH_OUT"
grep -q "Kernel push error" <<<"$PUSH_OUT" && { echo "kaggle_run: push failed (not retried)" >&2; exit 4; }
grep -q "not valid dataset sources\|not valid kernel sources" <<<"$PUSH_OUT" \
  && { echo "kaggle_run: inputs could not be attached; the run is likely to fail. Stopping." >&2; exit 4; }
VERSION="$(sed -n 's/.*Kernel version \([0-9][0-9]*\) successfully pushed.*/\1/p' <<<"$PUSH_OUT" | head -n1)"
TARGET="$KERNEL${VERSION:+/$VERSION}"
info "pushed $TARGET"

# ------------------------------------------------------------------ wait (never cancels)
START=$(date +%s)
STATE="unknown"
FAILS=0
while :; do
  sleep "$POLL"
  S="$("$KAGGLE" kernels status "$TARGET" 2>&1 || true)"
  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  # Only a 'has status "..."' line describes the run; anything else is a failed status call.
  STATUS_WORD="$(sed -n 's/.*has status "\([^"]*\)".*/\1/p' <<<"$S" | head -n1 | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$STATUS_WORD" ]]; then
    FAILS=$((FAILS + 1))
    info "${ELAPSED} min: status call failed (${FAILS}/5): $(head -n1 <<<"$S")"
    if (( FAILS >= 5 )); then
      info "5 status calls in a row failed; stopped waiting. The Kaggle run is not cancelled."
      exit 3
    fi
    continue
  fi
  FAILS=0
  case "$STATUS_WORD" in
    *cancel*)   STATE="cancelled"; break ;;
    *error*)    STATE="error"; break ;;
    *complete*) STATE="complete"; break ;;
    *)          info "${ELAPSED} min: $STATUS_WORD" ;;
  esac
  if (( ELAPSED >= MAX_WAIT )); then
    info "stopped waiting after ${MAX_WAIT} min; the Kaggle run continues. Check later with:"
    info "  $KAGGLE kernels status $TARGET"
    exit 3
  fi
done
info "finished with status: $STATE ($S)"

# ------------------------------------------------------------------ download REPORT.md + DASHBOARD.png only
DL="$HH/reports/_downloads/$STAMP"
mkdir -p "$DL"
"$KAGGLE" kernels output "$TARGET" -p "$DL" --file-pattern '^(REPORT\.md|DASHBOARD\.png)$' -o -q \
  || info "download of REPORT.md failed"
DATE="$(date +%Y-%m-%d)"
NAME="${DATE}_${RUN_NAME}_${STAGES//,/+}"
if [[ -f "$DL/REPORT.md" ]]; then
  cp "$DL/REPORT.md" "$HH/reports/$NAME.md"
  [[ -f "$DL/DASHBOARD.png" ]] && cp "$DL/DASHBOARD.png" "$HH/reports/$NAME.png"
  info "report -> hallu_hierarchy/reports/$NAME.md"
  sed -n 's/^STATUS: //p' "$HH/reports/$NAME.md" | head -n1 | sed 's/^/kaggle_run: STATUS /'
  if ! grep -q "commit $SHORT" "$HH/reports/$NAME.md"; then
    echo "kaggle_run: WARNING: REPORT.md does not mention commit $SHORT — it may come from other code" >&2
    exit 5
  fi
else
  echo "kaggle_run: no REPORT.md in the run output (status $STATE)" >&2
  [[ "$STATE" == "complete" ]] && exit 5
fi
[[ "$STATE" == "complete" ]] || exit 4
exit 0
