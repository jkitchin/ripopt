#!/usr/bin/env bash
# Run a solver against every .nl file in nl/, capture stdout/stderr per
# problem, and write a JSON results file with timing and termination status.
#
# Args: $1 solver_name (ripopt|ipopt)
#       $2 solver_binary (path or name on PATH)
#       $3 time_limit_seconds
#       $4 version_tag (e.g. v0.8.0)

set -u

SOLVER="$1"
BIN="$2"
TIMELIMIT="$3"
VERSION="$4"

LOGDIR="logs/${SOLVER}"
RESULT="results/${SOLVER}_${VERSION}.json"
mkdir -p "$LOGDIR"

# Start JSON array
echo "[" > "$RESULT"
first=1
total=$(ls nl/*.nl 2>/dev/null | wc -l | tr -d ' ')
i=0

for nl in nl/*.nl; do
  problem=$(basename "$nl" .nl)
  i=$((i+1))
  printf "[%2d/%d] %-25s %-7s ... " "$i" "$total" "$problem" "$SOLVER"

  log="${LOGDIR}/${problem}.log"
  start=$(python3 -c 'import time; print(time.time())')
  if [ "$SOLVER" = "ipopt" ]; then
    # Tell AMPL solver protocol it's running through AMPL
    timeout "$TIMELIMIT" "$BIN" "$nl" -AMPL \
      max_cpu_time="$TIMELIMIT" \
      > "$log" 2>&1
    rc=$?
  else
    timeout "$TIMELIMIT" "$BIN" "$nl" -AMPL \
      > "$log" 2>&1
    rc=$?
  fi
  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")

  # Determine status
  if [ "$rc" = "124" ]; then
    status="TIMEOUT"
  elif [ "$rc" -ne 0 ]; then
    status="ERROR"
  else
    status="OK"
  fi

  # Try to extract objective and iterations from the log
  obj=$(grep -oE 'Objective[: ]+[-+0-9.eE]+' "$log" | tail -1 | sed -E 's/^Objective[: ]+//')
  iter=$(grep -oE 'Number of Iterations[. :]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
  obj=${obj:-null}
  iter=${iter:-null}

  if [ $first -eq 0 ]; then echo "," >> "$RESULT"; fi
  first=0
  cat >> "$RESULT" <<EOF
  {"problem": "$problem", "solver": "$SOLVER", "status": "$status", "rc": $rc, "elapsed": $elapsed, "objective": $obj, "iterations": $iter}
EOF

  printf "%s (%.2fs)\n" "$status" "$elapsed"
done

echo "]" >> "$RESULT"
echo "wrote $RESULT"
