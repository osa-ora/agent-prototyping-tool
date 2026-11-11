#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8750}"
export PORT
URL="http://127.0.0.1:${PORT}"
DATA_DIR="${AGENT_PRO_HOME:-$HOME/.agent-prototyping-app}"
mkdir -p "$DATA_DIR"
PID_FILE="$DATA_DIR/server.pid"
LOG_FILE="$DATA_DIR/server.log"
PYTHON="${PYTHON:-python3}"
health(){ curl -fsS "$URL/api/health" >/dev/null 2>&1; }
read_pid(){ [[ -f "$PID_FILE" ]] && cat "$PID_FILE" || true; }
case "${1:-start}" in
  status)
    pid="$(read_pid)"
    if health; then echo "Agent Prototyping App is running at $URL (PID: ${pid:-unknown})"; exit 0; fi
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "Process exists (PID: $pid), but the application is not responding on port $PORT"; exit 1; fi
    echo "Agent Prototyping App is stopped"; exit 1
    ;;
  stop)
    pid="$(read_pid)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; for _ in {1..30}; do kill -0 "$pid" 2>/dev/null || break; sleep .2; done; kill -9 "$pid" 2>/dev/null || true; fi
    rm -f "$PID_FILE"
    echo "Agent Prototyping App stopped"; exit 0
    ;;
  start|"") ;;
  *) echo "Usage: $0 [start|status|stop]"; exit 2 ;;
esac
if health; then echo "Agent Prototyping App is already running at $URL"; exit 0; fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Creating application virtual environment: $(pwd)/.venv"
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
nohup .venv/bin/python run.py >>"$LOG_FILE" 2>&1 </dev/null &
echo $! > "$PID_FILE"
for _ in {1..60}; do
  if health; then
    echo "Agent Prototyping App is ready at $URL"
    if command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
    fi
    exit 0
  fi
  sleep .5
done
echo "Application did not become ready. Check: $LOG_FILE" >&2
exit 1
