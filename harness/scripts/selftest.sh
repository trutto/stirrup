#!/usr/bin/env bash
# Stirrup self-test: API/model checks + (optional) browser UI checks.
# Usage: ./scripts/selftest.sh [--no-ui]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 1
IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
BASE="${STIRRUP_URL:-http://${IP:-127.0.0.1}:8090}"
echo "Stirrup self-test @ $BASE"
rc=0
python3 tests/harness_selftest.py "$BASE" || rc=1
if [[ "${1:-}" == "--no-ui" ]]; then
  echo "UI checks skipped (--no-ui)"
elif command -v node >/dev/null 2>&1; then
  node tests/harness_selftest_ui.mjs "$BASE" || rc=1
else
  echo "UI checks skipped (node not found)"
fi
if [[ $rc -eq 0 ]]; then echo; echo "All Stirrup self-tests passed."; else echo; echo "Some Stirrup self-tests FAILED (see above)."; fi
exit $rc
