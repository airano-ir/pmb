#!/usr/bin/env bash
# Run CodeQL's `security-extended` suite (the exact one .github/workflows/codeql.yml
# uses) LOCALLY, so you can iterate on security findings without pushing to CI.
#
#   bash scripts/codeql_local.sh
#
# First run auto-downloads the CodeQL bundle (~1 GB) into ~/codeql-tools. Later
# runs just rebuild the DB and re-analyze (~1-2 min). Override the CLI path with
# CODEQL=/path/to/codeql, the DB dir with CODEQL_DB=/path, or the bundle version
# with CODEQL_BUNDLE=codeql-bundle-vX.Y.Z.
set -euo pipefail

BUNDLE_VER="${CODEQL_BUNDLE:-codeql-bundle-v2.25.6}"
TOOLS="${CODEQL_TOOLS:-$HOME/codeql-tools}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${CODEQL_DB:-$TOOLS/pmb-db}"
SARIF="$REPO/codeql-results.sarif"

# ── locate (or install) the CodeQL CLI ───────────────────────────────────────
CODEQL="${CODEQL:-}"
if [ -z "$CODEQL" ]; then
  for c in "$TOOLS/codeql/codeql.exe" "$TOOLS/codeql/codeql" "$(command -v codeql || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && CODEQL="$c" && break
  done
fi
if [ -z "$CODEQL" ]; then
  case "$(uname -s)" in
    Linux*)  ASSET=codeql-bundle-linux64.tar.gz ;;
    Darwin*) ASSET=codeql-bundle-osx64.tar.gz ;;
    MINGW*|MSYS*|CYGWIN*) ASSET=codeql-bundle-win64.tar.gz ;;
    *) echo "Unknown OS; set CODEQL=/path/to/codeql" >&2; exit 2 ;;
  esac
  echo ">> CodeQL CLI not found - downloading $BUNDLE_VER/$ASSET into $TOOLS (one-time)"
  mkdir -p "$TOOLS"
  curl -sL -o "$TOOLS/$ASSET" \
    "https://github.com/github/codeql-action/releases/download/$BUNDLE_VER/$ASSET"
  tar -xzf "$TOOLS/$ASSET" -C "$TOOLS"
  CODEQL="$TOOLS/codeql/codeql.exe"; [ -x "$CODEQL" ] || CODEQL="$TOOLS/codeql/codeql"
fi
echo ">> CodeQL: $("$CODEQL" --version | head -1)"

# ── build database + analyze (Python is a no-build language) ──────────────────
echo ">> building database -> $DB"
"$CODEQL" database create "$DB" --language=python --source-root="$REPO" \
  --overwrite --threads=0 >/dev/null

echo ">> analyzing (security-extended, same queries as CI)"
"$CODEQL" database analyze "$DB" \
  codeql/python-queries:codeql-suites/python-security-extended.qls \
  --format=sarif-latest --output="$SARIF" --threads=0 >/dev/null

# ── summarize ─────────────────────────────────────────────────────────────────
python - "$SARIF" <<'PY'
import json, sys
results = json.load(open(sys.argv[1], encoding="utf-8"))["runs"][0].get("results", [])
if not results:
    print("\n  CLEAN - 0 security-extended findings"); raise SystemExit(0)
print(f"\n  {len(results)} finding(s):")
for x in results:
    loc = x["locations"][0]["physicalLocation"]
    where = f"{loc['artifactLocation']['uri']}:{loc['region']['startLine']}"
    print(f"   [{x['ruleId']}] {where}\n     {x['message']['text'][:140]}")
PY
echo ">> SARIF written to $SARIF"
