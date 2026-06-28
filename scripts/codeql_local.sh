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

echo ">> SARIF written to $SARIF"

# ── summarize + gate ─────────────────────────────────────────────────────────
# Exits non-zero on any REAL finding (so this doubles as a pre-commit/pre-push
# gate). Known false positives - dismissed on GitHub - are listed but ignored.
python - "$SARIF" <<'PY'
import json
import sys

# Known false positives (also dismissed in GitHub code scanning):
#   (ruleId substring, path substring, why)
FALSE_POSITIVES = [
    ("clear-text-storage", "core/encryption.py",
     "Fernet ciphertext (AES-128-CBC+HMAC), not plaintext"),
]
results = json.load(open(sys.argv[1], encoding="utf-8"))["runs"][0].get("results", [])
real, ignored = [], []
for x in results:
    loc = x["locations"][0]["physicalLocation"]
    where = f"{loc['artifactLocation']['uri']}:{loc['region']['startLine']}"
    rid = x["ruleId"]
    if any(fr in rid and fp in where for fr, fp, _ in FALSE_POSITIVES):
        ignored.append((rid, where))
    else:
        real.append((rid, where, x["message"]["text"]))
if ignored:
    print(f"\n  {len(ignored)} known FP ignored:")
    for rid, where in ignored:
        print(f"   (fp) [{rid}] {where}")
if not real:
    print("\n  CLEAN - 0 real findings")
    sys.exit(0)
print(f"\n  {len(real)} REAL finding(s):")
for rid, where, msg in real:
    print(f"   [{rid}] {where}\n     {msg[:140]}")
sys.exit(1)
PY
