#!/usr/bin/env bash
# format.sh — gofmt every Go source tree in the repo (cmd, internal, pkg).
#
# The repo root has no loose .go files, so the three trees are exhaustive.
# Prints how many files were rewritten, or "already formatted" when clean.
set -euo pipefail

# Resolve the repo root from the script's own location, so this works from
# any cwd.
cd "$(dirname "$0")/.."

unformatted=$(gofmt -l cmd internal pkg)
if [ -z "$unformatted" ]; then
  echo "tcacomp: already formatted (cmd internal pkg)"
  exit 0
fi

count=$(printf '%s\n' "$unformatted" | wc -l)
gofmt -l -w cmd internal pkg
echo "tcacomp: rewrote ${count} file(s)"
