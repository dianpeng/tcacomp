#!/usr/bin/env bash
# build.sh — build the tcacomp compiler into bin/tcacomp.
#
# Pure Go, stdlib only: no dependencies to fetch, no cgo. The output lands in
# bin/ (gitignored) so it never pollutes the source tree.
set -euo pipefail

# Resolve the repo root from the script's own location, so this works from
# any cwd.
cd "$(dirname "$0")/.."

go build -o bin/tcacomp ./cmd/tcacomp

echo "tcacomp: built bin/tcacomp"
