#!/bin/bash
# kel-manifest.sh — emit every suite kernel .cu with the CURRENT
# bin/tcacomp and print a sha256 manifest in the uai baseline's format
# ("<sha256>  <name>"). quick-261101-kel §9 step c.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=$(mktemp -d /tmp/kel-manifest-XXXXXX)
trap 'rm -rf "$OUT"' EXIT
for k in testing/kernels/*/*.tca; do
    name=$(basename "$k" .tca)
    mkdir -p "$OUT/$name"
    bin/tcacomp compile -o "$OUT/$name" "$k" >/dev/null
    cu=$(ls "$OUT/$name"/*.cu | head -1)
    sha256sum "$cu" | awk -v n="$name" '{print $1 "  " n}'
done
