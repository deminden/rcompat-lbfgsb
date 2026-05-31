#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if find "$root" \
  -path "$root/target" -prune -o \
  -path "$root/external" -prune -o \
  -type f \( -iname '*.f' -o -iname '*.for' -o -iname '*.f90' -o -iname '*.c' -o -iname '*.h' \) \
  -print | grep -q .; then
  echo "Forbidden-looking native source files found outside ignored inspection/build areas." >&2
  exit 1
fi

if find "$root/external" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "external/ contains inspection-only material; keep it out of commits and packages." >&2
fi

echo "No forbidden source files found in tracked crate areas."
