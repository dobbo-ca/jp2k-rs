#!/bin/sh
# Instruction counts per function (callgrind) of the decode of a few fixtures, as the top of
# callgrind_annotate: tests/parity/profile.sh name...; run from the repo root with bench-dbg on PATH.
set -eu
for f in "$@"; do
  rm -rf /tmp/one && mkdir /tmp/one && cp "tests/data/$f" /tmp/one/
  valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out bench-dbg /tmp/one 1 >/dev/null 2>&1
  echo "#### $f"
  callgrind_annotate /tmp/cg.out 2>/dev/null | sed -n '1,45p' | cut -c1-200
done
