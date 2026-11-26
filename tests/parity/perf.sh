#!/bin/sh
# Where the time goes, Go against Rust: perf stat and the flat perf profile (self time per symbol)
# of the bigtiles fixture at one thread, for each op in $OPS (default "c0 dec"), as markdown:
# tests/parity/perf.sh <dir>, with `oracle` (Go) and `bench-dbg` (Rust, line tables) in <dir>.
# Run on the runner itself, not in a container (profile.yml, job perf).
set -eu
export LC_ALL=C
b=$1
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp tests/parity/bigtiles/*.j2k "$d"
echo "Runner: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1), $(nproc) CPUs"
for op in ${OPS:-c0 dec}; do
  for side in go rust; do
    if [ $side = go ]; then set -- "$b/oracle" -threads 1 -bench 10 "$d"; else set -- "$b/bench-dbg" "$d" 10 1; fi
    BENCH_OP=$op perf stat -e task-clock,cycles,instructions,branches,branch-misses,cache-misses -o "$d/stat" "$@" >"$d/out" || true
    BENCH_OP=$op perf record -q -F 1999 -o "$d/perf.data" "$@" >/dev/null 2>&1 || true
    echo
    echo "#### $op, $side: $(cut -f1,3 "$d/out" | tr '\t' ' ') ns"
    echo
    echo '```'
    grep -vE '^#|^$' "$d/stat" || true
    echo
    perf report -i "$d/perf.data" --no-children --sort symbol --stdio 2>/dev/null | grep -vE '^#|^$' | head -n 40 | cut -c1-160 || true
    echo '```'
  done
done
