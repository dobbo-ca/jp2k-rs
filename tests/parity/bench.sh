#!/bin/sh
# Decode time of the fixtures, Go and Rust on the same inputs, as markdown tables:
# tests/parity/bench.sh [N]. Needs `oracle` (Go) and `bench` (Rust) on PATH (or $ORACLE and $BENCH:
# other commands for the three default-build tables); run from the repo root.
#   default threads: Go at its default GOMAXPROCS, Rust at its default max_threads;
#   single thread:   Go with GOMAXPROCS=1 (`oracle -threads 1`), Rust with max_threads=1.
#   no pool:         Go at its default GOMAXPROCS, Rust with Options::default() (no caller pool: the
#                    crate's lazy shared pool).
# All three tables carry the 1.10 gate: the script fails if a class is above it.
# When `oracle-v4` (Go built with GOAMD64=v4) and `bench-v4` (Rust built with -C target-cpu=x86-64-v4
# and the false-deps features, docs/performance.md) are on PATH and $ORACLE and $BENCH are not set,
# two more tables follow, "v4 build, default threads" and "v4 build, single thread": the same
# comparison of the v4 builds. They are not gated (the 1.10 gate is for the default build).
# The default-threads gate is measured with a caller-held ThreadPool reused across calls (the
# server setup, like Go's warm goroutine runtime).
# On a CPU with AVX-512 (and $ORACLE and $BENCH not set), a last section times Rust alone with
# JP2K_SIMD=avx2 against its default (AVX-512) tier, per class, one thread and default threads, and
# prints the `kernels` example's table when it is on PATH. Not gated.
set -eu
export LC_ALL=C
n=${1:-5}
oracle=${ORACLE:-oracle} bench=${BENCH:-bench}
# The runner, in the job summary: the ratios move with the CPU model (docs/benchmarks.md).
echo "Runner: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1), $(nproc) CPUs," \
  "$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|bmi2|avx512(f|bw|vl))$' | tr '\n' ' ')"
echo
# The fixtures plus tests/parity/bigtiles: an 8192x4096 RGB 9/7 image in 32 tiles of 1024x1024 (made with
# `opj_compress -t 1024,1024 -I -n 6 -b 64,64 -r 40` from an ImageMagick plasma), where `tile_lanes`
# gives 2 lanes, well below the threads of a larger host: the few-lane case no other fixture is.
# Kept out of tests/data/ so the tests that loop over it do not decode it.
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp -R tests/data/. "$d" && cp tests/parity/bigtiles/* "$d"
# Best of three runs per file and side: the machine is shared, and one run can be slow.
best() {
  awk -F'\t' '!($1 in m) || $3 < m[$1] { m[$1] = $3; b[$1] = $2 } END { for (k in m) printf "%s\t%s\t%s\n", k, b[k], m[k] }' "$@" | sort
}
# The input class of a fixture name (awk).
classes='
function class(n) {
  if (match(n, /^bigtiles/)) return "bigtiles"
  if (match(n, /^fl_/)) return "fl"
  if (match(n, /^vsi512/)) return "vsi512"
  if (match(n, /^svs256/)) return "svs256"
  if (match(n, /^(g[1-4]_colr|c[12]_mono|p[01]_)/)) return "conformance"
  if (match(n, /^style_/)) return "style"
  if (match(n, /^plt_/)) return "plt"
  if (match(n, /^region_/)) return "region"
  if (match(n, /_53/)) return "small 5/3"
  if (match(n, /_97/)) return "small 9/7"
  return "other"
}
BEGIN { order = "bigtiles fl vsi512 svs256 conformance style plt region small 5/3 small 9/7 other all" }'
# tier <title> <oracle -threads> <bench threads> [gate 1|0]: both sides on every fixture, then the
# table. Returns 1 if a class is above the gate (gate 0: never).
tier() {
  title=$1 gt=$2 rt=$3 gate=${4:-1} bad=0
  for i in 1 2 3; do
    # `tier` runs in an AND-OR list, where `set -e` is off: a crashed side must fail it here.
    "$oracle" -threads "$gt" -bench "$n" $d >/tmp/go$i.tsv || { echo "::error::oracle -bench failed ($title)" >&2; return 1; }
    "$bench" $d "$n" "$rt" >/tmp/rs$i.tsv 2>/tmp/rs$i.err || { cat /tmp/rs$i.err >&2; echo "::error::bench failed ($title)" >&2; return 1; }
    cat /tmp/rs$i.err >&2
  done
  best /tmp/go?.tsv >/tmp/go.tsv
  best /tmp/rs?.tsv >/tmp/rs.tsv
  # Every side must have timed every file: a fixture that drops out of one side (a decode error
  # makes bench skip it) would otherwise vanish from the join and the gate would pass on the rest.
  ng=$(wc -l </tmp/go.tsv) nr=$(wc -l </tmp/rs.tsv) nj=$(join -t "$(printf '\t')" /tmp/go.tsv /tmp/rs.tsv | wc -l)
  nf=$(sed -n 's/^# files: //p' /tmp/rs3.err | tail -1)
  # bench prints `# files: N` (the fixtures it timed); tolerate its absence.
  if [ "$ng" != "$nr" ] || [ "$nj" != "$ng" ] || [ "$ng" -eq 0 ] ||
    { [ -n "$nf" ] && [ "$(cut -f1 /tmp/rs.tsv | sed 's/:.*//' | sort -u | wc -l)" != "$nf" ]; }; then
    echo "::error::$title: Go timed $ng rows, Rust $nr, joined $nj, bench files '$nf'" >&2
    return 1
  fi
  echo "### $title"
  echo
  # name:op, pixels, Go ns, Rust pixels, Rust ns. ns per pixel is ms per megapixel.
  join -t "$(printf '\t')" /tmp/go.tsv /tmp/rs.tsv | tee /tmp/joined.tsv | awk -F'\t' -v gate="$gate" "$classes"'
function add(k) { b[k] += $2; g[k] += $3; r[k] += $5; c[k]++ }
{
  k = class($1)
  op = ($1 ~ /:c0$/) ? "c0" : "dec"
  add(k " " op); add("all " op)
}
END {
  print "| input | op | files | Mpixel | Go ms/Mpixel | Rust ms/Mpixel | Rust / Go |"
  print "|---|---|---|---|---|---|---|"
  m = split(order, ks, " ")
  for (j = 1; j <= 2; j++) {
    op = (j == 1) ? "dec" : "c0"
    for (i = 1; i <= m; i++) {
      k = ks[i]
      if (k == "small") { k = ks[i] " " ks[i + 1]; i++ }
      kk = k " " op
      if (b[kk]) printf "| %s | %s | %d | %.2f | %.3f | %.3f | %.2f |\n", k, op, c[kk], b[kk] / 1e6, g[kk] / b[kk], r[kk] / b[kk], r[kk] / g[kk]
      # Rust at or below 1.10x Go on every input class.
      if (gate && b[kk] && r[kk] / g[kk] > 1.10) { printf "::warning::%s %s is %.2fx Go (limit 1.10)\n", k, op, r[kk] / g[kk] > "/dev/stderr"; bad = 1 }
    }
  }
  exit bad
}' || bad=1
  # Per file: the Go and Rust ns and the ratio, the worst first (stderr, for reading the log).
  echo "worst files: $title" >&2
  awk -F'\t' '$3 > 0 { printf "%s\t%d\t%d\t%.2f\n", $1, $3, $5, $5 / $3 }' /tmp/joined.tsv | sort -t"$(printf '\t')" -k4 -nr | head -20 >&2
  return $bad
}
rc=0
tier "Default threads: Go at its default GOMAXPROCS, Rust at its default max_threads on a reused ThreadPool (ms per megapixel, lower is better)" 0 0 || rc=1
echo
tier "Single thread: Go GOMAXPROCS=1, Rust max_threads=1 (ms per megapixel, lower is better)" 1 1 || rc=1
echo
tier "No pool, gated at 1.10: Go at its default GOMAXPROCS, Rust with Options::default() (no caller pool, default max_threads: the shared pool; ms per megapixel, lower is better)" 0 nopool || rc=1
if [ -z "${ORACLE:-}${BENCH:-}" ] && grep -qw avx512f /proc/cpuinfo 2>/dev/null && command -v oracle-v4 >/dev/null 2>&1 && command -v bench-v4 >/dev/null 2>&1; then
  oracle=oracle-v4 bench=bench-v4
  echo
  tier "v4 build, default threads, not gated: Go GOAMD64=v4 at its default GOMAXPROCS, Rust x86-64-v4 at its default max_threads on a reused ThreadPool (ms per megapixel, lower is better)" 0 0 0 || rc=1
  echo
  tier "v4 build, single thread, not gated: Go GOAMD64=v4 GOMAXPROCS=1, Rust x86-64-v4 max_threads=1 (ms per megapixel, lower is better)" 1 1 0 || rc=1
fi
# AVX-512 against AVX2: the default bench, best of three runs per file, with JP2K_SIMD=avx2 (the
# AVX2 tier) and without (AVX-512 on this CPU).
if [ -z "${ORACLE:-}${BENCH:-}" ] && grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
  echo
  echo "### AVX-512 against AVX2 (Rust only, not gated)"
  for t in 1 0; do
    for i in 1 2 3; do
      JP2K_SIMD=avx2 bench $d "$n" $t >/tmp/a2-$i.tsv 2>/dev/null || { echo "::error::bench failed (avx2, threads $t)" >&2; rc=1; }
      bench $d "$n" $t >/tmp/a5-$i.tsv 2>/dev/null || { echo "::error::bench failed (avx512, threads $t)" >&2; rc=1; }
    done
    best /tmp/a2-?.tsv >/tmp/a2.tsv
    best /tmp/a5-?.tsv >/tmp/a5.tsv
    echo
    if [ $t = 1 ]; then echo "Single thread (max_threads=1), ms per megapixel:"; else echo "Default threads, on a reused ThreadPool, ms per megapixel:"; fi
    echo
    # name:op, pixels, AVX2 ns, pixels, AVX-512 ns.
    join -t "$(printf '\t')" /tmp/a2.tsv /tmp/a5.tsv | awk -F'\t' "$classes"'
function add(k) { b[k] += $2; a[k] += $3; w[k] += $5; c[k]++ }
{ op = ($1 ~ /:c0$/) ? "c0" : "dec"; add(class($1) " " op); add("all " op) }
END {
  print "| input | op | files | avx2 ms/Mpixel | avx512 ms/Mpixel | avx512 / avx2 |"
  print "|---|---|---|---|---|---|"
  m = split(order, ks, " ")
  for (j = 1; j <= 2; j++) {
    op = (j == 1) ? "dec" : "c0"
    for (i = 1; i <= m; i++) {
      k = ks[i]
      if (k == "small") { k = ks[i] " " ks[i + 1]; i++ }
      kk = k " " op
      if (b[kk]) printf "| %s | %s | %d | %.3f | %.3f | %.3f |\n", k, op, c[kk], a[kk] / b[kk], w[kk] / b[kk], w[kk] / a[kk]
    }
  }
}'
  done
  if command -v kernels >/dev/null 2>&1; then
    echo
    echo "Kernels (\`examples/kernels.rs\`), ns per call, one thread:"
    echo
    kernels 2>&1 >/dev/null || { echo "::error::kernels failed" >&2; rc=1; }
  fi
fi
exit $rc
