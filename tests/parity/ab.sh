#!/bin/sh
# A/B of two builds of the Rust bench example on one runner, as a markdown table per thread
# setting: tests/parity/ab.sh <bench A> <bench B> [N]; run from the repo root. The runner type
# moves the Rust / Go ratio between bench runs (docs/benchmarks.md), so a change is measured
# against its base in the same job. A and B alternate, six runs each, each first in half of the pairs, and the best of the six
# counts per file. Not gated. ab.yml builds A from a base ref and B from the head.
set -eu
export LC_ALL=C
a=$1 b=$2 n=${3:-5}
echo "Runner: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1), $(nproc) CPUs," \
  "$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|bmi2|avx512(f|bw|vl))$' | tr '\n' ' ')"
d=$(mktemp -d) o=$(mktemp -d)
trap 'rm -rf "$d" "$o"' EXIT
cp -R tests/data/. "$d" && cp tests/parity/bigtiles/* "$d"
best() {
  awk -F'\t' '!($1 in m) || $3 < m[$1] { m[$1] = $3; b[$1] = $2 } END { for (k in m) printf "%s\t%s\t%s\n", k, b[k], m[k] }' "$@" | sort
}
for t in 1 0; do
  # A then B, then B then A: the second of a pair ran 1 to 3% faster on unchanged code.
  for i in 1 2 3 4 5 6; do
    if [ $((i % 2)) = 1 ]; then
      "$a" "$d" "$n" $t >"$o/a$i.tsv" 2>/dev/null
      "$b" "$d" "$n" $t >"$o/b$i.tsv" 2>/dev/null
    else
      "$b" "$d" "$n" $t >"$o/b$i.tsv" 2>/dev/null
      "$a" "$d" "$n" $t >"$o/a$i.tsv" 2>/dev/null
    fi
  done
  best "$o"/a?.tsv >"$o/a.tsv"
  best "$o"/b?.tsv >"$o/b.tsv"
  echo
  if [ $t = 1 ]; then echo "### One thread (max_threads=1), ms per megapixel"; else echo "### Default threads, reused ThreadPool, ms per megapixel"; fi
  echo
  # name:op, pixels, A ns, pixels, B ns. The classes are bench.sh's.
  join -t "$(printf '\t')" "$o/a.tsv" "$o/b.tsv" | awk -F'\t' '
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
function add(k) { p[k] += $2; x[k] += $3; y[k] += $5 }
{ op = ($1 ~ /:c0$/) ? "c0" : "dec"; add(class($1) " " op); add("all " op) }
END {
  print "| input | op | A ms/Mpixel | B ms/Mpixel | B / A |"
  print "|---|---|---|---|---|"
  m = split("bigtiles fl vsi512 svs256 conformance style plt region small 5/3 small 9/7 other all", ks, " ")
  for (j = 1; j <= 2; j++) {
    op = (j == 1) ? "dec" : "c0"
    for (i = 1; i <= m; i++) {
      k = ks[i]
      if (k == "small") { k = ks[i] " " ks[i + 1]; i++ }
      kk = k " " op
      if (p[kk]) printf "| %s | %s | %.3f | %.3f | %.3f |\n", k, op, x[kk] / p[kk], y[kk] / p[kk], y[kk] / x[kk]
    }
  }
}'
done
