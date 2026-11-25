#!/bin/sh
# Times the bench example built three ways against Go: tests/parity/pgo.sh [N]. Needs `oracle` (Go)
# and `bench` (the default build), `bench-pgo` (cargo-pgo, trained on tests/data) and, when BOLT was
# available, `bench-bolt` (PGO then BOLT) on PATH; run from the repo root (stage `pgo` of
# .github/ci/Dockerfile, workflow pgo.yml). Each build runs tests/parity/bench.sh (its Rust binary
# is $BENCH); the table gives the Rust ms/Mpixel of each input class and operation (dec: whole
# image, c0: component 0) in the three tables of bench.sh, and the change against the default build.
set -eu
export LC_ALL=C
n=${1:-5}
t=$(mktemp -d)
have=
labels=
for v in bench:default bench-pgo:PGO bench-bolt:"PGO + BOLT"; do
  b=${v%%:*}
  if ! command -v "$b" >/dev/null 2>&1; then
    echo >&2 "no $b: skipped"
    continue
  fi
  # bench.sh exits 1 when a class is above its 1.10 gate; the tables are still complete.
  BENCH=$b sh tests/parity/bench.sh "$n" >"$t/$b.md" || echo >&2 "$b: bench.sh exited $? (a class above the gate, or a failure above)"
  have="$have $t/$b.md"
  labels="$labels,${v#*:}"
done
[ -n "$have" ] || { echo >&2 "no bench binary on PATH"; exit 1; }
# A class row is `| input | op | files | Mpixel | Go | Rust | ratio |`, Rust ms/Mpixel in column 7;
# a "### " line starts a table.
# shellcheck disable=SC2086
awk -F'|' -v labels="${labels#,}" '
FNR == 1 { nf++; p = 0 }
/^### / { p++ }
/^[|] .* [|] (dec|c0) [|] [0-9]+ [|]/ {
  name = $2 " " $3; gsub(/^ +| +$/, "", name); gsub(/  +/, " ", name)
  v[nf, p, name] = $7 + 0
  if (!((p, name) in seen)) { seen[p, name] = 1; names[p, ++cnt[p]] = name }
  if (p > np) np = p
}
END {
  split(labels, lab, ",")
  split("default threads,single thread,no pool", pass, ",")
  for (p = 1; p <= np; p++) {
    print "#### Rust ms/Mpixel, " pass[p] " (lower is better; the change is against the default build)"
    print ""
    s = "| input op |"; r = "|---|"
    for (f = 1; f <= nf; f++) { s = s " " lab[f] " |"; r = r "---|"; if (f > 1) { s = s " change |"; r = r "---|" } }
    print s; print r
    for (i = 1; i <= cnt[p]; i++) {
      k = names[p, i]
      s = "| " k " |"
      for (f = 1; f <= nf; f++) {
        s = s sprintf(" %.3f |", v[f, p, k])
        if (f > 1 && v[1, p, k] > 0) s = s sprintf(" %+.1f%% |", (v[f, p, k] / v[1, p, k] - 1) * 100)
      }
      print s
    }
    print ""
  }
}' $have
for f in $have; do
  echo "<details><summary>$(basename "$f" .md): the full bench.sh tables</summary>"
  echo
  cat "$f"
  echo
  echo "</details>"
done
rm -rf "$t"
