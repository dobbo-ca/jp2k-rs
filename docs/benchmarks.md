# Benchmark

`.github/workflows/bench.yml` (dispatch and push to `main`; not a required check) runs
jp2k-go's decoder and `examples/bench.rs` on the fixtures in one container and prints three tables
in the job summary (`tests/parity/bench.sh`), each with the ratio per input class:
1. Default threads, gated at 1.10: Go at its default `GOMAXPROCS` against Rust at its default
   `max_threads`, with a caller-held `ThreadPool` made once and reused across calls. That is the
   server setup, the counterpart of Go's warm goroutine runtime.
2. One thread, gated at 1.10: `GOMAXPROCS=1` against `max_threads: 1`.
3. No pool, gated at 1.10: Go at its default `GOMAXPROCS` against Rust with `Options::default()`
   (no caller pool, default `max_threads`: the shared pool). The first decode of each file is
   untimed, as for the other tables.

Two more tables follow, "v4 build, default threads" and "v4 build, single thread": the same
comparison with Go built with `GOAMD64=v4` (`oracle-v4`) and Rust built with
`RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls"` (`bench-v4`,
its own cargo cache mount). They are not gated: the 1.10 gate is for the default build. `ci` runs
`tests/rounding.rs`, `tests/conformance.rs` and `tests/decode.rs` under the same flags, so the v4
build is held to the same output bytes. The build profile and the flags:
[performance.md](performance.md).

`ab.yml` (dispatch only, not gated) builds the bench example twice, from a base ref (A, default
`main`) and from the dispatched ref (B), and times the two alternately on one runner
(`tests/parity/ab.sh`): ms per megapixel per class, at one thread and at default threads, and
B / A. A hosted runner's CPU model changes from run to run and moves the Rust / Go ratio with it,
so the before and after numbers of a change come from the same job.

## Hosted runner baseline

Bench run 45726753219 (2026-11-25) on a GitHub-hosted `ubuntu-24.04` runner, 2 CPUs, AVX-512
present. It is the baseline for later hosted runs; do not compare it with the Ryzen 7 9800X3D
numbers below as a regression. Rust / Go, ms per megapixel, 69 files:

| run | op | all | bigtiles | worst other class |
|---|---|---|---|---|
| default threads | dec | 0.77 | 0.69 | 1.05 (fl) |
| default threads | c0 | 1.02 | 1.09 | 0.97 (fl) |
| single thread | dec | 0.83 | 0.77 | 1.00 (fl) |
| single thread | c0 | 1.13 | 1.23 | 0.93 (fl) |
| v4, default threads (not gated) | c0 | 0.99 | 1.05 | |
| v4, single thread (not gated) | c0 | 1.14 | 1.23 | |

On this runner single-thread `c0` failed the 1.10 gate (bigtiles 1.23, all 1.13), so the bench job
was red; the next section has the runs after the fixes. The decode (`dec`) rows are well inside it. The AVX-512 tier is within
1% of the AVX2 tier here, and the Rust v4 build is about 4% faster on `c0` at default threads (8.39
against 8.72 ms/Mpixel) and 2% at a single thread. That is close to the run-to-run noise, so this
run alone is not a reason to recommend v4 flags.

## Hosted runners, before and after the speed fixes

Hosted `ubuntu-24.04` runners change CPU model from run to run, and the Rust / Go ratio moves with
it, so each run is listed with its CPU. Rust / Go per gated class; "1 thread" and "default" are the
single-thread and default-threads tables (the no-pool table, also gated, is not listed). The gate
is 1.10 on every class of the three gated tables.

Before, `main` at the first release (the bench example and oracle unchanged since):

| run | CPU | CPUs | 1 thread bigtiles `c0` | 1 thread all `c0` | default `fl` `dec` / `c0` | gate |
|---|---|---|---|---|---|---|
| 45726753219 | AMD EPYC 9V74 | 2 | 1.23 | 1.13 | 1.05 / 0.97 | red |
| 45730587064 | AMD EPYC 9V74 | 2 | 1.22 | 1.14 | 1.03 / 0.95 | red |
| 45761367793 | Intel Xeon 6975P-C | 4 | 1.18 | 1.11 | 1.17 / 1.17 | red |
| 45764444537 | Intel Xeon Platinum 8573C | 4 | 1.19 | 1.12 | 1.20 / 1.11 | red |
| 45767949597 | Intel Xeon 6975P-C | 4 | 1.16 | 1.09 | 1.16 / 1.18 | red |

The single-thread gap was on every CPU type, so it was the code, not the CPU: the inverse DWT
swept the zero-high levels of the big tiles in 64-column blocks (each a 4 KB plane row apart), and
tier-1 had about 17% more branch misses than Go's (`profile.yml`'s perf job, Intel, with
counters). The `fl` gap at default threads showed only on 4-CPU runners: a pooled call got one
thread fewer than the pool, 3 on 4 CPUs, where Go uses 4.

After (#2 #4 #6 #8 #9 #10; the before/after of each change, both builds on one runner, is in its
PR, from `ab.yml`):

| run | `main` | CPU | CPUs | 1 thread bigtiles `c0` | 1 thread all `c0` | default `fl` `dec` / `c0` | gate |
|---|---|---|---|---|---|---|---|
| 45781919270 | 3643d10 (#2 #4 #6) | AMD EPYC 9V74 | 4 | 1.15 | 1.08 | 0.99 / 0.92 | red |
| 45798634318 | efc3785 | AMD EPYC 7763 | 4 | 1.01 | 0.97 | 0.93 / 0.84 | green |
| 45802374209 | 1767e1b | AMD EPYC 9V74 | 4 | 1.07 | 1.00 | 0.94 / 0.89 | green |
| 45809001270 | 1767e1b | Intel Xeon 6975P-C | 4 | 0.98 | 0.94 | 0.95 / 0.94 | green |

In each green run the worst gated class is single-thread bigtiles `c0`; every other class is at
or below 1.00. Single-thread `dec` is 0.65 to 0.78 and default threads 0.62 to 0.95.

What is left on single-thread bigtiles `c0` (perf, Intel, `main` before #8 #9 #10): tier-1's
branch misses, which #10 cut, and the page faults of the 64 MB output plane, which a free
`decode_components` allocates for each call while Go's heap reuses the last call's memory (Rust
system time 0.26 s against Go's 0.19 s over the 12 calls of the profile). A run with glibc's
transparent huge pages for B (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, `ab.yml` run 45795226179,
not in the gate) changed one thread by 0.99 to 1.02 and default threads by 0.94 to 1.01, so the
faults cost little at one thread. A reused `Decoder` keeps its buffers between calls.

## Bench of record

The numbers below are from one `bench.yml` run on `main` with the final kernel set, which was not
rerun. They were measured on an AMD Ryzen 7 9800X3D (AVX-512); the
hosted-runner baseline (`ubuntu-24.04`, 4 vCPU, AVX-512 only if the host has it) is recorded by the
first `bench` run on that runner and is not comparable with these numbers. No class is above 1.10 in any gated
table. The worst gated class is single-thread bigtiles `c0` at 1.05 (Go 7.348, Rust 7.713). The
"all" rows, ms/Mpixel:

| table | op | Go | Rust | Rust / Go |
|---|---|---|---|---|
| default threads | dec | 3.076 | 2.292 | 0.75 |
| default threads | c0 | 2.432 | 1.552 | 0.64 |
| single thread | dec | 12.302 | 9.861 | 0.80 |
| single thread | c0 | 7.154 | 7.246 | 1.01 |
| no pool | dec | 2.689 | 2.147 | 0.80 |
| no pool | c0 | 2.196 | 1.433 | 0.65 |
| v4, default threads (not gated) | dec | 2.713 | 2.179 | 0.80 |
| v4, default threads (not gated) | c0 | 2.206 | 1.458 | 0.66 |
| v4, single thread (not gated) | dec | 11.991 | 9.684 | 0.81 |
| v4, single thread (not gated) | c0 | 7.049 | 7.067 | 1.00 |

The margin is thin on single-thread bigtiles `c0`: 1.05 here and 1.06 in an earlier branch run.
Earlier runs failed the gate on their first run and passed when the same commit ran
again (svs256 `c0` 1.12 at a single thread; fl 1.23 and 1.27 at default threads). So a single run's ratio near 1.10 is noise
plus margin, not a stable figure.

`pgo.yml` (dispatch only) times the bench example as a default, a PGO and a PGO plus BOLT build
(`tests/parity/pgo.sh`); its table is in [performance.md](performance.md).

## AVX-512 against AVX2

On a host with AVX-512, `bench.sh` ends with "AVX-512 against AVX2 (Rust only, not gated)": the
whole decode of every fixture with `JP2K_SIMD=avx2` against the default tier (AVX-512), best of
three runs, one thread and default threads, per class. Then the table of `examples/kernels.rs`:
each SIMD kernel through the hidden hooks `jp2k::fuzz::simd`, at the lengths a decode calls it
with, in ns per call on every tier, and AVX-512 over AVX2.

An earlier bench run (the AVX-512 tier with `dequant`, `pack_u16` and `pack_u8x4` of its own),
Rust ms/Mpixel on the "all" rows, against the last bench run on `main` before it:

| table | op | before | after |
|---|---|---|---|
| default threads | dec | 2.464 | 2.221 |
| default threads | c0 | 1.564 | 1.490 |
| single thread | dec | 11.544 | 10.160 |
| single thread | c0 | 7.391 | 7.513 |
| no pool | dec | 2.613 | 2.330 |
| no pool | c0 | 1.659 | 1.547 |

Most of the `dec` gain is `pack_u8x4`, which writes the 8-bit RGBA output straight from the float
samples on both x86_64 tiers (single thread, bigtiles `dec`: 10.69 to 8.73). AVX-512 against AVX2
in the same run is within 2% on every class (single thread "all": `dec` 0.998, `c0` 0.992; default
threads: 1.007 and 0.982): the kernels it wins (`dequant` 0.86 at 64, `pack_u16` 0.44 and
`pack_u8x4` 0.46 at 1024, in ns per call) are a small part of a decode, where tier 1 dominates.

The 5/3 path (a later bench run, against the one above): the rows and columns are
lifted on the split bands with the integer kernels instead of the interleaved line transform, and
the inverse RCT and the reversible dequantization are AVX2 (`dequant_rev` AVX-512) kernels. Rust
ms/Mpixel:

| class | op | single thread before | after | default threads before | after |
|---|---|---|---|---|---|
| small 5/3 | dec | 34.114 | 30.809 | 16.064 | 13.539 |
| conformance | dec | 102.037 | 93.570 | 24.286 | 19.967 |
| style | dec | 49.971 | 46.933 | 40.610 | 37.987 |

AVX-512 against AVX2 on those classes is within 1% (0.997 to 1.008).

On `main` with the final kernel set (the bench of record above: `dequant`, `pack_u16`, `pack_u8x4` and
`dequant_rev` on AVX-512, the others on AVX2), AVX-512 against AVX2 is 0.983 to 1.005 per class
at one thread ("all": `dec` 0.992, `c0` 0.998). At default threads it is 0.887 to 1.002 ("all":
0.949 and 0.971).
