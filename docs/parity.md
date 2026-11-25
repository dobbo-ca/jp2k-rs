# Parity with Go

## Overview

`tests/parity/go/upstream` is the vendored jp2k-go source and `tests/parity/go/oracle` a Go program that decodes
a directory of inputs through its public API; `examples/oracle.rs` does the same with this crate and
CI (`parity`, and `parity-arm64` on a native arm64 runner) requires the two output trees to be equal, with the
SIMD tiers and with `JP2K_SIMD=scalar`. The inputs are the fixtures, deterministic mutations of
them, JP2 files with variants of `colr`, `cdef` and box structure, jp2k-go's crafted security
inputs, and random regions. Every output is compared: pixel format, bounds and a hash of the
samples, or the error text. The inputs, the operations and the counts follow.

## Tests

`cargo test` runs the unit tests in the modules (marker parsing, MQ decoder, tag trees, the
coding passes, DWT, rounding) and the integration tests in `tests/`, which port Go's decoder,
jp2, security, region, components, conformance and rounding tests against the fixtures in
`tests/data/`. `tests/mutations.rs` is a quick deterministic stand-in for Go's `Fuzz*` tests (the real fuzz targets are in `tests/fuzz/`): mutated copies of every fixture must not panic any entry point. Go's tests that need `opj_decompress`, the openjpeg-data corpus, the process-wide
memory budget (tickets, GC, cgroups) or `go test -bench` have no counterpart; this page lists each
Go test of `budget_test.go`, `concurrency_test.go`, `parallel_test.go` and `nosimd_test.go` that does not apply, with the reason.
`tests/threads.rs` and `tests/heap.rs` (a counting allocator, so its own binary) check the thread and memory limits.

## Parity job

`tests/parity/go/upstream/` is the non-test source of dobbo-ca/jp2k-go: the root package and
`internal/` (with the amd64 and arm64 assembly), `go.mod` and `LICENSE` (jp2k-go has no NOTICE).
Left out: the tests, `testdata`, `cmd/` (the `jp2kinfo` and `tojson` tools), `docs/`, `bench/`.
`tests/parity/go/oracle/` is a Go program with its own `go.mod` that `replace`s the module with `../upstream`.
The CI job `parity` (`.github/ci/Dockerfile`, stages `go-oracle`, `go-run`, `parity`) runs:

1. `oracle -gen tests/data /inputs` writes the inputs.
2. `oracle /inputs /go-out` decodes them with jp2k-go (public API only; the oracle sets
   `GOMAXPROCS(1)`, so the error of a failing multi-tile image is the first failing tile's, as in
   Rust, and `SetMemoryLimit(2 GiB)` bounds its memory).
3. `cargo run --release --example oracle -- /inputs /rust-out` decodes them with this crate
   (`Limits::max_alloc` 2 GiB, `max_threads` the default: every core of the runner, on one shared
   `ThreadPool`), then again with `JP2K_SIMD=avx2` (on a host with AVX-512 the default run is
   the AVX-512 tier and this one keeps the AVX2 kernels compared with Go; the `ubuntu-24.04` runners
   may not have AVX-512, and then both runs use AVX2) and with
   `JP2K_SIMD=scalar`. Rust's output does not depend on
   the thread count, and neither do its errors (tile lanes report the lowest failing tile's error,
   as one thread does) except the tile named when a crafted multi-tile input exhausts the shared
   tier-1 work budget ([go-differences.md](go-differences.md)); Go
   keeps `GOMAXPROCS(1)` in this mode only because its own lanes report whichever failing tile
   finishes first. The env var `JP2K_ORACLE_THREADS` picks the configuration: `pool` (the default
   above), `nopool` (the default `max_threads`, no caller pool: the crate's shared pool, as
   `Options::default()` gets), `percall` (the same with `Options::per_call_pool`: one set of
   workers built per call) or `1` (`max_threads` 1, the serial path). The job runs `nopool` (the
   real default of `Options::default()`) and `1` on the full set too, and `percall` on the `/sub`
   subset (below).
4. `diff -r` of each Rust tree with Go's must be empty: `/go-out` for the default, AVX2, scalar, `nopool` and `1` runs,
   `/go-sub` for the `percall` run. A difference is a bug in the Rust port, never in the
   oracle.

The `/sub` subset is the same one `parity-arm64` uses. The job `parity-arm64` runs the oracle example natively on an arm64 runner (the NEON
kernels, then `JP2K_SIMD=scalar`) on a subset (every input without `~` in its name, and every 20th
mutation), and diffs it with Go's output for the same subset.

## Inputs

The `parity` job prints the totals into the CI log: 5795 inputs and 164750 outputs (from the
operations below). `oracle -gen` prints the number per category in the `go-run` stage: about 69
fixtures, 5,350 mutations, 175 JP2 files and 70 crafted codestreams, plus the `seed-*` fuzz seeds;
the per-category numbers are approximate, the totals are from the log of an earlier run.
`tests/parity/bigtiles/bigtiles_rgb97.j2k` (8192x4096, 32 tiles of 1024x1024) is copied in as it is, not mutated, and only its whole decode and component 0 are compared (both oracles skip the pixel cap and the other operations for names starting with `bigtiles`); it is not in the `/sub` subset. It covers the few-lane thread split.

| category | what |
|---|---|
| `fixture` | every `.j2k`, `.j2c` and `.jp2` of `tests/data/` as `fx-<name>` |
| `mutation` | per fixture `<name>~<tag>`: truncations (`t<cut>`: 0, 1, 2, 4, 8, 16, 41, 50, 64, and fractions of the length), 24 bit flips (`f<bit>`, half of them in the first 160 bytes), 8 byte sets in the first 200 bytes (`s<pos>_<value>`: 0x00, 0xff, random), for each marker segment of the main header (up to 12) its length field set to length -1, +1, 0xffff and 0 (`l<offset>_<n>`) and the segment deleted (`d<offset>`), for each multi-tile fixture under 100 KB the first tile-part's Psot set to 0, 12, 13, 14, 100, 0x7fffffff and 0xffffffff (`psot<value>`) and a lying TLM marker inserted ahead of the first SOT (`tlm`; `TestHandleHostile`), one byte and 16 random bytes appended (`a1`, `a16`), three 16-byte zeroed ranges (`z<pos>`). Fixtures over 100 KB get 3 truncations, 6 flips and one zeroed range. Positions come from a xorshift64 seeded with the name's FNV-1a hash. |
| `jp2` | JP2 files built around `rgb8_rct_m5.j2k` and `g64_53.j2k`: colour specifications (`colr` 0, 1, 12, 14, 16, 17, 18, 20, 24, ICC, short), `cdef` (all six R/G/B permutations, incomplete, duplicate Cn, Asoc 0 and 9, Typ 1, 2 and 65535, Cn 9, empty, 700 duplicate entries, trailing bytes), box structure (0, 1, 100 and 5000 `free` boxes, 300 KB `xml ` before and after `jp2h`, `jp2c` first, no `jp2h`/`ihdr`/`ftyp`/`jp2c`, two `ihdr`/`jp2c`, `pclr`, `cmap`, an unknown box, a nested `jp2h`), the `jp2c` box in each length form (cut short, LBox too long, LBox 0, XLBox whole, cut, 1<<62, 2^64-1 and 8, LBox 4, a 0xfffffff0 box before it), a bad signature, every truncation of the first 100 bytes of a file and every 97th byte after. |
| `hostile` | the crafted codestreams of jp2k-go's security tests (`HostileJ2K`, `HeaderFlood`, `multiTileBomb`, `tier2Bomb`, ported into `gen.go`) with their truncations; inputs named `hostile-*` may declare up to 1<<24 pixels |

Also, in the oracle itself and not in the input files: the random regions of `regionCases` and
`checkRegions` (below).

### Ported Go generators

| Go test | Go cases | Generated here |
|---|---|---|
| `TestLayerBombEveryOrder` (`decoder/security_test.go`) | 5 | `hostile-layerbomb-<order>`, plus `hostile-layers999-<order>` and `hostile-packets3-<order>` (the shape of `TestPacketBudgetEveryOrder`) |
| `TestPOCWindowIsCheap`, `TestPOCCapIsPerTile`, `TestSegmentLengthWrap`, `TestPPMHugeNppm`, `TestPrecinctGeometryBudget`, `TestCodeBlockBudget`, `TestHeaderExhaustionFails`, `TestUncoveredPrecinctsCount` | 8 | one each: `hostile-poc-window`, `-poc-cap`, `-seglen`, `-ppm-huge`, `-precinct-geom`, `-cb-budget`, `-header-exhaustion`, `-uncovered-precincts` |
| `TestPrecinctAtDeepLevels` | 30 (size 1, 16 x orders RPCL, PCRL, CPRL x 5 level counts) | `hostile-deep-<size>-<order>-<levels>`: 30 |
| `TestTier2WorkBudgetRejectsBomb`, `TestTier2DeadlineHonoredPromptly`, `TestHandleSharesTier2Budget` | 3 | `hostile-t2bomb-<layers>` (10, 100, 999), `hostile-t2bomb-exp-<layers>` (10, 50: the shortcut-defeating shape, with fewer layers because the oracle has no deadline), `hostile-multitile-bomb` |
| `TestPPMBudgetIsShared`, `TestTilePartCap`, `TestTier1Amplifier` | 3 | `hostile-ppm-shared`, `hostile-tileparts-255` and `-256`, `hostile-tier1-amplifier` |
| `TestHeaderFloodHeap`, `TestTilePartFloodHeap` | 5 + 1 | `hostile-flood-*` at 64 KB (and 1000 and 70000 bare tile-parts) |
| `TestDecodeCdef*`, `TestDecodeConfig*`, `TestDecodeManyEmptyBoxesBounded` (`decoder/jp2_test.go`), `TestJP2Codestream*` (`codestream/jp2_test.go`) | 14 + 13 | the `jp2-*` inputs above |
| `regionCases`, `checkRegions` (`handle_test.go`, `region_test.go`): random regions | 200 per fixture in `TestHandleMatchesDecodeRegion`, 12 per file in `TestDecodeRegion*`, 40 more in two others | per input the whole image at reduce 0 and 1 and four random regions (reduce 0 to 3), `r0`..`r5`; the random numbers are the same xorshift64 on both sides, seeded with the input name |
| `FuzzDecode`, `FuzzPipeline`, `FuzzTier2` seeds (`testdata/fuzz`) | 5 | `tests/fuzz/seeds` (decoded), which `oracle -gen tests/data /inputs tests/fuzz/seeds` also writes as `seed-<name>`, and so inputs of the fuzz runs and of the parity job |

Go counts: 360 `Test` functions, 8 `Fuzz`, 15 `Benchmark`. Rust counts: `grep -rh '#\[test\]' tests | wc -l` and the same over `src` (the numbers move with
every change, so they are not kept here).

### Go test files and where they are covered

| Go file (`Test`/`Fuzz`) | Rust |
|---|---|
| `codestream/markers_test.go` (36/1) | `src/codestream/markers.rs` unit tests, `tests/codestream.rs`; `tests/fuzz/fuzz_targets/markers.rs` |
| `codestream/jp2_test.go` (13), `decoder/jp2_test.go` (14) | `tests/jp2.rs`; the `jp2-*` oracle inputs |
| `codestream/security_test.go` (3), `codestream/tile_test.go` (2) | `tests/codestream.rs`, `tests/security.rs` |
| `decoder/decode_test.go` (24/1) | `tests/decode.rs`; every oracle input |
| `decoder/conformance_test.go` (15), `style_fixtures_test.go` (4), `internal/jp2k/*` (9) | `tests/conformance.rs`, `tests/rounding.rs`, `tests/examples.rs` |
| `decoder/components_test.go` (10) | `tests/components.rs`; oracle `c*` ops |
| `decoder/region_test.go` (13/1), `handle_test.go` (30/1) | `tests/region.rs` (`bad_tile_part` is `TestHandleBadTilePart`; `open_rejects_low_marker`, `bad_tile_part_header_marker`; the lying Psot and TLM of `TestHandleHostile` are the `~psot*` and `~tlm` oracle inputs), oracle `r*`, `rc*`, `open*` ops on every input, including mutations; `fuzz/region` |
| `decoder/security_test.go` (29), `tier2/security_test.go` (2) | `tests/security.rs`, `tests/tier2.rs`; the `hostile-*` oracle inputs |
| `decoder/pipeline_test.go` (14/1), `tier1_pipeline_test.go` (2/1) | `tests/decode.rs`, `tests/fuzz/fuzz_targets/decode.rs` |
| `decoder/budget_test.go` (34), `concurrency_test.go` (11), `nosimd_test.go` (1), `parallel/parallel_test.go` (1) | `tests/heap.rs`, `tests/threads.rs` and `src/parallel.rs` unit tests for those that apply; the rest, with the reason, in [Go budget, concurrency and parallel tests](#go-budget-concurrency-and-parallel-tests) |
| `dsp/*` (47: `dwt` 8, `jj2000` 7, `kernel` 14, `kernel_amd64` 1, `nofma` 1, `nosimd` 1, `quant` 8, `rounding` 2, `sep` 5) | `src/dsp/` unit tests and the cross-tier tests in `src/dsp/simd/mod.rs`; `tests/fuzz/fuzz_targets/simd_kernels.rs`. `dsp/jj2000_test.go` (7): `TestSynth97JJ`, `TestInverseDWT97JJ`, `TestInverseDWT97JJZeroHigh`, `TestSamplesJJ` and `TestInverseDWT97JJBlockMatchesRef` are in `src/dsp/jj2000.rs` and `src/dsp/mod.rs` (the Java-model vectors are copied bit for bit); `TestSynth97JJBlock` and `TestSynth97JJSplit` test Go's block and split kernels, which this crate does not have (one JJ2000 line kernel only), so `inverse_dwt_97_jj_matches_line_by_line` checks the property they guard. The DWT tests that ran `Fused` and `Separate` (`row_kernels_match_line_transform`, `zero_high_levels_match_full_lifting`) run `JJ2000` too |
| `tier1/*` (9/1: `mq_test.go` 6/1, `passes_test.go` 1, `tier1_test.go` 2) | `src/tier1/` unit tests; `tests/fuzz/fuzz_targets/tier1.rs` |
| `internal/fixture/flfixtures_test.go` (5) | N/A: the five tests regenerate `testdata/fl_*`, `vsi512_*`, `svs256_*` and the band-tile fixtures with `opj_compress`, and are skipped unless `JP2K_GENERATE_FL=1`; they test no decoder code, and the files they write are committed in `testdata/` and read by the ported decoder tests |
| `tier2/tagtree_test.go` (6), `tier2_test.go` (26/1) | `src/tier2/tagtree.rs`, `tests/tier2.rs`; `tests/fuzz/fuzz_targets/tier2.rs` |

`tests/errors.rs` (the error variants, below) and the rounding check of `decode_region` have no Go twin.
Not every `Test` function has a Rust twin: the table says which files do, and the oracle covers
the public API on every generated input.
Not generated: the tests of internal packages that the public API cannot reach (`dsp` kernels,
`tier1` and `tier2` internals, the memory budget, `parallel`, concurrency tests). Rust ports the
`dsp` kernel tests as cross-tier tests (`src/dsp/simd/mod.rs`) and fuzzes them differentially
(`tests/fuzz/`).

### Go budget, concurrency and parallel tests

Go's memory limit is process-wide (tickets, an admission queue, GC pacing, cgroup limits) and its
decodes share goroutines and slots. This crate has none of that: `Limits::max_alloc` is the peak
of one call's ledger, `Limits::max_threads` the threads of one call, there is no admission queue,
no global state, and nothing waits for another decode. Each Go test of the four files is listed.

Ported or adapted (47 Go tests in `budget_test.go` 34, `concurrency_test.go` 11, `nosimd_test.go` 1 and
`parallel/parallel_test.go` 1; 12 have a Rust test or a CI step):

| Go test | Rust |
|---|---|
| `TestEstimateCoversPeak` | `tests/heap.rs::the_ledger_covers_the_heap`: for every fixture (whole, region, component; 1 thread and the default) the real heap peak is within 1 MiB of what the ledger counted; plus `max_alloc_caps_the_peak`, which holds the heap and ledger peaks to the limit for several limits and 1, 2, 3 and the default threads |
| `TestEstimateCoversPackedHeaders` | `tests/heap.rs::packed_headers_are_charged` (the same 2 tiles of 128 PPT markers, whole and region) |
| `TestHugeCanvasFailsUnderLimit` | `tests/heap.rs::a_huge_canvas_fails_under_the_limit` (136 bytes declaring 6 GB: `LimitExceeded` at once with under 1 MiB of heap) |
| `TestBigBoxJP2Decodes` | `tests/heap.rs::a_big_box_jp2_decodes_within_a_tight_limit` (50 MiB `xml ` box at 58 and 60 MiB; from a slice within 4 MiB). Its `checkSized` part is not ported: a Rust reader has no size |
| `TestSizedReaderFailsFast` | `tests/heap.rs::a_reader_over_max_alloc_fails_fast`: a reader has no size, so the doomed read stops as soon as `max_alloc` is passed, having read about the limit |
| `TestCancelWhileDecoding` | `tests/threads.rs::cancel_from_another_thread_stops_the_decode` (a flag set from another thread mid-decode, at 1, 2, 3 and the default threads; Go cancels from a fault hook at each stage) and `cancel_set_before_the_call` |
| `TestDecodeConcurrent` | `tests/threads.rs::concurrent_decodes_agree` |
| `TestDecodeAllocBudget` | `tests/heap.rs::decode_heap_is_the_output_and_a_plane` (heap peak, not Go's bytes allocated) |
| `TestParallelTilesMatchSerial` | `tests/threads.rs::every_fixture_is_identical_for_every_thread_count` (every fixture, whole, region and component, at 1, 2, 3 and the default threads; includes Go's five files) and `opened_image_is_identical_for_every_thread_count` |
| `TestDecodePanicRecovered` | `src/parallel.rs::tests::panic_is_an_internal_error` (a panic on a worker or the caller is `Error::Internal`); Go injects a panic at six named sites through a global `FaultHook`, which a crate with no global state does not have |
| `TestParallelForErrorJoinsWorkers` | `src/parallel.rs::tests::error_stops_new_items_and_joins_threads` (and `scratch_is_per_thread`, `cancel_stops_every_thread`, `items_run_once_on_at_most_nw_threads`) |
| `TestNoSIMDEnv` | CI runs the whole suite and the parity oracle with `JP2K_SIMD=scalar` as well as with the SIMD tiers |

Not applicable (35), with the reason:

| Go tests | Why |
|---|---|
| `TestFloorDecidesPermanentError`, `TestSetMemoryLimitConcurrent`, `TestDefaultMemoryLimit`, `TestCgroupV2Path`, `TestCgroupMaxFallsBackWhenResolvedRootIsEmpty`, `TestDefaultLimitFollowsRuntime` | Go's process-wide limit, set with `SetMemoryLimit` or derived from `GOMEMLIMIT` and the cgroup; here the limit is a field of the call |
| `TestDecodeWaitsForBudget`, `TestCancelWhileWaitingForBudget`, `TestParkedDecodesGoAhead`, `TestParkedReadersFail`, `TestCancelWhileParked`, `TestParkedWakesTargeted`, `TestTicketQueueOrder`, `TestEntrantThatFitsWakesBehindQueuedOne`, `TestCapWakesOnRun`, `TestReadCap`, `TestSlowReaderDoesNotBlockOthers`, `TestUnknownSizeReadersDoNotDeadlock`, `TestSlowReadersDoNotBlockSmallDecodes`, `TestBigReaderNotStarved`, `TestConcurrentFloodsShareBudget` | tickets and the admission queue: a decode that does not fit waits for others. Here a call that needs more than its own `max_alloc` fails at once and calls do not share a ledger |
| `TestGOGCFactor`, `TestCollectGarbageCoalesces`, `TestCollectForCoalesces`, `TestReleasedBytesWaitForGC`, `TestCollectForDoneDecodes`, `TestGarbageHeldUntilGC` | Go counts garbage until a GC runs; freed memory is returned to the ledger on drop |
| `TestDecodeWaitsForSlot`, `TestSlotLimitFollowsGOMAXPROCS`, `TestDecodeContextCancelWhileWaiting`, `TestSmallDecodesTakeMoreSlots`, `TestWakeSlotWakesSmallPastBaseLimit`, `TestRunSmallCountsHeldBytes` | decode slots over `GOMAXPROCS`: a process-wide limit on concurrent decodes; here each call has its own `max_threads` |
| `TestDecodeGOMAXPROCSChanges` | `max_threads` is fixed for a call; there is no runtime parallelism to change |
| `TestSectionReaderPastEnd` | Go's `io.ReaderAt` and `SectionReader` input; Rust reads a slice or an `io::Read` |

## Go errors and Rust variants

The oracle compares the error text (collapsed to `limit` or `internal` where Go's and Rust's words
differ), not the variant, so the mapping below is pinned by `tests/errors.rs` (one input per row
marked there) and by the tests named in the last column. `Display` keeps Go's text; the variant
is the class.

| Go | Rust `Error` | Test |
|---|---|---|
| `fmt.Errorf` for malformed input: bad marker segment, box or packet header, truncated stream, `no components requested`, `box length shorter than its header`, `tier2.ErrWork` (packet-header inclusion work budget) | `Invalid` (the work budget is a function of the input's size alone, so a stream that overruns it is one whose headers claim more than its bytes can hold: malformed, whatever `Limits` the caller set) | `tests/errors.rs::malformed_input_is_invalid`, `tests/security.rs` (work budget) |
| `fmt.Errorf` for valid input the decoder does not handle: RGN, palettes, ICC/sYCC, HT, more than 4 components, precision over 16 | `Unsupported` | `tests/errors.rs::rgn_is_unsupported`, `tests/tier2.rs` |
| `ErrMemoryLimit` and `decode needs more than the memory limit`, the parse-time heap guard, pixel, input and output size limits | `LimitExceeded` (Go's `ErrMemoryLimit` also covers the transient wait for other decodes, which this crate has no counterpart for) | `tests/errors.rs::limits_are_limit_exceeded`, `tests/heap.rs`, `tests/security.rs` |
| `fmt.Errorf` sanity caps on the shape of the stream: precincts per resolution and per tile, code-blocks per precinct, the packet count of a tile, a code-block's accumulated coding passes | `LimitExceeded` (they are fixed caps, not a function of the input's size) | `tests/security.rs`, `tests/tier2.rs` |
| a failed allocation (Go: the runtime aborts) | `OutOfMemory` (the oracle prints `limit`) | none: `try_reserve` cannot be made to fail portably |
| `parallel.ErrStopped` and `jp2k: decode stopped: %w` of `ctx.Err()`, `tier2.ErrStopped` (`packet parse stopped`) | `Cancelled` (the flag replaces the context; no deadline error) | `tests/errors.rs::cancel_is_cancelled`, `tests/threads.rs` |
| a reader's own error, `io.ErrUnexpectedEOF` and `io.EOF` from the reader | `Io(kind, message)` | `tests/errors.rs::reader_failure_is_io` |
| `ErrInternal` (a recovered panic) | `Internal` | `src/parallel.rs::tests::panic_is_an_internal_error` |

## Operations and output files

Per input the oracle writes `<input>.<op>.out` (the description) or `<input>.<op>.err` (the error
class and a newline), for these `<op>`:

| op | call |
|---|---|
| `cfg`, `cfg1` | `DecodeConfig` on the bytes, and on a reader that returns one byte at a time |
| `big` | instead of everything below, when the declared image has more than 1<<22 pixels (1<<24 for `hostile-*`): its size |
| `dec` | `Decode` |
| `dec1` | `Decode` on a reader that returns one byte at a time (Go's `readAll` path) against `decode_reader` |
| `r0`..`r5` | `DecodeRegion` (`hostile-*`: `r0` and `r1`) |
| `c0`, `cA` | `DecodeComponents` of component 0, and of 0, 1, 2 |
| `cS`, `cJ` | components 0, 1, 2 with `SeparateRounding`, and with `JJ2000` |
| `cS0`, `cJ0` | component 0 with each |
| `rc0`, `rcA` | `DecodeRegionComponents` of region 2 (component 0) and region 3 (0, 1, 2, `SeparateRounding`) |
| `rcJ` | `DecodeRegionComponentsOptions` of region 3 (0, 1, 2) with `JJ2000` |
| `open`, `openS`, `openJ` | `Open`, and `OpenOptions` with `SeparateRounding` and with `JJ2000` (`hostile-*`: `open` only). One result per call: `<op>` is `Open` plus `Config` and `Levels`; `<op>0`..`<op>3` are `DecodeRegion` of regions 2, 3, 2 again (the second call runs on what the first built) and 0 (the whole image, last), `<op>c` is `DecodeRegionComponents` of region 4. A failed call does not stop the ones after it. `Open` compares only whether it fails (besides `limit` and `internal`): Go words its header errors differently. The Image's calls compare the error class, which is the message, in full |

`hostile-*` inputs skip `r2`..`r5`, `openS`, `openJ`, `cS`, `cJ`, `cS0`, `cJ0`, `rc0`, `rcA`, `rcJ`: they are cheap to fail,
not to run a dozen times. Every other operation runs on every input whatever `Decode` returned, so
a malformed tile-part (which fails only the regions that read it, in Go and here) and `Open`'s
stricter header walk are compared. There is no structural gate; a known divergence would be listed
here. There are no known divergences. `openJ` and `rcJ` (the `JJ2000` region and handle calls) are compared like
every other op. Go computes a region with a windowed inverse DWT (one buffer per level, each window the one above grown by 4
samples for 9/7 or 2 for 5/3, `internal/decoder/region.go`), so its `JJ2000` region can differ by 1 at an interior pixel from
the crop of its own whole decode; this crate does the same in every rounding (`partial_dwt` in `src/decoder.rs`) and so matches
Go's bytes, not the crop. Not compared: the message of an `Open` that fails
(Go's `Open` reads a skeleton of the file and words its header errors differently, so only the
fact of failure, `limit` and `internal` are compared); `DecodeConfig` takes a cancel flag on a
reader only (`decode_config_reader`), because Go's `DecodeConfig(io.Reader)` has no context and a
slice cannot stall, so `decode_config` on a slice keeps its fresh flag.

Go's `Open` also falls back to the package `DecodeRegion` when the headers it keeps would pass a
quarter of the process memory limit. `Limits` has no process-wide share to divide, so `Image::open`
falls back only where Go's header parse fails with its plain size error (the `hostile-flood-tp-*`
inputs), not at that quarter share.
