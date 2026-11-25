# Performance

How to build and call this crate for speed. The measured numbers are in [benchmarks.md](benchmarks.md).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. For a worker that decodes tiles:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the kernels and the loops that call them inline across
  modules and across the crate boundary. A dependent that only wants the one-unit build of this
  crate can set `[profile.release.package.jp2k] codegen-units = 1`.
- `panic = "abort"` removes the unwinding tables. It also changes one thing here: the crate catches a
  panic on any thread and returns `Error::Internal` ([threading.md](threading.md)), and with `abort`
  the process stops at the panic instead. The crate does not panic on any input (clippy denies
  `unwrap`, `expect` and `panic` in its non-test code), so this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 (F, BW, CD, DQ, VL) everywhere, not only in
  the kernels. **Use it only when every machine that runs the binary has AVX-512.** On a CPU without
  it the program dies with `SIGILL` (illegal instruction) at the first such instruction, with no
  message. The numbers here were measured on an AMD Ryzen 7 9800X3D, which has AVX-512; GitHub-hosted
  runners may not. The
  default build needs no flag and runs everywhere. Do not set it for an aarch64 build.
- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` is part of the v4 recommendation. Zen 4 and
  Zen 5 make `tzcnt`, `blsr`, `blsi` and `blsmsk` wait for the last write to their output register,
  and LLVM breaks that wait only under `target-cpu=znver4`/`znver5` tuning, not under `x86-64-v4`.
  A Huffman decoding loop in another Rust crate measured 30 to 50% slower without the two features; the v4 column
  here is built with them, and whether this crate's loops need them is not measured.
  rustc prints a warning that it does not know the two features and passes them to LLVM, which
  does. `-C target-cpu=znver5` gives the same speed but runs only on Zen 5.
- The library keeps its runtime SIMD dispatch: the kernels pick a tier at first use, scalar,
  AVX2+FMA, AVX-512 or NEON, for the 9/7 kernels and the integer kernels of the 5/3 path (the tier
  table is in [safety.md](safety.md)). So the default build already runs the AVX-512 kernels on
  any machine that has AVX-512. The gain of the v4 build
  comes from the compiler using the wider baseline in the code around the kernels (tier-1 and
  tier-2 loops, the colour and output copies), not from new kernels. The output is the same: `ci` runs `tests/rounding.rs`, `tests/conformance.rs` and
  `tests/decode.rs` under the v4 build, released.

## What the v4 column measures

`bench.sh` prints two more tables, "v4 build, default threads" and "v4 build, single thread": the
same comparison as the default-build tables, with Go built with `GOAMD64=v4` (`oracle-v4`) and Rust
built with the flags above (`bench-v4`, its own cargo cache mount). It is the number for a
consumer that follows the profile on this page. The two tables are not gated: the 1.10 gate is for
the default build, which every consumer has. The numbers are in [benchmarks.md](benchmarks.md).

## Reusing a `Decoder`

Keep one `Decoder` for each worker thread and decode every tile on it: `Decoder::new()`, then
`decode`, `decode_components`, `decode_region` and `decode_region_components`, or
`decode_image_region` and `decode_image_region_components` on an opened `Image` (one `Image` can
serve several threads, each with its own `Decoder`). Each call takes its own `Options` and returns
what the free function of the same name returns, with the same bytes and the same errors. The
result is borrowed from the `Decoder` until its next call, so write it out (or clone it) first.

A `Decoder` keeps what a call allocates: the output pixels or planes, the tile-component planes and
wavelet levels, the tier-1 workers and the wavelet lines, the tile geometry with its tag trees, the
packet, code-block and segment lists, the vectors of the parsed headers, and a region call's compact
copy of the file. A later call of the same shape takes them back instead of allocating; a larger
image grows them once and a smaller one uses a part. The free functions run the same code with a
scratch of their own that they drop, so they allocate and free as before.

- With `Limits::max_threads` 1, the second decode of the same image makes no call to the allocator:
  whole, as components, as a region at reduce 0, 1 and 2, and as a region of an opened `Image`; 8
  and 16 bits, 5/3 and 9/7, one tile and several (edge tiles included), several layers, every
  code-block style, and a JP2 file. `tests/reuse_alloc.rs` counts the allocator calls and asserts
  zero.
- With more threads on a caller's pool (`Options::pool` and `max_threads` above 1), a call still
  makes a few allocations that are not its buffers: the boxed task rayon makes for each extra worker
  of each parallel pass, and the column groups of the threaded wavelet. Their count is bounded and
  does not grow with the image (22 for a 256x256 RGB tile on 4 threads); `tests/reuse_alloc.rs`
  asserts at most 64 and prints the count of each case. A call that runs several tile lanes on threads of its own (many large tiles) starts
  those threads each time. `max_threads` 1 is the setting that guarantees no allocation.
- Every buffer is still charged to the call's ledger, where and as much as a new one would be, and
  then taken from the `Decoder` instead of allocated. So the limits, the errors and the bytes are
  the free function's, and so is the ledger peak (`jp2k::stats::last`), except where a `Decoder`
  keeps a buffer charged that the free function frees: planes past a lane's three spares (tiles
  that grow), a region's lower wavelet levels, wavelet lines a threaded transform shed, kept for a
  later take until the call ends, and a lane's tier-2 lists, kept for its next tile up to 256 KiB
  and charged what they hold (by capacity) until it takes them. So the call holds nothing its
  ledger released but four small buffers of fixed bound: a tier-1 worker's join up to 64 KiB, a
  lane's row buffers up to 64 KiB, a tile component's code-block lists up to 64 KiB (by
  capacity), and the room the kept tier-2 lists hold beyond what the tile taking them charges
  (at most 256 KiB). Between calls a lane keeps at most 256 KiB of tier-2 lists, and frees bigger
  ones when the call ends. Where it keeps more, the ledger peak is higher, by at
  most those buffers (see [limits.md](limits.md)). `tests/reuse.rs` checks over every fixture, one `Decoder`
  for all of them in turn, that the peak is at least the free function's and under twice it, and
  for several images that it, and the least `max_alloc` a decode needs, is the same.
- Some streams still allocate on every call: PPM or PPT packet headers, several POC records,
  COD, COC, QCD or QCC markers in tile-part headers, TLM markers, a tile in several tile-parts (its
  data is joined), a code-block whose segments spanning several layers join to over 64 KiB (the
  tier-1 worker frees that join after the block, an error of the block included), a tile
  component whose code-block lists hold over 64 KiB (a hundred code-blocks or more, or many
  layers: they are freed after the component), row buffers over 64 KiB (a tile row of over 16384
  samples on one thread: they are freed after the component), with several tiles a tile whose tier-2 lists hold over 256 KiB (about a
  thousand code-blocks, or many packets: they are freed after the tile), a JP2 `cdef`
  box, subsampled components (their canvases, and the lanes' buffers, which the call frees before
  it upsamples), and parse warnings. So do errors (their messages, and the lanes' buffers, which a
  failed call frees), `decode_reader`, `decode_config` and `Image::open`.
- What a `Decoder` keeps stays with it until it drops: see [limits.md](limits.md).

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
worker that links this crate); the library does not change. `pgo.yml` (dispatch only) measures how
much they gain on the bench example: it builds the example with `cargo-pgo`, trains it on the
fixtures in `tests/data`, then times the default, the PGO and the PGO plus BOLT binaries with
`tests/parity/bench.sh` and prints the table in the job summary. BOLT is optional there: if the
package is not available the job reports PGO only.

An earlier `pgo.yml` run (with the BOLT fix), Rust
ms/Mpixel, the change against the default build, best of three runs per file and build (the full
tables are in the job summary):

| table | input op | default | PGO | PGO + BOLT |
|---|---|---|---|---|
| single thread | all dec | 11.630 | 11.580 (-0.4%) | 11.392 (-2.0%) |
| single thread | all c0 | 7.413 | 7.419 (+0.1%) | 7.280 (-1.8%) |
| single thread | style dec | 53.061 | 50.595 (-4.6%) | 49.753 (-6.2%) |
| single thread | plt dec | 41.637 | 39.313 (-5.6%) | 38.512 (-7.5%) |
| default threads | all dec | 2.623 | 2.639 (+0.6%) | 2.730 (+4.1%) |
| default threads | all c0 | 1.665 | 1.746 (+4.9%) | 1.718 (+3.2%) |
| no pool | all dec | 2.714 | 2.873 (+5.9%) | 2.355 (-13.2%) |
| no pool | all c0 | 1.723 | 1.824 (+5.9%) | 1.505 (-12.7%) |

Read the single-thread rows: PGO gains nothing on the whole set (-0.4%) and BOLT adds about 2%; the
classes that gain most are the code-block-heavy style and plt fixtures (5 to 8%). The two
multi-threaded tables move by +6% and -13% for the same PGO and BOLT code, in both directions and
the same size, which is the spread of thread scheduling on a shared runner, not an effect of the
build. The training input is the benchmark's own fixtures, so these gains are an upper bound for
another workload, and a gain of 2% is not worth a PGO build step in a consumer.

## The v4 build

An earlier `bench.yml` run, Rust ms/Mpixel, "all" rows, default
build then v4 build, and the ratio to Go's build of the same kind:

| table | op | default | v4 | Rust / Go default | Rust v4 / Go v4 |
|---|---|---|---|---|---|
| default threads | dec | 2.577 | 2.391 (-7.2%) | 0.79 | 0.84 |
| default threads | c0 | 1.645 | 1.523 (-7.4%) | 0.65 | 0.67 |
| single thread | dec | 11.609 | 10.770 (-7.2%) | 0.93 | 0.90 |
| single thread | c0 | 7.435 | 7.122 (-4.2%) | 1.03 | 1.02 |

The v4 build is about 7% faster for the whole-image decode. Go gains about as much from
`GOAMD64=v4` (default threads dec 3.250 to 2.846), so the ratio to Go moves little.
