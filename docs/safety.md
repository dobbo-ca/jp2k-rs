# Safety and SIMD

`unsafe` only in the SIMD kernel modules (`src/dsp/simd/`), plus two counting `GlobalAlloc`s, test-only exceptions: `tests/heap.rs` (it measures the real heap peak that `max_alloc` is held to) and `tests/reuse_alloc.rs` (it counts the allocator calls of a `Decoder`'s second decode, which must be zero).

## SIMD

The f32 kernels of the 9/7 path have AVX2+FMA and AVX-512 (x86_64) and NEON (aarch64) tiers beside
the scalar reference, picked once at first use by runtime detection (the AVX-512 tier needs avx2,
fma, avx512f and avx512bw); `JP2K_SIMD=scalar` forces scalar, and `JP2K_SIMD=avx2` caps the tier at
AVX2 (on aarch64 it changes nothing); `JP2K_SIMD=avx512` panics on a CPU without AVX-512 (for CI under an emulator). They are
bit-exact with scalar (`Rounding::Fused` uses a fused multiply-add, the others a separate multiply
and add), so output never depends on the tier. Eight kernels: one lifting step (`d += c*(u+v)`), the
band scale, `lift_to` and `scale_to` (the lift and the scale writing to a separate output), the inverse ICT, the
irreversible dequantization of a code-block row, `pack_u16` (the 16-bit output samples,
AVX2, AVX-512 and NEON) and `pack_u8x4` (an 8-bit output channel written straight into the
4-byte RGBA pixels, the other three bytes kept; AVX2 and AVX-512, scalar on aarch64). The
AVX-512 tier has its own kernel only where it beat AVX2 (below); for the others it runs the
AVX2 one. The 5/3 path has four integer kernels, with AVX2 versions (scalar on aarch64; AVX-512
for `dequant_rev` only): the inverse RCT, the two 5/3 lifting steps (`lift53_sub`, `lift53_add`) and the
reversible dequantization of a code-block row (`dequant_rev`); their adds and subtracts wrap as
`wrapping_*` does and their shifts are i32 `>>`, so they equal scalar for every input. The 9/7
and 5/3 rows are lifted on the split low/high bands (as Go does for 9/7) and the columns 64 at a
time as rows of a block, so the lifting steps run over contiguous memory; the 5/3 line transform
on interleaved samples is kept as the test oracle. Differences from Go's asm: no SSE2
tier, no packed-column `liftN`/`scaleN` kernels (the block rows are long enough for per-row calls),
and the JJ2000 rounding stays scalar. `cargo test` checks every available tier against scalar on pseudo-random inputs and edge
lengths, and asserts AVX2 (x86_64) or NEON (aarch64) was detected; with `JP2K_EXPECT_AVX512=1`
(set by CI only when the host CPU has AVX-512) it also asserts AVX-512. CI runs the tests both ways,
lints the aarch64 target, and runs the unit tests and `tests/rounding.rs` on a native arm64 runner, with the NEON tier and with `JP2K_SIMD=scalar` (job `arm64`; the full suite runs on x86_64 in `test`).
SIMD tiers in CI: scalar, AVX2 (x86_64) and NEON (aarch64) natively; the AVX-512 kernels under Intel SDE
(job `avx512-sde`: the unit tests on `-spr` with `JP2K_SIMD=avx512`, which fails on a CPU without AVX-512),
and natively when the hosted runner has AVX-512.

### Tiers and `unsafe`

| Tier | Detected by | Kernels of its own | `unsafe` |
|---|---|---|---|
| scalar | always | all (the oracle) | none |
| AVX2 | `avx2`, `fma` | all twelve | `avx2::mem` (loads and stores of `&[T; 8]` / `&[u8; 32]`), the dispatch arms |
| AVX-512 | `avx2`, `fma`, `avx512f`, `avx512bw` | `dequant`, `pack_u16`, `pack_u8x4`, `dequant_rev` | `avx512::mem` (loads and stores of `&[T; 16]` / `&[u8; 64]`), the dispatch arms |
| NEON | `neon` | the seven f32 kernels but `pack_u8x4` | `neon::mem` (loads and stores of `&[T; 4]`), the dispatch arms |

The kernels are safe `#[target_feature]` functions with no `unsafe` inside. An AVX-512 kernel
runs whole groups of 16 and hands the rest to the AVX2 kernel (at most one group of 8, then
scalar), so it never needs a masked load or store; it uses no AVX512DQ instruction (the float
`xor` and `and` are integer ones on casts, or masked moves). Each dispatch arm calls a kernel in
an `unsafe` block whose SAFETY note names the proof: `Tier::Avx512` holds a `Proof` built only in
`available()` after detecting the four features.

### Measured and dropped

A kernel's AVX-512 version stays if it takes at most 0.90 of the AVX2 time at the length a decode
mostly calls it with (`examples/kernels.rs`, ns per call, one thread), and no class of the whole
decode gets more than 1.02 slower. Measured in two bench runs (the second
with nothing else running on the machine). These lost, so their AVX-512 code was deleted and the AVX-512 tier runs the
AVX2 kernel:

| Kernel | Length | avx512 / avx2 | Why |
|---|---|---|---|
| `lift` | 64-float rows of a 64x1024 block (the column pass); 512 (row bands) | 1.13; 0.95 | three loads and a store per vector: bound by loads and stores, not arithmetic |
| `lift_to` | 64 (`sweep_low`); 64x1024 | 0.96; 1.01 | as `lift` |
| `scale` | 64; 512 | 1.41; 1.68 | one load, a multiply and a store: no arithmetic to win on |
| `scale_to` | 64 (`sweep_low`) | 0.88, then 0.95 | as `scale`; at 64 floats the gain is under 0.5 ns a call and did not hold |
| `ict` | 1024 (tile rows) | 1.05 | three loads and three stores per vector; wins only at 2040 (0.83) |
| `rct` | 1024 (tile rows) | 1.01 | as `ict`: three loads and three stores per vector, two adds and a shift |
| `lift53_sub` | 64-int rows of a 64x1024 block (the column pass); 64 | 0.93; 1.03 | as `lift`; wins on long row bands only (0.71 at 512) |
| `lift53_add` | 64; 512 | 1.01; 0.85 | as `lift53_sub` |

Kept: `dequant` 0.86 at 64 (a code-block row; 0.72 at 1024), `pack_u16` 0.44
and `pack_u8x4` 0.46 at 1024 (a tile row), and `dequant_rev` 0.72 at 64.

The bench of record on `main` (the final kernel set) confirms
both conditions for every kept kernel. In ns per call: `dequant_rev` 0.72 at 64, `dequant` 0.86 at
64, `pack_u16` 0.45 and `pack_u8x4` 0.48 at 1024. Its whole-decode "AVX-512 against AVX2" tables
have no class above 1.005 at one thread (plt `c0`) and none above 1.002 at default threads (other
`c0`). The 1.02 condition is judged on the single-thread table, because the default-threads table
moves with scheduling between runs. An earlier run showed 1.04 to 1.12 at default threads on 9/7
classes, where `dequant_rev` never runs, and was within 1.007 at one thread. A branch run
showed 1.020 and 1.021 on "all" at default threads. The single-thread table isolates the kernels.
Here both tables meet the condition.

The AVX2 versions of the integer kernels stay: against the scalar loops (which the compiler
vectorises with SSE2 at most) they take 0.58 (`rct`), 0.58 (`lift53_sub`) and 0.50 (`lift53_add`)
of the time at 1024, and 0.12 (`dequant_rev`) at 64.

### What each gate covers

The 5/3 kernels (`rct`,
`lift53_sub`, `lift53_add`, `dequant_rev`, AVX-512 included) are covered only by fixture parity
(the `parity` stage, on the default tier (AVX-512 on a host that has it), with `JP2K_SIMD=avx2` and with
`JP2K_SIMD=scalar`), the cross-tier tests and fuzzing.
