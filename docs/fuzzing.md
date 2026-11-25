# Fuzzing

`tests/fuzz/` is a cargo-fuzz (run with `cargo +nightly fuzz run --fuzz-dir tests/fuzz <target>`) crate: `decode` and `decode_limits` (`decode`, `decode_config`,
`decode_components` and the limits: tiny `Limits` drawn from the input, and a decode that works
stays within them), `reader` (the reader entry points with short reads equal the slice ones),
`region` (`decode_region`, and one `Image` opened once and used twice, equal to the one-shot
calls), `components`, `codestream` (`parse`, `jp2_codestream`), `markers` (SIZ, COD, PLT), `tier2`
(packet headers over several tiles with one `Cursor`), `tier1` (code-blocks through a `State` that
is reused for every input, against a fresh one, in the sign-magnitude, the reversible (5/3) and the irreversible write-out) and `mq` (the MQ decoder through `jp2k::fuzz::mq_decisions`), `threads` (`decode`,
`decode_components`, `decode_region` and `Image::open(..).decode_region` with a shared `ThreadPool`, the crate's shared pool or per-call workers and
1, 2, 8 or the default `max_threads`, equal to `max_threads` 1) and `simd_kernels` (every SIMD tier against scalar,
bit for bit, the twelve kernels including `pack_u8x4` and the integer ones of the 5/3 path; on x86_64 the tiers are AVX2 and AVX-512, and
fuzz.yml sets `JP2K_EXPECT_AVX512=1` when the runner's CPU has AVX-512 (GitHub-hosted runners
vary), so on such a runner a missing AVX-512 tier fails instead of fuzzing AVX2 only; also run
natively on an arm64 runner (`fuzz-arm64`, for the NEON kernels)). Seeds are `tests/fuzz/seeds` (jp2k-go's
`testdata/fuzz` entries) and the fixtures. `.github/workflows/fuzz.yml` runs them weekly (Tuesday
06:23 UTC) and on dispatch (`max_total_time` per target, default 600 s) with `-rss_limit_mb=2048
-malloc_limit_mb=512`; the decoders also run with `JP2K_SIMD=scalar`. The kernel hooks the fuzz
crate calls are `jp2k::fuzz::simd` (hidden from the docs, not API).
