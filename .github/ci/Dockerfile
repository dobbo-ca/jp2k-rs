# CI image: every check runs inside this Dockerfile under BuildKit, so CI and local runs match.
# Dependencies (rayon) are fetched in the toolchain stage, before the source is copied, and every later
# stage builds offline from that layer (so the cargo registry is not a cache mount).
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu
WORKDIR /src
# The manifests and lockfiles of the crate and of the fuzz crate, with stub targets, so the fetch is
# a layer that only a change to them rebuilds.
COPY Cargo.toml Cargo.lock ./
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in $(sed -n 's#^path = "fuzz_targets/\(.*\)\.rs"$#\1#p' tests/fuzz/Cargo.toml); do echo 'fn main() {}' > tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo fetch --locked && cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml
ENV CARGO_NET_OFFLINE=true

FROM toolchain AS src
COPY . .

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The target dirs are cache mounts; each cargo RUN touches the sources first,
# so cargo never reuses stale artifacts. The x86_64 test runs set JP2K_EXPECT_AVX512=1 only when
# the host CPU has avx512f, so on such a host the AVX-512 tier is tested, not skipped.
FROM src AS ci
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo fmt --check
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings
# The NEON module is not compiled for x86_64, so lint the aarch64 target too.
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
# The fuzz targets must keep compiling (stable is enough to check them; fuzz.yml runs them).
RUN --mount=type=cache,id=jp2k-fuzz-target,target=/src/tests/fuzz/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo clippy --locked --manifest-path tests/fuzz/Cargo.toml --all-targets -- -D warnings
# The second run prints the tier the tests used ("SIMD tier: ..."), so the log shows what was tested.
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && JP2K_EXPECT_AVX512=$(grep -qw avx512f /proc/cpuinfo && echo 1 || echo 0) cargo test --locked --no-fail-fast \
    && cargo test --locked --lib -- --nocapture tier_is_detected avx512_where_expected
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && JP2K_EXPECT_AVX512=$(grep -qw avx512f /proc/cpuinfo && echo 1 || echo 0) JP2K_SIMD=scalar cargo test --locked --no-fail-fast
# The x86-64-v4 build (docs/performance.md, the bench's v4 column) must give the same bytes as the
# default one: the tests that pin or compare the output (the 9/7 rounding hashes, the conformance
# files against opj_decompress, the fixtures against their sources), released, with their own cache
# mount, since RUSTFLAGS changes every artifact. Skipped on a host without AVX-512 (SIGILL).
RUN --mount=type=cache,id=jp2k-target-v4,target=/src/target \
    if grep -qw avx512f /proc/cpuinfo; then \
    find src examples tests -name "*.rs" -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" \
       cargo test --locked --release --test rounding --test conformance --test decode; \
    else echo "no AVX-512: x86-64-v4 tests skipped"; fi
# The measured peaks (ledger, heap, threads) of the memory and thread tests, and the allocation
# counts of a reused Decoder, for the log.
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo test --locked --test heap --test threads --test reuse_alloc -- --nocapture --test-threads=1
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo build --locked --release --target aarch64-unknown-linux-gnu

# aarch64 tests on a native arm64 runner (ubuntu-24.04-arm), both SIMD tiers (NEON, then
# JP2K_SIMD=scalar): only the tests that the target changes. The unit tests hold the NEON kernel tests against scalar (lift, scale,
# lift_to/scale_to, ict, dequant, pack_u16) and the tier detection; tests/rounding.rs pins the 9/7 decode output, which runs
# through the NEON kernels. The full suite runs natively in `ci`; Go parity on aarch64 is the
# parity-arm64 stage.
FROM src AS arm64
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && cargo test --locked --no-fail-fast --lib --test rounding \
    && cargo test --locked --lib -- --nocapture tier_is_detected
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && JP2K_SIMD=scalar cargo test --locked --no-fail-fast --lib --test rounding

# Go parity: the Go oracle writes the inputs and decodes them; the Rust example decodes the same
# inputs; the two output trees must be equal. Go is built with CGO off, which is how the asm of its
# 9/7 kernels is used (AVX2+FMA on x86_64, NEON on arm64).
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
RUN --mount=type=cache,id=jp2k-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle .
# The same oracle built for x86-64-v4 (the bench's v4 column); the parity stages use /out/oracle.
RUN --mount=type=cache,id=jp2k-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && GOAMD64=v4 go build -o /out/oracle-v4 .

FROM go-oracle AS go-run
COPY tests/data tests/data
COPY tests/fuzz/seeds tests/fuzz/seeds
COPY tests/parity/bigtiles tests/parity/bigtiles
# The generated inputs: the fixtures, their mutations, JP2 files and the crafted codestreams, and
# tests/parity/bigtiles as it is (the oracles decode it whole and as component 0 only).
RUN /out/oracle -gen tests/data /inputs tests/fuzz/seeds && cp tests/parity/bigtiles/* /inputs/ \
    && ls /inputs | wc -l && du -sh /inputs
RUN /out/oracle /inputs /go-out && ls /go-out | wc -l && du -sh /go-out
# The subset the aarch64 oracle runs (stage parity-arm64): every input that is not a
# mutation (`~` in its name), and every 20th mutation. Go decodes it again, so the two trees have
# the same names.
RUN mkdir /sub && ls /inputs | awk '!/^bigtiles/ && (!/~/ || NR % 20 == 0)' | (cd /inputs && xargs cp -t /sub) \
    && /out/oracle /sub /go-sub && ls /sub | wc -l && ls /go-sub | wc -l

# The Go trees are bind mounts from go-run, not COPY: a COPY of the output tree after `COPY . .`
# would run on every commit. The Rust output trees (about 140000 small files, 555 MB with 4 KB
# blocks) are written to a tmpfs, not the BuildKit disk, which filled up once ("No space
# left on device"); the tree is deleted after each diff. The full /inputs set runs in the caller-pool
# default, scalar, nopool (no caller pool: the shared pool, which Options::default() gets) and
# serial (max_threads 1) modes, and with JP2K_SIMD=avx2 (the default tier is AVX-512 on an AVX-512 host,
# so this keeps the AVX2 kernels compared with Go); per-call workers (percall) run on the /sub
# subset (see go-run) to keep that run cheap. JP2K_ORACLE_THREADS picks the mode.
FROM src AS parity
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs \
    --mount=type=bind,from=go-run,source=/go-out,target=/go-out \
    --mount=type=bind,from=go-run,source=/sub,target=/sub \
    --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub \
    --mount=type=tmpfs,target=/rust-out,size=1g \
    find src examples tests -name "*.rs" -exec touch {} + && cargo run --locked --release --example oracle -- /inputs /rust-out \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs" \
    && [ "$(ls /go-out | wc -l)" -ge 50000 ] \
    && sh tests/parity/diffsum.sh /go-out /rust-out \
    && find /rust-out -mindepth 1 -delete \
    && JP2K_SIMD=avx2 cargo run --locked --release --example oracle -- /inputs /rust-out \
    && sh tests/parity/diffsum.sh /go-out /rust-out \
    && find /rust-out -mindepth 1 -delete \
    && JP2K_SIMD=scalar cargo run --locked --release --example oracle -- /inputs /rust-out \
    && sh tests/parity/diffsum.sh /go-out /rust-out \
    && find /rust-out -mindepth 1 -delete \
    && JP2K_ORACLE_THREADS=nopool cargo run --locked --release --example oracle -- /inputs /rust-out \
    && sh tests/parity/diffsum.sh /go-out /rust-out \
    && find /rust-out -mindepth 1 -delete \
    && JP2K_ORACLE_THREADS=1 cargo run --locked --release --example oracle -- /inputs /rust-out \
    && sh tests/parity/diffsum.sh /go-out /rust-out \
    && find /rust-out -mindepth 1 -delete \
    && JP2K_ORACLE_THREADS=percall cargo run --locked --release --example oracle -- /sub /rust-out \
    && sh tests/parity/diffsum.sh /go-sub /rust-out

# aarch64 against Go: the oracle example run natively on an arm64 runner (the NEON kernels) on
# the subset, once per SIMD tier; the output must equal Go's. Its output goes to a tmpfs too, and
# is deleted between the tiers.
FROM src AS parity-arm64
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/sub,target=/sub \
    --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub \
    --mount=type=tmpfs,target=/arm-out,size=512m \
    find src examples tests -name "*.rs" -exec touch {} + && cargo run --locked --release --example oracle -- /sub /arm-out \
    && echo "$(ls /sub | wc -l) inputs, $(ls /go-sub | wc -l) outputs" \
    && [ "$(ls /go-sub | wc -l)" -ge 5000 ] \
    && sh tests/parity/diffsum.sh /go-sub /arm-out \
    && find /arm-out -mindepth 1 -delete \
    && JP2K_SIMD=scalar cargo run --locked --release --example oracle -- /sub /arm-out \
    && sh tests/parity/diffsum.sh /go-sub /arm-out

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The
# workflow runs it with `docker run`, so a BuildKit cache hit cannot skip the fuzzing.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && CARGO_NET_OFFLINE=false cargo install cargo-fuzz --locked --version 0.13.1

FROM fuzz-tools AS fuzz
COPY . .
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz


# Benchmark (bench.yml): the Go oracle and the Rust bench example in one image, each built twice:
# the default build, and x86-64-v4 (GOAMD64=v4, RUSTFLAGS -C target-cpu=x86-64-v4 and the Zen 4/5
# false-dependency features, docs/performance.md) as oracle-v4 and bench-v4; tests/parity/bench.sh
# prints both. The default build also builds examples/kernels.rs (the kernel timings bench.sh prints
# on an AVX-512 host).
FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + \
    && cargo build --locked --release --example bench --example kernels \
    && cp target/release/examples/bench target/release/examples/kernels /usr/local/bin/
# Its own cache mount (RUSTFLAGS changes every artifact); the sources are touched first, since the
# mount is shared by every branch.
RUN --mount=type=cache,id=jp2k-target-v4,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --example bench \
    && cp target/release/examples/bench /usr/local/bin/bench-v4

# Where the time goes (profile.yml): the bench example with debug info under callgrind.
FROM toolchain AS profile-tools
RUN apt-get update && apt-get install -y --no-install-recommends valgrind \
    && rm -rf /var/lib/apt/lists/*

FROM profile-tools AS profile
COPY . .
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench-dbg

# perf (profile.yml, job perf): the bench example with line tables and the Go oracle in /out, which
# the job copies out and runs under perf on the runner (tests/parity/perf.sh).
FROM src AS perf
COPY --from=go-oracle /out/oracle /out/oracle
RUN --mount=type=cache,id=jp2k-target,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + && CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --locked --release --example bench && cp target/release/examples/bench /out/bench-dbg

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on the fixtures, rebuilds it with the profile, and, when BOLT is
# installed, once more through BOLT. The tools are installed before `COPY . .`, so a new commit
# does not install them again. BOLT comes from apt.llvm.org (bookworm has none of its own); if the
# repository or the package is missing the image still builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && CARGO_NET_OFFLINE=false cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle; tests/parity/pgo.sh times them. The training runs are
# `bench tests/data 5 <threads>`: 1 (one thread) and 0 (default threads on a reused pool). The
# profiles of an earlier run stay in the cache mount, so they are deleted first.
FROM pgo-tools AS pgo
COPY . .
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
RUN --mount=type=cache,id=jp2k-target-pgo,target=/src/target \
    find src examples tests -name "*.rs" -exec touch {} + \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- tests/data 5 1 \
    && cargo pgo run -- --locked --example bench -- tests/data 5 0 \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" tests/data 5 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi
