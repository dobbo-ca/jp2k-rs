# jp2k-rs

JPEG 2000 decoder (raw J2K codestreams and JP2 files), decode only, in Rust. A port of
[dobbo-ca/jp2k-go](https://github.com/dobbo-ca/jp2k-go), which has the design. Pure
Rust, one dependency (`rayon`, for the worker pool); `unsafe` only in the SIMD kernel modules.
Every Go package and public entry point is here, with Go's tests and fixtures, and the Go parity
job, the fuzz targets and the benchmark.

```toml
jp2k = { git = "https://github.com/dobbo-ca/jp2k-rs", rev = "<sha>" }
```

## Quick start

```rust,no_run
use jp2k::{Limits, Options, Rect, ThreadPool};
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("image.j2k")?;

    // Once, at startup, and kept: a call given this pool uses the caller plus one worker of its
    // 2 threads (max_threads 2) instead of starting threads (see "Resource limits").
    let pool = ThreadPool::new(2)?;

    // Explicit limits: at most 2 threads and 768 MiB of working memory for any one call.
    let opts = Options {
        limits: Limits { max_threads: 2, max_alloc: 768 << 20, ..Limits::default() },
        pool: Some(pool),
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);

    let img = jp2k::decode(&bytes, &opts, &cancel)?;                       // Raster
    let cfg = jp2k::decode_config(&bytes, &opts.limits)?;                  // size and pixel format only
    let rect = Rect { x0: 4096, y0: 4096, x1: 5120, y1: 5120 };
    let tile = jp2k::decode_region(&bytes, rect, 2, &opts, &cancel)?;
    let planes = jp2k::decode_components(&bytes, &[0, 2], &opts, &cancel)?; // Vec<Plane>, 16-bit samples

    // Many regions of one file: parse the headers once.
    let image = jp2k::Image::open(&bytes, &opts, &cancel)?;
    let region = image.decode_region(rect, 1, &cancel)?;
    Ok(())
}
```

## Resource limits

Every call takes `Limits` (no unlimited entry point), with finite defaults: `max_alloc` 4 GiB
(peak working memory of one call), `max_threads` the available parallelism, `max_pixels` 2^28,
`max_width` and `max_height` `u32::MAX`, `max_input` 4 GiB, `max_output` 2 GiB. `max_threads = 1`
runs everything on the calling thread and spawns none. A caller's `Options::pool` always wins;
without one, a call with `max_threads` above 1 uses the crate's lazily created shared pool.

## Docs

[limits](docs/limits.md), [threading and dependencies](docs/threading.md),
[safety and SIMD](docs/safety.md), [parity with Go](docs/parity.md) (and the tests),
[fuzzing](docs/fuzzing.md), [benchmarks](docs/benchmarks.md), [performance](docs/performance.md),
[differences from Go and format support](docs/go-differences.md).
