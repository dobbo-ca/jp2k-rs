# Resource limits

The README has the short version; this page is the full text. Threads and pools are in [threading.md](threading.md).

Every call takes `Limits` (in `Options::limits`, or directly for `decode_config` and
`decode_config_reader`); there is no unlimited entry point. A constrained call, which never uses
more than one thread or 64 MiB:

```rust,no_run
use jp2k::{Limits, Options};
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("image.j2k")?;
    let cancel = AtomicBool::new(false);
    let opts = Options {
        limits: Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() },
        ..Options::default()
    };
    let img = jp2k::decode(&bytes, &opts, &cancel)?;   // no thread is spawned
    Ok(())
}
```

| field | default | unit | error when exceeded |
|---|---|---|---|
| `max_alloc` | 4 GiB | bytes | `Error::LimitExceeded` |
| `max_threads` | `std::thread::available_parallelism()`, or 1; a call uses at most the pool's size (see [threading.md](threading.md)) | threads | none: the call uses fewer |
| `max_pixels` | 2^28 (Go's) | pixels of the result | `Error::LimitExceeded` |
| `max_width`, `max_height` | `u32::MAX` | pixels of the result | `Error::LimitExceeded` |
| `max_input` | 4 GiB | bytes a reader entry point reads | `Error::LimitExceeded` |
| `max_output` | 2 GiB (Go's 8 * 2^28) | bytes of the result | `Error::LimitExceeded` |

- `max_alloc` is the PEAK working memory of one call, all its threads together. Every allocation
  that scales with the input is charged to one ledger before it is made and given back when it
  is dropped, so a long call does not leak budget and a caller in a 1 GiB pod can set
  `max_alloc: 768 << 20` and stay under it. It is not an estimate: `tests/heap.rs` runs a
  counting global allocator and holds the real heap peak to the limit, for several limits and
  thread counts. The input slice is the caller's and is not charged; a reader's bytes are. A
  decode that needs more fails with `Error::LimitExceeded` before it allocates. With several
  threads, each extra thread's scratch is taken only if it fits, and tile lanes only as far as the
  remaining limit allows, so a limit that one thread meets is normally met with more.
- A `Decoder` ([performance.md](performance.md)) keeps the buffers of its calls from one call to
  the next. Between calls they are not charged to any call: a call charges each one as new when it
  takes it, so the process holds the kept buffers it has not taken on top of `max_alloc`. That is
  usually about what the largest call the `Decoder` made needed: for a 256x256 RGB tile on one
  thread, about 1.1 MB counted from the buffer sizes (the 256 KiB output, three planes of 4 bytes a
  sample, a tier-1 worker's 64 KiB, the wavelet lines and the tile geometry), and with more threads
  a tier-1 worker, wavelet lines and a row buffer per extra thread. The bound is larger, because a
  kept plane goes to the request it fits best, so a small request can take a large one and the
  next requests allocate new ones: kept memory is at most the number of lanes a call ran (one per
  thread at most) times the buffers a lane holds at once (its planes and wavelet levels, its
  tier-1 workers, wavelet lines and row buffers) times the largest plane any call took, plus the
  last call's output (pixels or component planes) and its parsed headers, plus fixed bounds a
  lane: its tier-2 lists, at most 256 KiB by capacity (bigger ones are freed when the call ends,
  whatever kept them), its code-block lists and their index, at most 64 KiB by capacity, each
  tier-1 worker's join, at most 64 KiB, and its row buffers, at most 64 KiB. A
  kept buffer taken for a smaller request is charged for the request; its extra capacity is part
  of that bound.
- Within a call, nothing the ledger released stays held, uncharged, while the call goes on, but
  four small buffers of a `Decoder`'s lane, each with a fixed bound:
  - A plane or wavelet level the call is done with stays charged while the lane keeps it for a
    later take: the three spare planes of each kind, as in the free functions, and, in a
    `Decoder` only, the planes past those three and a region's lower wavelet levels, which the
    free functions free. A later take reuses one with its charge; the rest are charged until the
    lanes end. So a `Decoder` call's ledger peak is at least the free function's, and it is higher
    by at most the bytes of those planes and levels the lanes hold, not taken again, at the free
    function's peak. That happens in region decodes whose windows leave out part of a tile (the
    lower levels) and in images whose tiles grow, so a lane finishes with more than three planes
    (a small first tile, say); there a `Decoder` needs a larger `max_alloc` than the free
    function, under twice it in `tests/reuse.rs`'s fixtures. A whole image or components of equal
    tiles needs the same, and `tests/reuse.rs` checks that exactly for several fixtures.
  - In a `Decoder`, a lane's tier-2 lists (the tile geometry, the packets, their sequence and
    queue, the code-block records and their segment lists) stay after its tile for the lane's
    next tile, charged what they hold, by capacity, until that tile's tier 2 takes them; then it
    charges what it uses, as for new lists. Tier 2 gives each list to the same packet or
    code-block of the next tile, so a list can hold room that tile does not use (a packet that
    had many code-blocks and now has none): counting by capacity charges that room too. The lists
    are kept only when they hold at most 256 KiB and the call's ledger has already been as high
    as it is with their charge, so the charge does not raise the call's ledger peak; else they
    are freed when the tile is done, as the free functions free them. After the call's last tile
    on one lane the lists are kept uncharged, as between calls, again only up to 256 KiB: the
    lanes end with that tile.
    The ledger peak can still be higher than the free function's, by at most 256 KiB a lane,
    where something else is charged while the lists are: other lanes' tiles, with several, and
    the tier 2 of a tile a region decode skips (it does not take them).
  - In a `Decoder`, the wavelet lines a transform sheds (with several threads, a transform that
    uses fewer threads than the one before) stay charged until a later transform takes them or
    the call ends; the free functions free them. With one thread the only line shed is one too
    small for the next transform, which takes it back at once, and with several the ledger peak
    can be higher by the lines shed.
  - The four uncharged buffers: a tier-1 worker keeps the joined segments of a code-block only
    up to 64 KiB, and a bigger join is freed when its block ends, an error of the block included;
    a lane keeps its row buffers only up to 64 KiB of capacity, and frees the rest when a tile
    component's rows are written; a lane keeps a tile component's code-block lists (its groups,
    their index, its tier-1 jobs and their corners) for the next component only when they hold at most 64 KiB,
    by capacity (the index's estimated from its capacity), and frees bigger ones when the
    component is decoded, dropping the index as soon as the lists and it hold over 64 KiB, as
    the free function drops it (charging them between components would raise the ledger while the
    component's rows are written, which the free function does not hold); and while a tile's
    tier 2 uses the lane's kept tier-2 lists, the room they hold beyond what it charges is not
    charged, at most 256 KiB.
  - The lanes are kept, uncharged, only when nothing is charged after them: a call that upsamples
    subsampled components, fails, or retries on one thread after a `LimitExceeded` frees its
    lanes first, as the free functions do. Lanes a panic left (one thread, so no pool caught it)
    are freed when the next call starts, before it charges anything.

  `tests/heap.rs` holds a `Decoder` call's heap peak to the free function's plus 128 KiB (a
  code-block join, subsampled components, an error before a call, a heavy tier 2 and a wide row
  buffer before a bigger tile, tier-2 and code-block lists that grow at a new position each tile,
  a heavy tier 2 before light tiles), and its heap over its own ledger to the free function's over
  its ledger plus 128 KiB (a region's lower levels, those five streams of tiles). It also holds
  what one `Decoder` keeps after each of many calls, whose tier-2 lists grow at a new position
  each call, to what a fresh one keeps plus 256 KiB of tier-2 lists and 64 KiB of code-block
  lists. What a `Decoder` keeps is freed when it drops.
- Cancellation: set the `AtomicBool` passed to the call from any thread. Every thread of the
  call checks it per tile, code-block and row, stops, and the call returns `Error::Cancelled`
  once they have all returned. This holds when `decode` is called from a thread that is not a
  worker of another rayon pool. From such a worker (inside a `par_iter` of the caller's own
  rayon pool, say), a wait inside the call may run that pool's other jobs, such as another decode,
  so the time to return after a cancel is not bounded there: call from a plain thread. The first error of a call stops new work on every thread, and the
  error returned is the one a serial run gives.
- A panic on any thread is caught and returned as `Error::Internal`.
- Other errors: `Error::Invalid` (malformed input), `Error::Unsupported` (valid input this
  decoder does not handle), `Error::OutOfMemory` (an allocation failed), `Error::Io` (a reader
  failed).

## Other entry points and options

`decode_reader` and `decode_config_reader` take a `std::io::Read`. `Raster` has the byte layout
of Go's `image` package (`Gray`, `Gray16`, `RGBA`, `RGBA64`, `NRGBA`, `NRGBA64`; 16-bit channels
big-endian); `Plane` holds `u16` samples. `Options { separate_rounding, jj2000, limits, pool }` has four
documented fields and a hidden `per_call_pool` field for tests (a struct literal needs
`..Options::default()`): the two rounding flags are Go's `SeparateRounding` and `JJ2000`, bit-exact with Go
(`tests/rounding.rs` pins the SHA-256 of every fixture in each mode); `limits` and `pool` replace
Go's `Workers` and memory budget (see above).
