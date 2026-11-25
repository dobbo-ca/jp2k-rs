# Threading and dependencies

How `Limits::max_threads` and the worker pool work. The other limits are in [limits.md](limits.md).

- `max_threads` is the most OS threads one call uses, the calling thread included. 1 runs
  everything on the calling thread and spawns none (`tests/threads.rs` counts the threads);
  0 counts as 1. The threads are split over tiles (lanes), the code-blocks of a tile component,
  the rows and columns of the inverse wavelet, the colour transform and the output rows, and
  never more than this in total. The threads are shared out over the lanes as in Go (`threads / lanes` each, the first
  `threads % lanes` lanes one more). With as many lanes as threads each lane has one thread and
  is a job on the pool, which never waits there. With 2 or more lanes but fewer than threads, the
  lanes run on the caller and `lanes - 1` plain `std::thread::scope` threads started once for the
  call (not pool workers, so their wait for their nested passes blocks instead of taking another
  call's work), and each lane runs its code-block, wavelet, colour transform and output passes on
  the pool with its share; the lane threads and the pool workers together stay within the
  limit. If a lane thread cannot be started the call runs with fewer lanes. With one lane the
  caller takes the tiles one after another, each using all the threads. The lane count is limited
  by the threads, the tiles, the input size and the memory the limit leaves (at most 512 MiB, and
  half of what is left, for the lanes past the first, at about 24 bytes a sample of a tile; Go
  stops at 64 MiB and a quarter of the output, so a call on big tiles runs more lanes here), and
  then lowered to the fewest lanes that take as many rounds of tiles (32 tiles on 15 lanes are 3
  rounds, the last of 2 tiles; 11 lanes are the same 3 rounds). The output is the same bytes for
  every value.
- Threads come from a pool, so small images use them too (a pass is worth a worker from about
  32 K samples; Go starts a goroutine per item with no threshold):
  - `Options::pool` (a `ThreadPool`, made once with `ThreadPool::new(n)` and kept) always wins.
    Clones share the workers, and calls on other threads can share one pool.
  - Without one, a call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock`,
    created lazily on the first such call, with `std::thread::available_parallelism()` threads
    and never more. It keeps no other state. `max_threads: 1` never touches it (it is not even
    created) and starts nothing.
  - Each call takes at most `max_threads` threads from the pool, counting the caller. A pool of
    2 threads or fewer gives a call all of them; a larger pool gives it at most its size minus 1
    (a pass never wants every CPU of a pool sized to them). So with the default `max_threads` on
    the shared pool, a host with more than 2 CPUs gets at most `available_parallelism - 1`
    threads per call.
  - If the shared pool cannot be created, the call starts one set of workers of its own at
    entry, sized from the input (none for a small one, never more than `max_threads`), and every
    stage shares them. `tests/threads.rs` forces this with a hidden `Options` field.
  The pool changes speed only, not the bytes.

## Dependencies

One crate, `rayon`, with what it pulls in (`rayon-core`, `crossbeam-deque`, `crossbeam-epoch`,
`crossbeam-utils`, `either`; exact versions in `Cargo.lock`). All are pure Rust: no C, no `-sys`
crate, no build script that compiles anything. It is here only for `ThreadPool`: safe Rust cannot
reuse worker threads for jobs that borrow short-lived buffers (`std::thread::scope` must start
and join its threads on every call, tens of microseconds each), and rayon's scoped pool does. It
backs the caller's `Options::pool`, the lazy shared pool and a call's own fallback workers (see
[limits.md](limits.md) and the sections above).
