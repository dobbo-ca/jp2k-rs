# Differences from Go

## Format support

progression orders, POC, PPM/PPT, every code-block style bit, component subsampling, signed
components, and JP2 with `jP`, `ihdr`, `colr` (sRGB, greyscale) and `cdef`. RGN, palettes, ICC
and sYCC colour, HT block coding, `MCT=2` and more than 4 components (except through
`decode_components`) are an `Error::Unsupported`.

- Go's `context.Context` is a cancel flag (`&AtomicBool`), checked per tile, code-block, DWT
  level and packet batch, and returns `Error::Cancelled`.
- Go's process-wide memory budget (`SetMemoryLimit`, decodes that wait for each other, the
  admission queue, `ErrMemoryLimit`) is replaced by `Limits`, charged per call: `max_alloc`
  (the peak working memory of the call, all threads together: parse, tile geometry, sample planes,
  every thread's scratch, output; charged to a ledger before each allocation, which is made with
  `try_reserve`, and released when dropped), `max_threads`, `max_pixels` (Go: 1 << 28), `max_width`, `max_height`, `max_input`
  (reader entry points) and `max_output`. Defaults are finite.
- `decode_config` on a byte slice skips a box that runs past the end without error, as Go does on
  its seekable `bytes.Reader` (the next read finds the end), and a reader does not; both are as in Go.
- Go's `DecodeRegion` and `Open` index tile-parts lazily and check headers more strictly than
  `Decode`. `decode_region` and `Image::open` parse every tile-part header, but a malformed SOT
  (TPsot out of order, TPsot >= TNsot, more than 256 tile-parts, a repeated Zppt) fails only the
  regions that read that tile, and an out-of-range Isot is ignored, as in Go
  (`bad_tile_part` in `tests/region.rs` ports `TestHandleBadTilePart`). `Image::open` also
  rejects a main-header marker id below 0xFF30, as Go's `Open` does. A tile-part header whose
  markers (not its SOT) are malformed fails only the regions that read that tile, for
  `decode_region` and for an opened `Image`: as Go's `Open` does, `Image::open` falls back, when
  the headers after a valid main header do not parse, to an Image that decodes every region as
  `decode_region` does (`bad_tile_part_header_marker`).
- The parse-time heap guard ("the headers before offset N need more memory than their size
  allows") sums the sizes of the parsed records; Rust's records are not the size of Go's, so the
  offset where it trips differs. The parity job prints it as `limit`.
- Go's `Open(io.ReaderAt, size)` becomes `Image::open(&[u8])`. Go reads only the tile-parts a
  region needs from a seekable file and keeps per-tile packet indexes between calls; here the
  caller holds the bytes, and a region call parses the packet headers of the tiles it touches
  (tiles it misses are not parsed, as in Go, so corruption there is not seen).
- Go decodes tiles on several goroutines and its `Options.Workers` caps them; here `Limits::max_threads`
  does (default `available_parallelism`, 1 = no thread; on a host with more than 2 CPUs a call
  uses at most `available_parallelism - 1` threads on the shared pool, one fewer than Go's default `GOMAXPROCS`,
  which avoids stalls when every CPU is busy). The threaded stages are the same as Go's:
  tier-1 code-blocks, the inverse DWT rows and columns, the output rows and the colour transform
  rows, and tile lanes (not with PPM, as in Go). Go gives each lane `procs / len(lanes)` workers, and so does this crate, with one difference: with fewer lanes than threads (and more than one) the lanes run on the caller and `lanes - 1` `std::thread::scope` threads rather than on pool workers (a pool worker that waits for a nested pass can run another call's lane), and the threads they use for nested passes are pool workers, so the call still uses `max_threads` threads at most. Go keeps the lanes past the first within 64 MiB and a quarter of the output (2 lanes for the 1024x1024 tiles of a 33 Mpixel image); this crate allows 512 MiB and half of the ledger that is left, because each pass of a lane's nested threads costs a pool wake and wait of tens of microseconds, which a lane of one thread never pays (the bench's `bigtiles` class). Tier 2
  runs in tile order, one tile at a time, so the tier-2 budgets (and the errors they give) do not
  depend on timing; where Go can report a different tile's error with several lanes, this crate
  reports the lowest tile's, which is what one thread reports. The tier-1 work budget is the one
  exception: it is shared by the lanes, so on a crafted multi-tile input that exhausts it, the
  tile named in the error can depend on the thread count. Output never does.
- A region decode uses Go's windowed inverse DWT (one buffer per level, each window grown by the
  filter reach, `partial_dwt`), in every rounding, so a region's bytes match Go's region. With
  JJ2000 rounding that can differ by 1 at an interior pixel from a crop of the whole decode,
  in Go and here. Tier-2 keeps only the code-blocks the window needs, so the errors match too.
- Errors are one enum. Go reports malformed input as `fmt.Errorf` text; the variants group those
  messages (`Invalid`, `Unsupported`, `LimitExceeded`, ...) and keep the text. A panic caught at
  an entry point is `Error::Internal` (Go: `ErrInternal`).
- Go's `image.RegisterFormat` init and its `cmd/` tools are not ported.
- Wavelet float math is bit-exact with Go: the default rounding uses `f32::mul_add`, which is
  correctly rounded on every target.
