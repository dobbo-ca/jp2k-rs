//! JPEG 2000 decoder: raw J2K codestreams and JP2 files, decode only. A Rust port of
//! `dobbo-ca/jp2k-go`.
//!
//! Entry points take a byte slice (or a reader), an [`Options`] holding the [`Limits`], and a
//! cancel flag. See the README for what differs from Go.
//!
//! # Quick start
//!
//! The first call passes explicit [`Limits`]: at most 2 threads and 64 MiB of working memory.
//!
//! ```
//! use jp2k::{Limits, Options};
//! use std::sync::atomic::AtomicBool;
//!
//! let data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/g64_53.j2k"));
//! let opts = Options {
//!     limits: Limits { max_threads: 2, max_alloc: 64 << 20, ..Limits::default() },
//!     ..Options::default()
//! };
//! let cancel = AtomicBool::new(false);
//! let img = jp2k::decode(data, &opts, &cancel)?;
//! assert_eq!((img.bounds.x1, img.bounds.y1), (64, 64));
//! # Ok::<(), jp2k::Error>(())
//! ```
//!
//! # Resource limits
//!
//! Every call is bounded by [`Limits`], in [`Options::limits`] (or passed directly to
//! [`decode_config`] and [`decode_config_reader`]). There is no unlimited entry point.
//!
//! | field | default | unit | over it |
//! |---|---|---|---|
//! | `max_alloc` | 4 GiB | bytes | `Error::LimitExceeded` |
//! | `max_threads` | `std::thread::available_parallelism()` (1 if unknown) | threads | none: the call uses fewer |
//! | `max_pixels` | 2^28 | pixels of the result | `Error::LimitExceeded` |
//! | `max_width`, `max_height` | `u32::MAX` | pixels | `Error::LimitExceeded` |
//! | `max_input` | 4 GiB | bytes a reader entry point reads | `Error::LimitExceeded` |
//! | `max_output` | 2 GiB | bytes of the result | `Error::LimitExceeded` |
//!
//! - `max_alloc` is the PEAK working memory of one call, all its threads together: every
//!   allocation that scales with the input is charged to one ledger before it is made and
//!   given back when it is dropped. It is not an estimate. The input slice is the caller's and
//!   is not charged; a reader's bytes are.
//! - `max_threads` is the most OS threads one call uses, the calling thread included. 1 runs on
//!   the calling thread and never spawns one; 0 counts as 1. The threads are split over tiles,
//!   code-blocks, wavelet rows and columns and output rows, never more than this in total, and
//!   the output is the same bytes for every value.
//! - Threads come from a pool, so small images use them too. [`Options::pool`] (a
//!   [`ThreadPool`], made once with `ThreadPool::new(n)` and kept) always wins. Without one, a
//!   call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock` built lazily
//!   on the first such call, with `std::thread::available_parallelism()` threads and never more,
//!   and holding no other state. `max_threads` 1 never touches it (it is not even built). Each
//!   call takes at most `max_threads` threads from the pool, the calling thread included, and,
//!   when the pool has more than 2 threads, at most one fewer than its size (a pool of 1 or 2
//!   threads gives them all, so the default on a 2-CPU host still uses 2). If the shared pool cannot be built, the call
//!   starts one set of workers of its own at entry, sized from the input (none for a small
//!   one), which every stage shares. The pool changes speed only, not the bytes.
//! - Cancellation: set the `AtomicBool` passed to the call, from any thread; every thread of
//!   the call stops at its next tile, code-block or row and the call returns
//!   `Error::Cancelled`.
//! - A panic on any thread is caught and returned as `Error::Internal`.
//! - Other errors: `Error::Invalid` (malformed input), `Error::Unsupported` (valid input this
//!   decoder does not handle), `Error::OutOfMemory` (an allocation failed), `Error::Io` (a reader
//!   failed).
//!
//! A caller in a 2-CPU, 1 GiB pod can set `max_threads: 2, max_alloc: 768 << 20` and the call
//! never goes above either. A constrained call, and what a limit looks like when it is hit:
//!
//! ```
//! use jp2k::{Error, Limits, Options};
//! use std::sync::atomic::{AtomicBool, Ordering};
//!
//! let data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/g64_53.j2k"));
//! let cancel = AtomicBool::new(false);
//! let one_thread = Options {
//!     limits: Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() },
//!     ..Options::default()
//! };
//! // No thread is spawned, and the bytes are those any other `max_threads` gives.
//! let img = jp2k::decode(data, &one_thread, &cancel)?;
//! let more = Options { limits: Limits { max_threads: 4, ..one_thread.limits }, ..one_thread.clone() };
//! assert!(img == jp2k::decode(data, &more, &cancel)?);
//!
//! // Too little memory: `Error::LimitExceeded`, before anything is allocated.
//! let tiny = Options {
//!     limits: Limits { max_alloc: 1000, ..one_thread.limits },
//!     ..one_thread.clone()
//! };
//! assert!(matches!(jp2k::decode(data, &tiny, &cancel), Err(Error::LimitExceeded(_))));
//!
//! // A cancel flag that is set: `Error::Cancelled`.
//! cancel.store(true, Ordering::Relaxed);
//! assert_eq!(jp2k::decode(data, &one_thread, &cancel).unwrap_err(), Error::Cancelled);
//! # Ok::<(), Error>(())
//! ```
#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]
// The port keeps Go's index-based numeric loops and long argument lists where they follow the
// reference; items kept for the ported tests are not used by the library itself.
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]

/// The README's code examples, compiled as doc-tests so they cannot go stale.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[macro_use]
mod error;
mod codestream;
mod decoder;
mod dsp;
#[doc(hidden)]
pub mod fuzz;
mod limits;
mod parallel;
#[doc(hidden)]
pub mod stats;
mod tier1;
mod tier2;

pub use codestream::Rect;
pub use decoder::{
    Decoder, Image, decode, decode_components, decode_config, decode_config_reader, decode_reader,
    decode_region, decode_region_components,
};
pub use error::{Error, Result};
#[doc(hidden)]
pub use limits::Budget;
pub use limits::Limits;
pub use parallel::ThreadPool;
#[doc(hidden)]
pub use parallel::shared_pool_started;

/// How a decode computes its samples.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Rounds the multiply and the add of each 9/7 lifting step and of the ICT separately,
    /// matching OpenJPEG built for x86-64 without FMA contraction (for example Ubuntu's
    /// libopenjp2, which libvips uses). The default rounds each multiply-add once, matching
    /// opj_decompress where its compiler contracts them (arm64). Both give the same bytes on
    /// every architecture. Reversible (5/3) decodes are the same either way.
    pub separate_rounding: bool,
    /// Reproduces the 9/7 decode of JJ2000 (jai-imageio), which Bio-Formats uses: the subband
    /// gain stays in the dequantization step, the bands are divided by 1/K and K in JJ2000's
    /// lifting order with no FMA, and samples round half away from zero before the level shift.
    /// It implies `separate_rounding`. An ICT still uses `separate_rounding`'s.
    pub jj2000: bool,
    /// Caps on one call.
    pub limits: Limits,
    /// Worker threads to share between calls (see [`ThreadPool`]). With one, a call runs its
    /// parallel passes on the pool's workers and the calling thread, no more than
    /// `limits.max_threads` of them, and starts no thread of its own. Without one, a call with
    /// `max_threads` above 1 takes the same from the crate's shared pool: built on first use
    /// (a `OnceLock`), never more than `available_parallelism` threads, holding no other state.
    /// A pooled call, the shared pool included, uses at most `max_threads` threads and, when the
    /// pool has more than 2 threads, at most the pool's size minus 1.
    /// If that cannot be built, the call starts one pool of its own at entry, sized from the
    /// input, for all its stages. `max_threads` 1 never touches any pool, and never builds the
    /// shared one. The bytes are the same either way.
    pub pool: Option<ThreadPool>,
    /// For tests: skip the shared pool and build the call's own workers (see `pool`).
    #[doc(hidden)]
    pub per_call_pool: bool,
}

impl Options {
    pub(crate) fn rounding(&self) -> dsp::Rounding {
        if self.jj2000 {
            dsp::Rounding::JJ2000
        } else if self.separate_rounding {
            dsp::Rounding::Separate
        } else {
            dsp::Rounding::Fused
        }
    }
}

/// The pixel format of a decoded [`Raster`], by component count and precision:
///
/// | components | 8-bit | 9-16 bit |
/// |---|---|---|
/// | 1 | `Gray` | `Gray16` |
/// | 2 | `Nrgba` (gray+alpha, R=G=B=Y) | `Nrgba64` |
/// | 3 | `Rgba` (opaque) | `Rgba64` |
/// | 4 | `Nrgba` | `Nrgba64` |
///
/// 16-bit channels are big-endian, as Go's `image` package stores them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorModel {
    Gray,
    Gray16,
    Rgba,
    Rgba64,
    Nrgba,
    Nrgba64,
}

impl ColorModel {
    /// Bytes per pixel in [`Raster::pix`]. Enforces no limit and cannot fail.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            ColorModel::Gray => 1,
            ColorModel::Gray16 => 2,
            ColorModel::Rgba | ColorModel::Nrgba => 4,
            ColorModel::Rgba64 | ColorModel::Nrgba64 => 8,
        }
    }
}

/// The size and pixel format of an image, from [`decode_config`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The pixel format a decode of this image returns.
    pub color_model: ColorModel,
    /// Width of the decoded image in pixels.
    pub width: u32,
    /// Height of the decoded image in pixels.
    pub height: u32,
}

/// A decoded image: `bounds` in (reduced) image coordinates, `stride` bytes per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raster {
    /// How `pix` is laid out; see [`ColorModel`].
    pub color_model: ColorModel,
    /// The region this raster covers, in (reduced) image coordinates.
    pub bounds: Rect,
    /// Bytes from the start of one row to the start of the next.
    pub stride: usize,
    /// The samples, `stride` bytes per row, 16-bit channels big-endian.
    pub pix: Vec<u8>,
}

/// One component's decoded samples, not scaled to 16 bits: an unsigned component of precision p
/// gives 0 to 2^p-1, and a signed one its value plus 2^(p-1). Row-major, `bounds` wide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    /// The region this plane covers.
    pub bounds: Rect,
    /// The samples, row-major, `bounds` wide.
    pub pix: Vec<u16>,
}
