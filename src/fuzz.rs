//! Hooks for the fuzz targets, the integration tests and `examples/jp2kinfo.rs`. Not part of the
//! API.
//!
//! The decoder's stages (`codestream`, `tier1`, `tier2`, `dsp`) are crate-private: their kernels
//! trust sizes that the decoder has already checked. Everything reachable from here either takes
//! `&Limits` (or a [`Budget`] built from one) and a cancel flag where it loops over the input and
//! validates its own arguments, or only reads a structure a limit-checked parse produced.

use crate::codestream::{self, Parse};
use crate::error::Result;
use crate::limits::{Budget, Limits};
use crate::tier1;
use std::sync::atomic::AtomicBool;

pub use crate::codestream::jp2::{CdefEntry, JP2_SIGNATURE};
pub use crate::codestream::{
    Coc, Cod, Codestream, CodingStyle, Com, Plt, Poc, Qcc, Qcd, Quantization, Rgn, Siz, Tlm,
    effective_coding_style, effective_mct, quant_for, tile_overrides,
};
pub use crate::dsp::Rounding;
// The SIMD tier kernels, checked against the scalar one. Each checks its slice lengths.
pub use crate::dsp::fuzz as simd;
pub use crate::tier1::{Segment, State};
pub use crate::tier2::{
    CodeBlockRef, CodeBlockSegment, Cursor, PacketRef, TileGeom, TileWant, assemble_tile,
    build_tile_geom, decode_tile, passes_read,
};

fn check_input(len: usize, limits: &Limits) -> Result<()> {
    if len > limits.max_input {
        return Err(limit!(
            "input of {len} bytes exceeds the limit of {} bytes",
            limits.max_input
        ));
    }
    Ok(())
}

/// Parses a raw codestream, keeping every marker.
///
/// # Errors
/// Invalid or truncated input, `Error::LimitExceeded` over `limits`, `Error::Cancelled`.
pub fn parse(data: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Codestream> {
    parse_codestream(data, true, limits, cancel)
}

/// Parses a raw codestream; with `keep` false, COM, PLT and repeated markers are dropped.
///
/// # Errors
/// As [`parse`].
pub fn parse_codestream(
    data: &[u8],
    keep: bool,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Codestream> {
    check_input(data.len(), limits)?;
    codestream::parse_tiles(data, keep, limits, Parse::Whole, cancel)
}

/// The jp2c payload and cdef entries of a JP2 file.
///
/// # Errors
/// Malformed boxes, or input over `limits.max_input`.
pub fn jp2_codestream<'a>(
    data: &'a [u8],
    limits: &Limits,
) -> Result<(&'a [u8], Option<Vec<CdefEntry>>)> {
    check_input(data.len(), limits)?;
    codestream::jp2::jp2_codestream(data)
}

/// Parses the SIZ segment whose marker id starts at `pos`.
///
/// # Errors
/// Malformed or truncated segment, or input over `limits.max_input`.
pub fn parse_siz_segment(data: &[u8], pos: usize, limits: &Limits) -> Result<(Siz, usize)> {
    check_input(data.len(), limits)?;
    codestream::parse_siz_segment(data, pos)
}

/// Parses a COD segment payload.
///
/// # Errors
/// Malformed segment, or input over `limits.max_input`.
pub fn parse_cod(payload: &[u8], marker_offset: usize, limits: &Limits) -> Result<Cod> {
    check_input(payload.len(), limits)?;
    codestream::parse_cod(payload, marker_offset)
}

/// Parses a PLT segment payload.
///
/// # Errors
/// Malformed segment, or input over `limits.max_input`.
pub fn parse_plt(payload: &[u8], marker_offset: usize, keep: bool, limits: &Limits) -> Result<Plt> {
    check_input(payload.len(), limits)?;
    codestream::parse_plt(payload, marker_offset, keep)
}

/// EBCOT-decodes one code-block into sign-magnitude coefficients and per-sample undecoded
/// bit-plane counts.
///
/// # Errors
/// A size over T.800's 4096 samples, `Error::LimitExceeded` if the two output vectors do not fit
/// `limits.max_alloc`, or a malformed segment.
pub fn decode_code_block(
    segs: &[Segment],
    width: i64,
    height: i64,
    orientation: usize,
    zero_bit_planes: i32,
    mb: i32,
    style: u8,
    limits: &Limits,
) -> Result<(Vec<i32>, Vec<i32>)> {
    tier1::validate_code_block_dims(width, height, style)?;
    // Validated: 1 <= width * height <= 4096.
    let n = (width as usize) * (height as usize);
    Budget::new(limits).charge(2 * n * std::mem::size_of::<i32>())?;
    tier1::decode_code_block(segs, width, height, orientation, zero_bit_planes, mb, style)
}

/// Runs the MQ decoder over 10000 decisions on contexts `i % 19`, and the raw decoder over 10000
/// bits, of `data`; returns a hash of them.
///
/// # Errors
/// Input over `limits.max_input`.
pub fn mq_decisions(data: &[u8], limits: &Limits) -> Result<u32> {
    check_input(data.len(), limits)?;
    Ok(tier1::mq_decisions(data))
}
