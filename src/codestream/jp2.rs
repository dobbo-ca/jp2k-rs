//! JP2 box parsing (ISO/IEC 15444-1 Annex I): only enough of the box tree to find the jp2c
//! codestream and the cdef channel definitions.

use crate::error::Result;
use std::collections::HashSet;

/// The JP2 signature box (I.5.1): a file starting with it is a JP2 box file rather than a raw
/// codestream.
pub const JP2_SIGNATURE: [u8; 8] = [0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20];

pub const BOX_JP2H: u32 = 0x6a70_3268; // "jp2h"
const BOX_IHDR: u32 = 0x6968_6472; // "ihdr"
pub const BOX_JP2C: u32 = 0x6a70_3263; // "jp2c"
const BOX_PCLR: u32 = 0x7063_6c72; // "pclr"
const BOX_CMAP: u32 = 0x636d_6170; // "cmap"
const BOX_COLR: u32 = 0x636f_6c72; // "colr"
const BOX_CDEF: u32 = 0x6364_6566; // "cdef"
pub const BOX_JP: u32 = 0x6a50_2020; // "jP  " (signature box)

/// The jP signature box's required 4-byte payload (I.5.1).
pub const JP_MAGIC: [u8; 4] = [0x0d, 0x0a, 0x87, 0x0a];

/// One (Cn, Typ, Asoc) record from a cdef box (I.5.3.6): component Cn plays channel role Typ
/// (0 = colour, 1 = opacity, 2 = premultiplied opacity), associated with output channel Asoc
/// (1-based for Typ 0; ignored for Typ 1/2, which always mean "the alpha channel").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CdefEntry {
    pub cn: u16,
    pub typ: u16,
    pub asoc: u16,
}

fn parse_cdef(payload: &[u8]) -> Result<Vec<CdefEntry>> {
    if payload.len() < 2 {
        return Err(invalid!("malformed cdef box"));
    }
    let n = u16::from_be_bytes([payload[0], payload[1]]) as usize;
    if payload.len() != 2 + 6 * n {
        return Err(invalid!("malformed cdef box"));
    }
    let mut entries = Vec::with_capacity(n);
    let mut seen = HashSet::with_capacity(n);
    for i in 0..n {
        let p = &payload[2 + 6 * i..];
        let e = CdefEntry {
            cn: u16::from_be_bytes([p[0], p[1]]),
            typ: u16::from_be_bytes([p[2], p[3]]),
            asoc: u16::from_be_bytes([p[4], p[5]]),
        };
        // A repeated Cn silently drops its channel role.
        if !seen.insert(e.cn) {
            return Err(invalid!("cdef declares component {} more than once", e.cn));
        }
        entries.push(e);
    }
    Ok(entries)
}

/// Turns a cdef box's entries into a permutation mapping output channel index to source
/// component index (SIZ order), so the channel that opj_decompress calls "R" is whichever
/// component the cdef box says is Asoc 1. A cdef this decoder cannot fully resolve is an error
/// rather than a guess. `None` for no cdef.
pub fn cdef_permutation(
    cdef: Option<&[CdefEntry]>,
    num_comps: usize,
) -> Result<Option<Vec<usize>>> {
    let cdef = match cdef {
        None => return Ok(None),
        Some(c) => c,
    };
    let n_alpha = cdef.iter().filter(|e| e.typ == 1 || e.typ == 2).count();
    if n_alpha > 1 {
        return Err(unsupported!(
            "cdef declares more than one alpha channel, not supported"
        ));
    }
    if n_alpha == 1 && num_comps != 2 && num_comps != 4 {
        return Err(unsupported!(
            "cdef declares an alpha channel for {num_comps} components, not supported"
        ));
    }
    let colour_channels = num_comps - n_alpha;
    let mut perm = vec![0usize; num_comps];
    let mut filled = vec![false; num_comps];
    for e in cdef {
        if e.cn as usize >= num_comps {
            return Err(invalid!(
                "cdef component index {} out of range for {num_comps} components",
                e.cn
            ));
        }
        let pos = match e.typ {
            0 => {
                if e.asoc == 0 || e.asoc as usize > colour_channels {
                    return Err(unsupported!(
                        "cdef entry has unsupported Asoc {} for a colour channel",
                        e.asoc
                    ));
                }
                e.asoc as usize - 1
            }
            // Premultiplied opacity: NRGBA output is defined non-premultiplied, so passing it
            // through renders wrong.
            2 => {
                return Err(unsupported!(
                    "cdef premultiplied opacity (Typ 2) is not supported"
                ));
            }
            1 => num_comps - 1,
            t => return Err(unsupported!("cdef entry has unsupported Typ {t}")),
        };
        if filled[pos] {
            return Err(invalid!("cdef maps two components to output channel {pos}"));
        }
        filled[pos] = true;
        perm[pos] = e.cn as usize;
    }
    for (i, f) in filled.iter().enumerate() {
        if !f {
            return Err(invalid!("cdef does not cover output channel {i}"));
        }
    }
    Ok(Some(perm))
}

/// Streams JP2 boxes (I.4): LBox(4) TBox(4) [XLBox(8)] payload; LBox 0 runs to EOF. `visit`
/// returns whether to keep walking; an error stops the walk.
pub fn walk_boxes<'a>(
    data: &'a [u8],
    mut visit: impl FnMut(u32, &'a [u8]) -> Result<bool>,
) -> Result<()> {
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + 8 > data.len() {
            return Err(invalid!("truncated box header at offset {pos}"));
        }
        let lbox = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let tbox = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        let mut header = 8usize;
        let mut length: i64 = match lbox {
            1 => {
                if pos + 16 > data.len() {
                    return Err(invalid!("truncated XLBox at offset {pos}"));
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[pos + 8..pos + 16]);
                header = 16;
                u64::from_be_bytes(b) as i64
            }
            0 => (data.len() - pos) as i64,
            l => l as i64,
        };
        if length < header as i64 {
            return Err(invalid!(
                "box at offset {pos} has length {length} shorter than its header"
            ));
        }
        // Compare against the remaining bytes, not pos+length: an XLBox can hold values up to
        // 2^63-1.
        let remaining = (data.len() - pos) as i64;
        // A jp2c box that runs past the data is tolerated (DecodeConfig's bounded read).
        if tbox == BOX_JP2C && length > remaining {
            length = remaining;
        }
        if length > remaining {
            return Err(invalid!(
                "box at offset {pos} (length {length}) runs past end of data"
            ));
        }
        let keep_going = visit(tbox, &data[pos + header..pos + length as usize])?;
        pos += length as usize;
        if !keep_going {
            return Ok(());
        }
    }
    Ok(())
}

/// Covers any real component count.
const CDEF_PEEK_BYTES: i64 = 4096;

/// Validates jp2h children box by box. Shared across boxes; the first colr/cdef wins.
#[derive(Default)]
pub struct Jp2hChild {
    pub saw_ihdr: bool,
    pub cdef: Option<Vec<CdefEntry>>,
    saw_colr: bool,
}

impl Jp2hChild {
    /// How many bytes of a `typ` box `finish` needs.
    pub fn peek(&self, typ: u32) -> i64 {
        match typ {
            BOX_COLR => {
                if self.saw_colr {
                    0
                } else {
                    7 // check_colr only looks at METH and EnumCS
                }
            }
            BOX_CDEF => {
                if self.cdef.is_some() {
                    0
                } else {
                    CDEF_PEEK_BYTES
                }
            }
            _ => 0,
        }
    }

    /// Validates up to `peek(typ)` bytes. `truncated` marks bytes left unread.
    pub fn finish(&mut self, typ: u32, partial: &[u8], truncated: bool) -> Result<()> {
        match typ {
            BOX_IHDR => self.saw_ihdr = true,
            BOX_PCLR | BOX_CMAP => {
                // An unapplied palette would silently give wrong pixels.
                return Err(unsupported!(
                    "palette (pclr/cmap) JP2 files are not supported"
                ));
            }
            BOX_COLR => {
                // Only the first colr box is used (I.5.3.3); later ones are not validated.
                if !self.saw_colr {
                    self.saw_colr = true;
                    check_colr(partial)?;
                }
            }
            BOX_CDEF => {
                if self.cdef.is_some() {
                    return Ok(()); // a later cdef box is ignored
                }
                if truncated {
                    return Err(invalid!("cdef box too large to validate"));
                }
                self.cdef = Some(parse_cdef(partial)?);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Walks a JP2 box file's top-level boxes and returns the raw codestream payload of the jp2c
/// box and the cdef entries. jp2h must appear (with an ihdr child) before jp2c.
pub fn jp2_codestream(data: &[u8]) -> Result<(&[u8], Option<Vec<CdefEntry>>)> {
    let mut first = true;
    let mut child = Jp2hChild::default();
    let mut cs_data: Option<&[u8]> = None;
    walk_boxes(data, |typ, payload| {
        if first {
            first = false;
            if typ != BOX_JP || payload != JP_MAGIC {
                return Err(invalid!("missing or malformed jP signature box"));
            }
        }
        if typ == BOX_JP2H {
            walk_boxes(payload, |ctyp, cpayload| {
                child.finish(ctyp, cpayload, false)?;
                Ok(true)
            })?;
        }
        if typ == BOX_JP2C {
            if !child.saw_ihdr {
                return Err(invalid!(
                    "jp2c box appears before a jp2h box with an ihdr child"
                ));
            }
            cs_data = Some(payload);
            return Ok(false); // what follows is not read
        }
        Ok(true)
    })?;
    if first {
        return Err(invalid!("missing or malformed jP signature box"));
    }
    match cs_data {
        None => Err(invalid!("no jp2c box found")),
        Some(cs) => Ok((cs, child.cdef)),
    }
}

/// Rejects a colr box this decoder cannot render faithfully: there is no colour management
/// step, so an ICC profile or any enumerated colourspace other than sRGB (16) or greyscale (17)
/// is an error rather than wrong-but-plausible colours.
pub fn check_colr(payload: &[u8]) -> Result<()> {
    if payload.len() < 3 {
        return Err(invalid!("malformed colr box"));
    }
    if payload[0] != 1 {
        return Err(unsupported!(
            "colr specifies an ICC profile colourspace, not supported"
        ));
    }
    if payload.len() < 7 {
        return Err(invalid!("malformed colr box"));
    }
    let enumcs = u32::from_be_bytes([payload[3], payload[4], payload[5], payload[6]]);
    if enumcs != 16 && enumcs != 17 {
        return Err(unsupported!(
            "colr enumerated colourspace {enumcs} not supported (only sRGB=16, greyscale=17)"
        ));
    }
    Ok(())
}
