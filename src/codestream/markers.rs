use crate::error::{Error, Result};
use crate::limits::{Limits, check_cancel};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

// Marker codes, ISO/IEC 15444-1 Annex A.
pub const MARKER_SOC: u16 = 0xFF4F;
pub const MARKER_SIZ: u16 = 0xFF51;
pub const MARKER_COD: u16 = 0xFF52;
pub const MARKER_COC: u16 = 0xFF53;
pub const MARKER_TLM: u16 = 0xFF55;
pub const MARKER_PLT: u16 = 0xFF58;
pub const MARKER_QCD: u16 = 0xFF5C;
pub const MARKER_QCC: u16 = 0xFF5D;
pub const MARKER_RGN: u16 = 0xFF5E;
pub const MARKER_POC: u16 = 0xFF5F;
pub const MARKER_COM: u16 = 0xFF64;
pub const MARKER_PPM: u16 = 0xFF60;
pub const MARKER_PPT: u16 = 0xFF61;
pub const MARKER_SOT: u16 = 0xFF90;
pub const MARKER_SOP: u16 = 0xFF91;
pub const MARKER_EPH: u16 = 0xFF92;
pub const MARKER_SOD: u16 = 0xFF93;
pub const MARKER_EOC: u16 = 0xFFD9;

fn marker_name(id: u16) -> String {
    match id {
        MARKER_SOC => "SOC".into(),
        MARKER_SIZ => "SIZ".into(),
        MARKER_COD => "COD".into(),
        MARKER_COC => "COC".into(),
        MARKER_TLM => "TLM".into(),
        MARKER_PLT => "PLT".into(),
        MARKER_QCD => "QCD".into(),
        MARKER_QCC => "QCC".into(),
        MARKER_RGN => "RGN".into(),
        MARKER_POC => "POC".into(),
        MARKER_COM => "COM".into(),
        MARKER_PPM => "PPM".into(),
        MARKER_PPT => "PPT".into(),
        MARKER_SOT => "SOT".into(),
        MARKER_SOP => "SOP".into(),
        MARKER_EPH => "EPH".into(),
        MARKER_SOD => "SOD".into(),
        MARKER_EOC => "EOC".into(),
        _ => format!("0x{id:04X}"),
    }
}

/// Big-endian u16 at the start of `b`; `b` must hold 2 bytes.
pub fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

/// Big-endian u32 at the start of `b`; `b` must hold 4 bytes.
pub fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// One entry of SIZ's per-component array.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SizComponent {
    /// Bit depth, 1..=31 (SIZ parsing caps it at 31).
    pub precision: u8,
    pub signed: bool,
    pub xrsiz: u8,
    pub yrsiz: u8,
}

/// The decoded Image and tile size marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Siz {
    pub rsiz: u16,
    pub xsiz: u32,
    pub ysiz: u32,
    pub xosiz: u32,
    pub yosiz: u32,
    pub xtsiz: u32,
    pub ytsiz: u32,
    pub xtosiz: u32,
    pub ytosiz: u32,
    pub csiz: u16,
    pub components: Vec<SizComponent>,
    /// Derived: tiles across and down.
    pub tiles_wide: u32,
    pub tiles_high: u32,
}

/// One entry of a COD/COC custom precinct-size list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrecinctSize {
    /// Precinct width exponent: the width is `1 << ppx`.
    pub ppx: u8,
    pub ppy: u8,
}

/// The SPcod/SPcoc body shared by COD and COC.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodingStyle {
    /// Raw value; resolutions = this + 1.
    pub num_decomp_levels: u8,
    /// log2 of the code-block width, i.e. `(raw & 0xF) + 2`.
    pub code_block_width_exp: u8,
    pub code_block_height_exp: u8,
    pub code_block_style: u8,
    /// 0 = 9/7 irreversible, 1 = 5/3 reversible.
    pub transformation: u8,
    /// `None` unless custom precincts are present.
    pub precinct_sizes: Option<Vec<PrecinctSize>>,
}

/// The decoded Coding style default marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cod {
    pub scod: u8,
    pub uses_custom_precincts: bool,
    pub uses_sop: bool,
    pub uses_eph: bool,
    pub progression_order: u8,
    pub number_of_layers: u16,
    pub multiple_component_transform: u8,
    pub coding_style: CodingStyle,
}

/// The decoded Coding style component marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coc {
    pub component_index: u16,
    pub scoc: u8,
    pub uses_custom_precincts: bool,
    pub coding_style: CodingStyle,
}

/// The SQcd/SQcc body shared by QCD and QCC.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    /// 0 none, 1 scalar derived, 2 scalar expounded.
    pub quantization_style: u8,
    pub guard_bits: u8,
    pub exponents: Vec<u8>,
    /// 0 for the no-quantization style.
    pub mantissas: Vec<u16>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qcd {
    pub sqcd: u8,
    pub quantization: Quantization,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qcc {
    pub component_index: u16,
    pub sqcc: u8,
    pub quantization: Quantization,
}

/// The decoded Region-of-interest marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rgn {
    pub component_index: u16,
    pub srgn: u8,
    pub sprgn: u8,
}

/// One progression-change entry inside a POC marker segment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PocChange {
    pub rspoc: u8,
    pub cspoc: u16,
    pub lyepoc: u16,
    pub repoc: u8,
    pub cepoc: u16,
    pub ppoc: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poc {
    pub changes: Vec<PocChange>,
}

/// The decoded Comment marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Com {
    /// 0 = binary, 1 = ISO 8859-15 text.
    pub rcom: u16,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TlmEntry {
    pub ttlm: u32,
    pub ptlm: u32,
}

/// The decoded Tile-part lengths marker segment. Any internal inconsistency invalidates the
/// index rather than failing the parse: TLM is an optional hint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tlm {
    pub ztlm: u8,
    pub stlm: u8,
    pub entries: Vec<TlmEntry>,
    pub invalid: bool,
}

/// The decoded Packet length, tile-part header marker segment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plt {
    pub zplt: u8,
    pub packet_lengths: Vec<u64>,
}

/// The decoded Start of tile-part marker segment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sot {
    pub isot: u16,
    pub psot: u32,
    pub tpsot: u8,
    pub tnsot: u8,
}

/// One PPM marker's payload, keyed by Zppm (T.800 A.7.4): PPM markers can arrive out of Z order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PpmEntry {
    pub z: u8,
    pub data: Vec<u8>,
}

/// One PPT marker's payload, keyed by Zppt (T.800 A.7.5).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PptEntry {
    pub z: u8,
    pub data: Vec<u8>,
}

/// The markers of one tile-part header. Most tile-parts have none, so `TilePart` boxes them: a
/// flood of bare tile-parts then costs 32 bytes each, not the size of every marker list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TpMarkers {
    pub cod: Option<Cod>,
    pub coc: Vec<Coc>,
    pub qcd: Option<Qcd>,
    pub qcc: Vec<Qcc>,
    pub rgn: Vec<Rgn>,
    pub poc: Vec<Poc>,
    pub com: Vec<Com>,
    pub plt: Vec<Plt>,
    pub ppt: Vec<PptEntry>,
}

/// One tile-part's header markers plus the location of its (unparsed) packet data within the
/// codestream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TilePart {
    pub sot: Sot,
    pub markers: Option<Box<TpMarkers>>,
    /// Offset of the first data byte after SOD.
    pub data_offset: usize,
    pub data_length: usize,
}

impl TilePart {
    pub fn cod(&self) -> Option<&Cod> {
        self.markers.as_deref().and_then(|m| m.cod.as_ref())
    }

    pub fn coc(&self) -> &[Coc] {
        self.markers.as_deref().map_or(&[], |m| &m.coc)
    }

    pub fn qcd(&self) -> Option<&Qcd> {
        self.markers.as_deref().and_then(|m| m.qcd.as_ref())
    }

    pub fn qcc(&self) -> &[Qcc] {
        self.markers.as_deref().map_or(&[], |m| &m.qcc)
    }

    pub fn rgn(&self) -> &[Rgn] {
        self.markers.as_deref().map_or(&[], |m| &m.rgn)
    }

    pub fn poc(&self) -> &[Poc] {
        self.markers.as_deref().map_or(&[], |m| &m.poc)
    }

    pub fn com(&self) -> &[Com] {
        self.markers.as_deref().map_or(&[], |m| &m.com)
    }

    pub fn plt(&self) -> &[Plt] {
        self.markers.as_deref().map_or(&[], |m| &m.plt)
    }

    pub fn ppt(&self) -> &[PptEntry] {
        self.markers.as_deref().map_or(&[], |m| &m.ppt)
    }
}

/// The result of parsing a full codestream's marker segments (main header plus every tile-part
/// header).
#[derive(Debug)]
pub struct Codestream {
    pub siz: Siz,
    pub cod: Cod,
    pub coc: Vec<Coc>,
    pub qcd: Qcd,
    pub qcc: Vec<Qcc>,
    pub rgn: Vec<Rgn>,
    pub poc: Vec<Poc>,
    pub com: Vec<Com>,
    pub tlm: Vec<Tlm>,
    pub ppm: Vec<PpmEntry>,
    pub tile_parts: Vec<TilePart>,
    pub eoc: bool,
    /// Set when the stream ends mid tile-part with no EOC (OpenJPEG's NEOC state); the parsed
    /// prefix is still returned.
    pub truncated: bool,
    pub warnings: Vec<String>,
    /// Roughly the heap the parse held.
    pub parse_cost: i64,
    ppm_merged: OnceLock<Result<Vec<u8>>>,
    ppt_merged: OnceLock<Result<HashMap<u16, Vec<u8>>>>,
    tp_start: Vec<u32>,
    tp_idx: Vec<u32>,
}

/// How `parse_tiles` treats a malformed tile-part SOT.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Parse {
    /// Any malformed SOT fails the parse.
    Whole,
    /// Like `Whole`, and a header marker id below 0xFF30 fails the parse, as in Go's `Open`.
    Open,
}

/// Caps `Codestream::warnings`: a stream of 4-byte unknown markers would otherwise cost ~30 bytes
/// of warnings per input byte.
const MAX_WARNINGS: usize = 100;

/// Caps the heap a decode's parse holds at any point to `PARSE_BYTES_PER` per input byte up to
/// it, plus `PARSE_SLACK`.
const PARSE_BYTES_PER: i64 = 8;
const PARSE_SLACK: i64 = 1 << 20;

/// Header markers walked between two cancel checks.
const CANCEL_EVERY: usize = 4096; // a power of two

struct ParseState<'a> {
    cancel: &'a AtomicBool,
    /// Markers walked, so a header of millions of them checks `cancel` every `CANCEL_EVERY`.
    markers: usize,
    warnings: Vec<String>,
    dropped: usize,
    /// Keep what decoding never reads or only reads the first of: PLT, COM, a repeated COC,
    /// QCC or RGN in one header. Without it, the parse is a decode's and its heap is capped.
    keep: bool,
    cost: i64,
    charged: i64,
    max_alloc: i64,
    /// `seen[k * csiz + c] == hdr` when this header already has marker kind `k` for component
    /// `c`; `hdr` counts headers.
    seen: Vec<u32>,
    hdr: u32,
    /// Marker ids below this are an error in a header.
    min_marker: u16,
    /// Quantization vectors and COD precinct-size lists to parse into (a `Decoder`'s, from its
    /// last parse).
    quants: Vec<(Vec<u8>, Vec<u16>)>,
    precincts: Vec<Vec<PrecinctSize>>,
}

/// The vectors a parse fills that a `Decoder` keeps from one parse to the next: `reclaim` takes
/// them back from a codestream it is done with, and `parse_tiles_in` fills them again.
#[derive(Default)]
pub(crate) struct ParseBufs {
    components: Vec<SizComponent>,
    quants: Vec<(Vec<u8>, Vec<u16>)>,
    precincts: Vec<Vec<PrecinctSize>>,
    parts: Vec<TilePart>,
    tp_next: HashMap<u16, usize>,
    seen: Vec<u32>,
    tp_start: Vec<u32>,
    tp_idx: Vec<u32>,
}

impl ParseBufs {
    /// Keeps the vectors of `cs` for the next parse.
    pub(crate) fn reclaim(&mut self, cs: Codestream) {
        let Codestream {
            siz,
            cod,
            qcd,
            qcc,
            mut tile_parts,
            tp_start,
            tp_idx,
            ..
        } = cs;
        self.components = siz.components;
        self.precincts.extend(cod.coding_style.precinct_sizes);
        // Last first: the next parse pops QCD's first, then the QCCs' in order.
        for q in qcc
            .into_iter()
            .rev()
            .map(|q| q.quantization)
            .chain(std::iter::once(qcd.quantization))
        {
            self.quants.push((q.exponents, q.mantissas));
        }
        tile_parts.clear();
        self.parts = tile_parts;
        self.tp_start = tp_start;
        self.tp_idx = tp_idx;
    }
}

impl ParseState<'_> {
    fn spend(&mut self, n: i64, pos: usize) -> Result<()> {
        self.account(n, n, pos)
    }

    fn account(&mut self, n: i64, charge: i64, pos: usize) -> Result<()> {
        self.cost += n;
        self.charged += charge;
        // A keeping parse holds what it likes per input byte, but never past `max_alloc`.
        if !self.keep && self.charged > PARSE_BYTES_PER * pos as i64 + PARSE_SLACK {
            return Err(limit!(
                "the headers before offset {pos} need more memory than their size allows"
            ));
        }
        if 2 * self.cost > self.max_alloc {
            return Err(limit!(
                "the headers need more than the {} bytes of working memory the limit allows",
                self.max_alloc
            ));
        }
        Ok(())
    }

    /// Whether this header already had marker kind `k` (0 COC, 1 QCC, 2 RGN) for component `c`,
    /// when the parse drops repeats.
    fn repeat(&mut self, k: usize, c: u16, csiz: u16, pos: usize) -> Result<bool> {
        if self.keep {
            return Ok(false);
        }
        if self.seen.is_empty() {
            let n = 3 * csiz as usize;
            self.spend(n as i64 * 4, pos)?;
            self.seen.resize(n, 0);
        }
        let i = k * csiz as usize + c as usize;
        if self.seen[i] == self.hdr {
            return Ok(true);
        }
        self.seen[i] = self.hdr;
        Ok(false)
    }

    fn warn(&mut self, msg: impl FnOnce() -> String) {
        if self.warnings.len() >= MAX_WARNINGS {
            self.dropped += 1;
            return;
        }
        self.warnings.push(msg());
    }
}

/// Parses a raw JPEG 2000 codestream (no JP2 box wrapper): SOC, then the main header up to the
/// first SOT, then every tile-part (header markers decoded, packet data located but not decoded)
/// up to EOC. Everything is kept, including COM, PLT and repeated markers.
pub fn parse(data: &[u8], limits: &Limits) -> Result<Codestream> {
    parse_codestream(data, true, limits)
}

pub fn parse_codestream(data: &[u8], keep: bool, limits: &Limits) -> Result<Codestream> {
    parse_tiles(data, keep, limits, Parse::Whole, &AtomicBool::new(false))
}

pub(crate) fn parse_tiles(
    data: &[u8],
    keep: bool,
    limits: &Limits,
    mode: Parse,
    cancel: &AtomicBool,
) -> Result<Codestream> {
    parse_tiles_in(data, keep, limits, mode, cancel, &mut ParseBufs::default())
}

/// `parse_tiles`, filling the vectors `bufs` holds (a `Decoder`'s, from its last parse).
pub(crate) fn parse_tiles_in(
    data: &[u8],
    keep: bool,
    limits: &Limits,
    mode: Parse,
    cancel: &AtomicBool,
    bufs: &mut ParseBufs,
) -> Result<Codestream> {
    let mut seen = std::mem::take(&mut bufs.seen);
    seen.clear();
    let mut ps = ParseState {
        cancel,
        markers: 0,
        warnings: Vec::new(),
        dropped: 0,
        keep,
        cost: 0,
        charged: 0,
        max_alloc: limits.max_alloc.min(i64::MAX as usize) as i64,
        seen,
        hdr: 0,
        min_marker: if mode == Parse::Open { 0xFF30 } else { 0xFF00 },
        quants: std::mem::take(&mut bufs.quants),
        precincts: std::mem::take(&mut bufs.precincts),
    };
    if data.len() < 2 || be16(&data[0..2]) != MARKER_SOC {
        return Err(invalid!("missing SOC marker at offset 0"));
    }
    let mut pos = 2usize;
    if pos + 2 > data.len() || be16(&data[pos..]) != MARKER_SIZ {
        return Err(invalid!(
            "expected SIZ marker immediately after SOC at offset {pos}"
        ));
    }
    let (siz, next) = parse_siz_in(data, pos, std::mem::take(&mut bufs.components))?;
    pos = next;

    let (hm, next) = parse_markers(
        data,
        pos,
        MARKER_SOT,
        siz.csiz,
        true,
        false,
        siz.tiles_wide,
        siz.tiles_high,
        &mut ps,
    )?;
    pos = next;
    let cod = hm
        .cod
        .ok_or_else(|| invalid!("missing mandatory COD marker in main header"))?;
    let qcd = hm
        .qcd
        .ok_or_else(|| invalid!("missing mandatory QCD marker in main header"))?;

    // pos now points at the first SOT. Walk tile-parts until EOC. `tp_next` tracks, per tile,
    // the TPsot expected on its next tile-part (an int, so a 257th tile-part cannot wrap to 0).
    let mut tp_next = std::mem::take(&mut bufs.tp_next);
    tp_next.clear();
    // Zppt dedup across every tile-part of the SAME tile: PPT is scoped per tile (T.800 A.7.5),
    // but parse_markers sees one tile-part header at a time.
    let mut ppt_seen_z: HashMap<u16, HashSet<u8>> = HashMap::new();
    let rec_size = size_of::<TilePart>() as i64 + 4;
    let (n, end) = count_tile_parts(data, pos);
    ps.account(n as i64 * rec_size, 0, end)?;
    let mut parts = std::mem::take(&mut bufs.parts);
    parts.clear();
    parts.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    // The room `parts` is charged for, growing as a new vector's would (a kept one may have more).
    let mut cap = n;
    let mut truncated_stream = false;
    let mut eoc = false;
    while pos < data.len() {
        check_cancel(cancel)?;
        if pos + 2 > data.len() {
            return Err(invalid!(
                "truncated codestream at offset {pos}, expected SOT or EOC"
            ));
        }
        let id = be16(&data[pos..]);
        if id == MARKER_EOC {
            eoc = true;
            break;
        }
        if id != MARKER_SOT {
            return Err(invalid!(
                "unexpected marker {} at offset {pos}, expected SOT or EOC",
                marker_name(id)
            ));
        }

        let (tp, next, truncated) = parse_tile_part(data, pos, siz.csiz, &mut ps)?;
        if tp.sot.isot as u32 >= siz.tiles_wide * siz.tiles_high {
            return Err(invalid!(
                "SOT Isot {} out of range at offset {pos} (tile grid is {}x{})",
                tp.sot.isot,
                siz.tiles_wide,
                siz.tiles_high
            ));
        }
        let want_tpsot = tp_next.get(&tp.sot.isot).copied();
        let mut cost = 0i64;
        if want_tpsot.is_none() {
            cost += 48; // a map entry
        }
        if parts.len() == cap {
            cost += 2 * rec_size;
            cap = (2 * cap).max(4);
        }
        if tp.markers.is_some() {
            cost += size_of::<TpMarkers>() as i64;
        }
        let want_tpsot = want_tpsot.unwrap_or(0);
        let order = if want_tpsot > 255 {
            Err(invalid!(
                "tile {} has more than 256 tile-parts at offset {pos}",
                tp.sot.isot
            ))
        } else if tp.sot.tpsot as usize != want_tpsot {
            Err(invalid!(
                "SOT Isot {} TPsot {} out of order at offset {pos} (expected {want_tpsot})",
                tp.sot.isot,
                tp.sot.tpsot
            ))
        } else if tp.sot.tnsot != 0 && tp.sot.tpsot >= tp.sot.tnsot {
            Err(invalid!(
                "SOT Isot {} TPsot {} >= TNsot {} at offset {pos}",
                tp.sot.isot,
                tp.sot.tpsot,
                tp.sot.tnsot
            ))
        } else {
            Ok(())
        };
        order?;
        tp_next.insert(tp.sot.isot, want_tpsot + 1);
        if !tp.ppt().is_empty() && !ppt_seen_z.contains_key(&tp.sot.isot) {
            ppt_seen_z.insert(tp.sot.isot, HashSet::new());
            cost += 256;
        }
        ps.account(cost, cost + rec_size, next)?;
        let mut ppt_dup = None;
        for e in tp.ppt() {
            let seen = ppt_seen_z.entry(tp.sot.isot).or_default();
            if !seen.insert(e.z) {
                ppt_dup = Some(invalid!(
                    "tile {} PPT Zppt {} already read at offset {pos}",
                    tp.sot.isot,
                    e.z
                ));
                break;
            }
        }
        if let Some(e) = ppt_dup {
            return Err(e);
        }
        parts.push(tp);
        pos = next;
        if truncated {
            // No EOC follows; tolerate per the reference NEOC state and return the
            // already-parsed prefix instead of failing outright.
            truncated_stream = true;
            break;
        }
    }

    if !hm.ppm.is_empty() {
        // T.800 A.7.5: PPT is illegal anywhere in a codestream that also carries PPM.
        for tp in &parts {
            if !tp.ppt().is_empty() {
                return Err(invalid!(
                    "tile {} has PPT markers in a codestream that also uses PPM, which T.800 forbids",
                    tp.sot.isot
                ));
            }
        }
    }

    let mut warnings = ps.warnings;
    if ps.dropped > 0 {
        warnings.push(format!("{} more warnings not kept", ps.dropped));
    }
    bufs.tp_next = tp_next;
    bufs.seen = ps.seen;
    bufs.quants = ps.quants;
    bufs.precincts = ps.precincts;
    let (mut tp_start, mut tp_idx) = (
        std::mem::take(&mut bufs.tp_start),
        std::mem::take(&mut bufs.tp_idx),
    );
    index_tile_parts(&parts, &mut tp_start, &mut tp_idx);
    Ok(Codestream {
        siz,
        cod,
        coc: hm.coc,
        qcd,
        qcc: hm.qcc,
        rgn: hm.rgn,
        poc: hm.poc,
        com: hm.com,
        tlm: hm.tlm,
        ppm: hm.ppm,
        tile_parts: parts,
        eoc,
        truncated: truncated_stream,
        warnings,
        parse_cost: ps.cost,
        ppm_merged: OnceLock::new(),
        ppt_merged: OnceLock::new(),
        tp_start,
        tp_idx,
    })
}

/// The main header of a raw codestream: its SIZ, COD, COCs and the offset of the first SOT.
/// Go's `Open` falls back to decoding as `decode_region` does when the headers after it fail.
pub(crate) fn parse_main(data: &[u8], limits: &Limits) -> Result<(Siz, Cod, Vec<Coc>, usize)> {
    let never = AtomicBool::new(false);
    let mut ps = ParseState {
        cancel: &never,
        markers: 0,
        warnings: Vec::new(),
        dropped: 0,
        keep: false,
        cost: 0,
        charged: 0,
        max_alloc: limits.max_alloc.min(i64::MAX as usize) as i64,
        seen: Vec::new(),
        hdr: 0,
        min_marker: 0xFF30,
        quants: Vec::new(),
        precincts: Vec::new(),
    };
    if data.len() < 2 || be16(&data[0..2]) != MARKER_SOC {
        return Err(invalid!("missing SOC marker at offset 0"));
    }
    if 4 > data.len() || be16(&data[2..]) != MARKER_SIZ {
        return Err(invalid!(
            "expected SIZ marker immediately after SOC at offset 2"
        ));
    }
    let (siz, pos) = parse_siz_segment(data, 2)?;
    let (hm, pos) = parse_markers(
        data,
        pos,
        MARKER_SOT,
        siz.csiz,
        true,
        false,
        siz.tiles_wide,
        siz.tiles_high,
        &mut ps,
    )?;
    let cod = hm
        .cod
        .ok_or_else(|| invalid!("missing mandatory COD marker in main header"))?;
    if hm.qcd.is_none() {
        return Err(invalid!("missing mandatory QCD marker in main header"));
    }
    Ok((siz, cod, hm.coc, pos))
}

/// Counts the tile-parts from `pos` by hopping Psot, up to the first one of unstated length, and
/// returns where the last starts.
fn count_tile_parts(data: &[u8], mut pos: usize) -> (usize, usize) {
    let mut n = 0;
    let mut last = pos;
    while pos + 12 <= data.len() && be16(&data[pos..]) == MARKER_SOT {
        n += 1;
        last = pos;
        let psot = be32(&data[pos + 6..]) as u64;
        if psot < 14 || pos as u64 + psot > data.len() as u64 {
            break;
        }
        pos += psot as usize;
    }
    (n, last)
}

/// A counting sort of the tile-parts by tile, into `start` and `idx`: tile t's parts are at
/// `idx[start[t]..start[t+1]]`, in codestream order.
fn index_tile_parts(parts: &[TilePart], start: &mut Vec<u32>, idx: &mut Vec<u32>) {
    let n = parts
        .iter()
        .map(|tp| tp.sot.isot as usize + 1)
        .max()
        .unwrap_or(0);
    start.clear();
    start.resize(n + 2, 0);
    for tp in parts {
        start[tp.sot.isot as usize + 2] += 1;
    }
    for t in 2..start.len() {
        start[t] += start[t - 1];
    }
    idx.clear();
    idx.resize(parts.len(), 0);
    for (i, tp) in parts.iter().enumerate() {
        let s = &mut start[tp.sot.isot as usize + 1];
        idx[*s as usize] = i as u32;
        *s += 1;
    }
    start.truncate(n + 1);
}

impl Codestream {
    /// Tile `tile`'s indices into `tile_parts`, in codestream order.
    pub fn tile_part_index(&self, tile: u16) -> &[u32] {
        let t = tile as usize;
        if t + 1 >= self.tp_start.len() {
            return &[];
        }
        &self.tp_idx[self.tp_start[t] as usize..self.tp_start[t + 1] as usize]
    }

    /// The merged PPM packet-header stream, merged once per codestream.
    pub fn ppm_headers(&self) -> Result<&[u8]> {
        self.ppm_merged
            .get_or_init(|| merge_ppm(&self.ppm))
            .as_ref()
            .map(|v| v.as_slice())
            .map_err(|e| e.clone())
    }

    /// Tile `tile`'s merged PPT packet-header stream, `None` when the tile has no PPT markers.
    /// PPT is scoped to one tile (T.800 A.7.5); every tile's stream is merged on the first call.
    pub fn ppt_headers(&self, tile: u16) -> Result<Option<&[u8]>> {
        let m = self
            .ppt_merged
            .get_or_init(|| {
                let mut by_tile: HashMap<u16, Vec<&PptEntry>> = HashMap::new();
                for tp in &self.tile_parts {
                    if !tp.ppt().is_empty() {
                        by_tile
                            .entry(tp.sot.isot)
                            .or_default()
                            .extend(tp.ppt().iter());
                    }
                }
                let mut merged = HashMap::with_capacity(by_tile.len());
                for (ti, entries) in by_tile {
                    merged.insert(ti, merge_ppt(entries)?);
                }
                Ok(merged)
            })
            .as_ref()
            .map_err(|e| e.clone())?;
        Ok(m.get(&tile).map(|v| v.as_slice()))
    }
}

/// Concatenates PPM marker payloads, ascending by Zppm, applying the Nppm length framing that
/// spans marker boundaries (T.800 A.7.4): the Nppm fields are framing only and are stripped.
fn merge_ppm(entries: &[PpmEntry]) -> Result<Vec<u8>> {
    let mut sorted: Vec<&PpmEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.z);
    let size: usize = sorted.iter().map(|e| e.data.len()).sum();
    let mut out = Vec::new();
    out.try_reserve_exact(size)
        .map_err(|_| Error::OutOfMemory)?;
    let mut remaining: u64 = 0; // bytes of the current Nppm run still owed
    for e in sorted {
        let mut data: &[u8] = &e.data;
        if remaining >= data.len() as u64 {
            out.extend_from_slice(data);
            remaining -= data.len() as u64;
            continue;
        }
        out.extend_from_slice(&data[..remaining as usize]);
        data = &data[remaining as usize..];
        remaining = 0;
        while !data.is_empty() {
            if data.len() < 4 {
                return Err(invalid!("PPM Zppm {}: not enough bytes to read Nppm", e.z));
            }
            let n = be32(data) as u64;
            data = &data[4..];
            if data.len() as u64 >= n {
                out.extend_from_slice(&data[..n as usize]);
                data = &data[n as usize..];
            } else {
                out.extend_from_slice(data);
                remaining = n - data.len() as u64;
                data = &[];
            }
        }
    }
    if remaining != 0 {
        return Err(invalid!(
            "corrupted PPM markers, {remaining} bytes of the final Nppm run never arrived"
        ));
    }
    Ok(out)
}

/// Concatenates PPT marker payloads for one tile, ascending by Zppt: no length framing.
fn merge_ppt(mut sorted: Vec<&PptEntry>) -> Result<Vec<u8>> {
    sorted.sort_by_key(|e| e.z);
    let n: usize = sorted.iter().map(|e| e.data.len()).sum();
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    for e in sorted {
        out.extend_from_slice(&e.data);
    }
    Ok(out)
}

/// The shared "1 byte if Csiz <= 256 else 2 bytes" rule used by COC, QCC, RGN and POC component
/// index fields.
fn component_index_width(csiz: u16) -> usize {
    if csiz <= 256 { 1 } else { 2 }
}

fn read_component_index(b: &[u8], width: usize) -> u16 {
    if width == 1 { b[0] as u16 } else { be16(b) }
}

/// Reads the Lxxx length field at `data[pos..]` (marker id already consumed) and returns the
/// payload (Lxxx-2 bytes) plus the offset of the next marker.
fn read_segment<'a>(
    data: &'a [u8],
    pos: usize,
    name: &str,
    marker_offset: usize,
) -> Result<(&'a [u8], usize)> {
    if pos.checked_add(2).is_none_or(|e| e > data.len()) {
        return Err(invalid!(
            "truncated {name} length field at offset {marker_offset}"
        ));
    }
    let l = be16(&data[pos..]);
    if l < 2 {
        return Err(invalid!(
            "invalid {name} length {l} at offset {marker_offset}"
        ));
    }
    let seg_end = pos + l as usize;
    if seg_end > data.len() {
        return Err(invalid!(
            "truncated {name} segment at offset {marker_offset} (length {l} exceeds remaining data)"
        ));
    }
    Ok((&data[pos + 2..seg_end], seg_end))
}

/// Parses SIZ. `pos` points at the marker id (0xFF51).
pub fn parse_siz_segment(data: &[u8], pos: usize) -> Result<(Siz, usize)> {
    parse_siz_in(data, pos, Vec::new())
}

/// `parse_siz_segment`, with the components in `components` (emptied first).
pub(crate) fn parse_siz_in(
    data: &[u8],
    pos: usize,
    mut components: Vec<SizComponent>,
) -> Result<(Siz, usize)> {
    components.clear();
    let marker_offset = pos;
    let Some(len_pos) = pos.checked_add(2) else {
        return Err(invalid!(
            "truncated SIZ length field at offset {marker_offset}"
        ));
    };
    let (payload, next) = read_segment(data, len_pos, "SIZ", marker_offset)?;
    if payload.len() < 36 {
        return Err(invalid!("SIZ segment too short at offset {marker_offset}"));
    }
    if !(payload.len() - 36).is_multiple_of(3) {
        return Err(invalid!(
            "SIZ component array length not a multiple of 3 at offset {marker_offset}"
        ));
    }
    let mut s = Siz {
        rsiz: be16(&payload[0..]),
        xsiz: be32(&payload[2..]),
        ysiz: be32(&payload[6..]),
        xosiz: be32(&payload[10..]),
        yosiz: be32(&payload[14..]),
        xtsiz: be32(&payload[18..]),
        ytsiz: be32(&payload[22..]),
        xtosiz: be32(&payload[26..]),
        ytosiz: be32(&payload[30..]),
        csiz: be16(&payload[34..]),
        components,
        tiles_wide: 0,
        tiles_high: 0,
    };
    if s.csiz < 1 || s.csiz > 16384 {
        return Err(invalid!(
            "SIZ Csiz {} out of range [1,16384] at offset {marker_offset}",
            s.csiz
        ));
    }
    if s.csiz as usize != (payload.len() - 36) / 3 {
        return Err(invalid!(
            "SIZ Csiz {} does not match component array length at offset {marker_offset}",
            s.csiz
        ));
    }
    if s.xosiz >= s.xsiz || s.yosiz >= s.ysiz {
        return Err(invalid!(
            "SIZ has zero or negative image size at offset {marker_offset}"
        ));
    }
    if s.xtsiz == 0 || s.ytsiz == 0 {
        return Err(invalid!("SIZ tile size is zero at offset {marker_offset}"));
    }
    let tx1 = sat_add32(s.xtosiz, s.xtsiz);
    let ty1 = sat_add32(s.ytosiz, s.ytsiz);
    if s.xtosiz > s.xosiz || s.ytosiz > s.yosiz || tx1 <= s.xosiz || ty1 <= s.yosiz {
        return Err(invalid!(
            "SIZ tile offset is illegal at offset {marker_offset}"
        ));
    }
    let comps = &payload[36..];
    s.components.reserve_exact(s.csiz as usize);
    for i in 0..s.csiz as usize {
        let b = &comps[i * 3..i * 3 + 3];
        let ssiz = b[0];
        let prec = (ssiz & 0x7F) + 1;
        if prec > 31 {
            return Err(invalid!(
                "SIZ component {i} precision {prec} exceeds 31 at offset {marker_offset}"
            ));
        }
        if b[1] == 0 || b[2] == 0 {
            return Err(invalid!(
                "SIZ component {i} has zero subsampling factor at offset {marker_offset}"
            ));
        }
        s.components.push(SizComponent {
            precision: prec,
            signed: ssiz & 0x80 != 0,
            xrsiz: b[1],
            yrsiz: b[2],
        });
    }

    // Promote to 64-bit before the ceil-div: Xsiz-XTOsiz+XTsiz can exceed 2^32 for a legal
    // single-tile image.
    let tw = ((s.xsiz - s.xtosiz) as u64).div_ceil(s.xtsiz as u64);
    let th = ((s.ysiz - s.ytosiz) as u64).div_ceil(s.ytsiz as u64);
    let (tw, th) = (tw as u32, th as u32);
    if tw == 0 || th == 0 || tw > 65535 / th {
        return Err(invalid!(
            "SIZ tile grid {tw}x{th} is illegal at offset {marker_offset}"
        ));
    }
    s.tiles_wide = tw;
    s.tiles_high = th;
    Ok((s, next))
}

fn sat_add32(a: u32, b: u32) -> u32 {
    a.saturating_add(b)
}

/// Parses the SPcod/SPcoc body shared by COD and COC. Returns the decoded style and the number
/// of payload bytes consumed. Custom precinct sizes go in `kept` (emptied first).
fn parse_coding_style(
    payload: &[u8],
    custom_precincts: bool,
    name: &str,
    marker_offset: usize,
    kept: Vec<PrecinctSize>,
) -> Result<(CodingStyle, usize)> {
    if payload.len() < 5 {
        return Err(invalid!(
            "{name} coding-style body too short at offset {marker_offset}"
        ));
    }
    let num_decomp = payload[0];
    let num_resolutions = num_decomp as usize + 1;
    if num_decomp > 32 {
        return Err(invalid!(
            "{name} decomposition level count {num_decomp} exceeds 32 at offset {marker_offset}"
        ));
    }
    // Reserved bits 4-7 are masked and tolerated rather than rejected.
    let xcb = (payload[1] & 0x0F) + 2;
    let ycb = (payload[2] & 0x0F) + 2;
    if xcb > 10 || ycb > 10 || xcb + ycb > 12 {
        return Err(invalid!(
            "{name} code-block size {}x{} is illegal at offset {marker_offset}",
            1u32 << xcb,
            1u32 << ycb
        ));
    }
    let cblk_style = payload[3];
    let transform = payload[4];
    if transform > 1 {
        return Err(invalid!(
            "{name} transformation {transform} is invalid at offset {marker_offset}"
        ));
    }
    let mut cs = CodingStyle {
        num_decomp_levels: num_decomp,
        code_block_width_exp: xcb,
        code_block_height_exp: ycb,
        code_block_style: cblk_style,
        transformation: transform,
        precinct_sizes: None,
    };
    let mut consumed = 5;
    if custom_precincts {
        if payload.len() - 5 < num_resolutions {
            return Err(invalid!(
                "{name} precinct size list too short at offset {marker_offset}"
            ));
        }
        let mut sizes = kept;
        sizes.clear();
        sizes.reserve_exact(num_resolutions);
        for i in 0..num_resolutions {
            let b = payload[5 + i];
            let (ppx, ppy) = (b & 0x0F, b >> 4);
            if i != 0 && (ppx == 0 || ppy == 0) {
                return Err(invalid!(
                    "{name} precinct exponent 0 at resolution {i} is illegal at offset {marker_offset}"
                ));
            }
            sizes.push(PrecinctSize { ppx, ppy });
        }
        cs.precinct_sizes = Some(sizes);
        consumed += num_resolutions;
    }
    Ok((cs, consumed))
}

pub fn parse_cod(payload: &[u8], marker_offset: usize) -> Result<Cod> {
    parse_cod_in(payload, marker_offset, Vec::new())
}

/// `parse_cod`, with custom precinct sizes in `kept` (emptied first).
fn parse_cod_in(payload: &[u8], marker_offset: usize, kept: Vec<PrecinctSize>) -> Result<Cod> {
    if payload.len() < 5 {
        return Err(invalid!("COD segment too short at offset {marker_offset}"));
    }
    let scod = payload[0];
    if scod & !0x07 != 0 {
        return Err(invalid!(
            "COD Scod value 0x{scod:02X} has reserved bits set at offset {marker_offset}"
        ));
    }
    let num_layers = be16(&payload[2..]);
    if num_layers < 1 {
        return Err(invalid!(
            "COD number of layers must be >= 1 at offset {marker_offset}"
        ));
    }
    let mct = payload[4];
    if mct > 1 {
        return Err(invalid!(
            "COD MCT value {mct} is invalid at offset {marker_offset}"
        ));
    }
    let custom = scod & 0x01 != 0;
    let (style, consumed) = parse_coding_style(&payload[5..], custom, "COD", marker_offset, kept)?;
    if consumed != payload.len() - 5 {
        return Err(invalid!(
            "COD segment has {} trailing bytes at offset {marker_offset}",
            payload.len() - 5 - consumed
        ));
    }
    Ok(Cod {
        scod,
        uses_custom_precincts: custom,
        uses_sop: scod & 0x02 != 0,
        uses_eph: scod & 0x04 != 0,
        progression_order: payload[1],
        number_of_layers: num_layers,
        multiple_component_transform: mct,
        coding_style: style,
    })
}

fn parse_coc(payload: &[u8], csiz: u16, marker_offset: usize) -> Result<Coc> {
    let room = component_index_width(csiz);
    if payload.len() < room + 1 {
        return Err(invalid!("COC segment too short at offset {marker_offset}"));
    }
    let ccoc = read_component_index(&payload[..room], room);
    if ccoc >= csiz {
        return Err(invalid!(
            "COC component index {ccoc} >= Csiz {csiz} at offset {marker_offset}"
        ));
    }
    let scoc = payload[room];
    let custom = scoc & 0x01 != 0;
    let (style, consumed) = parse_coding_style(
        &payload[room + 1..],
        custom,
        "COC",
        marker_offset,
        Vec::new(),
    )?;
    if consumed != payload.len() - room - 1 {
        return Err(invalid!(
            "COC segment has trailing bytes at offset {marker_offset}"
        ));
    }
    Ok(Coc {
        component_index: ccoc,
        scoc,
        uses_custom_precincts: custom,
        coding_style: style,
    })
}

/// Parses the SQcd/SQcc body shared by QCD and QCC. It consumes the entire remaining payload:
/// the band count is derived from the length, not declared.
fn parse_quantization(payload: &[u8], name: &str, marker_offset: usize) -> Result<Quantization> {
    parse_quantization_in(payload, name, marker_offset, Default::default())
}

/// `parse_quantization` into the exponent and mantissa vectors `kept` (emptied first).
fn parse_quantization_in(
    payload: &[u8],
    name: &str,
    marker_offset: usize,
    (mut exponents, mut mantissas): (Vec<u8>, Vec<u16>),
) -> Result<Quantization> {
    exponents.clear();
    mantissas.clear();
    if payload.is_empty() {
        return Err(invalid!(
            "{name} segment too short at offset {marker_offset}"
        ));
    }
    let sqcx = payload[0];
    let style = sqcx & 0x1F;
    let guard = sqcx >> 5;
    let body = &payload[1..];
    match style {
        1 => {
            // scalar derived: exactly one band, 2 bytes
            if body.len() != 2 {
                return Err(invalid!(
                    "{name} scalar-derived body must be 2 bytes at offset {marker_offset}"
                ));
            }
            exponents.push(body[0] >> 3);
            mantissas.push((((body[0] & 0x07) as u16) << 8) | body[1] as u16);
        }
        0 => {
            // no quantization: 1 byte per band
            exponents.extend(body.iter().map(|b| b >> 3));
            mantissas.resize(body.len(), 0);
        }
        2 => {
            // scalar expounded: 2 bytes per band
            if !body.len().is_multiple_of(2) {
                return Err(invalid!(
                    "{name} scalar-expounded body has a trailing byte at offset {marker_offset}"
                ));
            }
            let n = body.len() / 2;
            exponents.reserve_exact(n);
            mantissas.reserve_exact(n);
            for i in 0..n {
                let (b0, b1) = (body[2 * i], body[2 * i + 1]);
                exponents.push(b0 >> 3);
                mantissas.push((((b0 & 0x07) as u16) << 8) | b1 as u16);
            }
        }
        _ => {
            return Err(invalid!(
                "{name} has reserved quantization style {style} at offset {marker_offset}"
            ));
        }
    };
    Ok(Quantization {
        quantization_style: style,
        guard_bits: guard,
        exponents,
        mantissas,
    })
}

fn parse_rgn(payload: &[u8], csiz: u16, marker_offset: usize) -> Result<Rgn> {
    let room = component_index_width(csiz);
    if payload.len() != 2 + room {
        return Err(invalid!(
            "RGN segment length {} != {} at offset {marker_offset}",
            payload.len(),
            2 + room
        ));
    }
    let crgn = read_component_index(&payload[..room], room);
    if crgn >= csiz {
        return Err(invalid!(
            "RGN component index {crgn} >= Csiz {csiz} at offset {marker_offset}"
        ));
    }
    Ok(Rgn {
        component_index: crgn,
        srgn: payload[room],
        sprgn: payload[room + 1],
    })
}

fn parse_poc(payload: &[u8], csiz: u16, marker_offset: usize) -> Result<Poc> {
    let room = component_index_width(csiz);
    let chunk = 5 + 2 * room;
    if payload.is_empty() || !payload.len().is_multiple_of(chunk) {
        return Err(invalid!(
            "POC segment length {} is not a multiple of {chunk} at offset {marker_offset}",
            payload.len()
        ));
    }
    let n = payload.len() / chunk;
    let mut changes = Vec::with_capacity(n);
    for i in 0..n {
        let b = &payload[i * chunk..(i + 1) * chunk];
        let mut off = 1;
        let cspoc = read_component_index(&b[off..off + room], room);
        off += room;
        let lyepoc = be16(&b[off..]);
        off += 2;
        let repoc = b[off];
        off += 1;
        let cepoc = read_component_index(&b[off..off + room], room);
        off += room;
        let ppoc = b[off];
        changes.push(PocChange {
            rspoc: b[0],
            cspoc,
            lyepoc,
            repoc,
            cepoc,
            ppoc,
        });
    }
    Ok(Poc { changes })
}

fn parse_com(payload: &[u8], marker_offset: usize) -> Result<Com> {
    if payload.len() < 2 {
        return Err(invalid!("COM segment too short at offset {marker_offset}"));
    }
    Ok(Com {
        rcom: be16(payload),
        data: payload[2..].to_vec(),
    })
}

fn parse_tlm(
    payload: &[u8],
    marker_offset: usize,
    tiles_wide: u32,
    tiles_high: u32,
    ps: &mut ParseState,
) -> Result<Tlm> {
    if payload.len() < 2 {
        return Err(invalid!("TLM segment too short at offset {marker_offset}"));
    }
    let (ztlm, stlm) = (payload[0], payload[1]);
    let st = (stlm >> 4) & 0x03;
    let sp = (stlm >> 6) & 0x01;
    let body = &payload[2..];
    let mut tlm = Tlm {
        ztlm,
        stlm,
        entries: Vec::new(),
        invalid: false,
    };
    if st == 3 {
        tlm.invalid = true;
        ps.warn(|| {
            format!("TLM at offset {marker_offset}: ST field is invalid (3), index discarded")
        });
        return Ok(tlm);
    }
    let ptlm_size = (sp as usize + 1) * 2;
    let entry_size = ptlm_size + st as usize;
    if !body.len().is_multiple_of(entry_size) {
        tlm.invalid = true;
        ps.warn(|| {
            format!(
                "TLM at offset {marker_offset}: entry data is not a multiple of entry size, index discarded"
            )
        });
        return Ok(tlm);
    }
    let n = body.len() / entry_size;
    let tile_count = tiles_wide * tiles_high;
    tlm.entries.reserve_exact(n);
    for i in 0..n {
        let off = i * entry_size;
        let ttlm = match st {
            0 => i as u32,
            1 => body[off] as u32,
            _ => be16(&body[off..]) as u32,
        };
        if ttlm >= tile_count {
            tlm.invalid = true;
            ps.warn(|| {
                format!(
                    "TLM at offset {marker_offset}: tile index {ttlm} out of range (tile grid is {tiles_wide}x{tiles_high}), index discarded"
                )
            });
            return Ok(tlm);
        }
        let poff = off + st as usize;
        let ptlm = if sp == 0 {
            be16(&body[poff..]) as u32
        } else {
            be32(&body[poff..])
        };
        tlm.entries.push(TlmEntry { ttlm, ptlm });
    }
    Ok(tlm)
}

/// Decodes Iplt's 7-bit continuation-encoded packet lengths. Caps the continuation run at 5
/// bytes so the u64 accumulator never loses bits. Without `keep` the lengths are checked but not
/// stored.
pub fn parse_plt(payload: &[u8], marker_offset: usize, keep: bool) -> Result<Plt> {
    if payload.is_empty() {
        return Err(invalid!("PLT segment too short at offset {marker_offset}"));
    }
    let zplt = payload[0];
    let mut lengths = Vec::new();
    let mut cur: u64 = 0;
    let mut cont_bytes = 0;
    for &b in &payload[1..] {
        cur |= (b & 0x7F) as u64;
        if b & 0x80 != 0 {
            cont_bytes += 1;
            if cont_bytes > 5 {
                return Err(invalid!(
                    "PLT packet length continuation run exceeds 5 bytes at offset {marker_offset}"
                ));
            }
            cur <<= 7;
        } else {
            if cur > 0xFFFF_FFFF {
                return Err(invalid!(
                    "PLT packet length exceeds 1<<32-1 at offset {marker_offset}"
                ));
            }
            if keep {
                lengths.push(cur);
            }
            cur = 0;
            cont_bytes = 0;
        }
    }
    if cont_bytes > 0 {
        return Err(invalid!(
            "PLT truncated, trailing continuation byte at offset {marker_offset}"
        ));
    }
    Ok(Plt {
        zplt,
        packet_lengths: lengths,
    })
}

fn parse_sot_segment(payload: &[u8], marker_offset: usize) -> Result<Sot> {
    if payload.len() != 8 {
        return Err(invalid!(
            "SOT segment length {} != 8 at offset {marker_offset}",
            payload.len()
        ));
    }
    let sot = Sot {
        isot: be16(payload),
        psot: be32(&payload[2..]),
        tpsot: payload[6],
        tnsot: payload[7],
    };
    if sot.psot != 0 && sot.psot < 14 && sot.psot != 12 {
        return Err(invalid!(
            "SOT Psot {} is illegal (must be 0 or >= 14) at offset {marker_offset}",
            sot.psot
        ));
    }
    Ok(sot)
}

/// The markers parsed by `parse_markers`, shared by the main header and every tile-part header.
#[derive(Default)]
struct HeaderMarkers {
    cod: Option<Cod>,
    coc: Vec<Coc>,
    qcd: Option<Qcd>,
    qcc: Vec<Qcc>,
    rgn: Vec<Rgn>,
    poc: Vec<Poc>,
    com: Vec<Com>,
    tlm: Vec<Tlm>,
    plt: Vec<Plt>,
    ppm: Vec<PpmEntry>,
    ppt: Vec<PptEntry>,
}

/// Reads marker segments starting at `pos` until it sees `terminator` (SOT for the main header,
/// SOD for a tile-part header), returning the offset of that terminator's marker id.
/// `allow_tlm`/`allow_plt` gate the two markers that are legal in only one of the two contexts;
/// an out-of-place TLM/PLT is skipped with a warning. `allow_tlm` doubles as "this is the main
/// header".
#[allow(clippy::too_many_arguments)]
fn parse_markers(
    data: &[u8],
    mut pos: usize,
    terminator: u16,
    csiz: u16,
    allow_tlm: bool,
    allow_plt: bool,
    tiles_wide: u32,
    tiles_high: u32,
    ps: &mut ParseState,
) -> Result<(HeaderMarkers, usize)> {
    let mut hm = HeaderMarkers::default();
    let mut poc_changes = 0usize; // POC accumulates across markers; J2K_MAX_POCS is 32
    ps.hdr += 1;
    loop {
        ps.markers += 1;
        if ps.markers & (CANCEL_EVERY - 1) == 0 {
            check_cancel(ps.cancel)?;
        }
        if pos + 2 > data.len() {
            return Err(invalid!(
                "truncated codestream at offset {pos}, expected marker"
            ));
        }
        let id = be16(&data[pos..]);
        if id == terminator {
            return Ok((hm, pos));
        }
        if id < ps.min_marker {
            return Err(invalid!("invalid marker 0x{id:04X} at offset {pos}"));
        }
        let marker_offset = pos;

        // T.800 Table A.2: 0xFF30-0xFF3F are reserved delimiting markers with no marker
        // segment (no length field).
        if (0xFF30..=0xFF3F).contains(&id) {
            pos += 2;
            continue;
        }

        match id {
            MARKER_SOC | MARKER_SOD | MARKER_EOC | MARKER_SOP | MARKER_EPH => {
                return Err(invalid!(
                    "unexpected {} marker (no length field) at offset {marker_offset}",
                    marker_name(id)
                ));
            }
            MARKER_SIZ => {
                return Err(invalid!(
                    "misplaced SIZ marker at offset {marker_offset} (SIZ is only legal once, immediately after SOC)"
                ));
            }
            MARKER_PPM => {
                let (payload, next) = read_segment(data, pos + 2, "PPM", marker_offset)?;
                if !allow_tlm {
                    // PPM is main-header only; silently dropping a misplaced one would make
                    // the packet headers be read out of the tile data: garbage.
                    return Err(invalid!(
                        "PPM marker outside main header at offset {marker_offset}"
                    ));
                }
                // OpenJPEG rejects a Z byte with an empty payload; match it.
                if payload.len() < 2 {
                    return Err(invalid!("PPM segment too short at offset {marker_offset}"));
                }
                let z = payload[0];
                if hm.ppm.iter().any(|e| e.z == z) {
                    return Err(invalid!(
                        "PPM Zppm {z} already read at offset {marker_offset}"
                    ));
                }
                // Twice the payload: the merged header stream is a second copy.
                ps.spend(
                    2 * size_of::<PpmEntry>() as i64 + 2 * payload.len() as i64,
                    next,
                )?;
                hm.ppm.push(PpmEntry {
                    z,
                    data: payload[1..].to_vec(),
                });
                pos = next;
            }
            MARKER_PPT => {
                let (payload, next) = read_segment(data, pos + 2, "PPT", marker_offset)?;
                if !allow_plt {
                    return Err(invalid!(
                        "PPT marker outside tile-part header at offset {marker_offset}"
                    ));
                }
                if payload.len() < 2 {
                    return Err(invalid!("PPT segment too short at offset {marker_offset}"));
                }
                let z = payload[0];
                if hm.ppt.iter().any(|e| e.z == z) {
                    return Err(invalid!(
                        "PPT Zppt {z} already read at offset {marker_offset}"
                    ));
                }
                // Twice the payload: the merged header stream is a second copy.
                ps.spend(
                    2 * size_of::<PptEntry>() as i64 + 2 * payload.len() as i64,
                    next,
                )?;
                hm.ppt.push(PptEntry {
                    z,
                    data: payload[1..].to_vec(),
                });
                pos = next;
            }
            MARKER_COD => {
                let (payload, next) = read_segment(data, pos + 2, "COD", marker_offset)?;
                let kept = ps.precincts.pop().unwrap_or_default();
                let cod = parse_cod_in(payload, marker_offset, kept)?;
                hm.cod = Some(cod);
                ps.spend(size_of::<Cod>() as i64 + 2 * payload.len() as i64, next)?;
                pos = next;
            }
            MARKER_COC => {
                let (payload, next) = read_segment(data, pos + 2, "COC", marker_offset)?;
                let coc = parse_coc(payload, csiz, marker_offset)?;
                let mut cost = size_of::<Coc>() as i64 + 2 * payload.len() as i64;
                let dup = ps.repeat(0, coc.component_index, csiz, next)?;
                if !dup {
                    hm.coc.push(coc);
                    cost += 2 * size_of::<Coc>() as i64;
                }
                ps.spend(cost, next)?;
                pos = next;
            }
            MARKER_QCD => {
                let (payload, next) = read_segment(data, pos + 2, "QCD", marker_offset)?;
                if payload.is_empty() {
                    return Err(invalid!("QCD segment too short at offset {marker_offset}"));
                }
                let kept = ps.quants.pop().unwrap_or_default();
                let q = parse_quantization_in(payload, "QCD", marker_offset, kept)?;
                hm.qcd = Some(Qcd {
                    sqcd: payload[0],
                    quantization: q,
                });
                ps.spend(
                    size_of::<Qcd>() as i64 + 3 * payload.len() as i64 + 32,
                    next,
                )?;
                pos = next;
            }
            MARKER_QCC => {
                let (payload, next) = read_segment(data, pos + 2, "QCC", marker_offset)?;
                let room = component_index_width(csiz);
                if payload.len() < room + 1 {
                    return Err(invalid!("QCC segment too short at offset {marker_offset}"));
                }
                let cqcc = read_component_index(&payload[..room], room);
                if cqcc >= csiz {
                    return Err(invalid!(
                        "QCC component index {cqcc} >= Csiz {csiz} at offset {marker_offset}"
                    ));
                }
                let kept = ps.quants.pop().unwrap_or_default();
                let q = parse_quantization_in(&payload[room..], "QCC", marker_offset, kept)?;
                let mut cost = 3 * payload.len() as i64;
                if !q.exponents.is_empty() {
                    cost += 32;
                }
                let dup = ps.repeat(1, cqcc, csiz, next)?;
                if !dup {
                    hm.qcc.push(Qcc {
                        component_index: cqcc,
                        sqcc: payload[room],
                        quantization: q,
                    });
                    cost += 2 * size_of::<Qcc>() as i64;
                }
                ps.spend(cost, next)?;
                pos = next;
            }
            MARKER_RGN => {
                let (payload, next) = read_segment(data, pos + 2, "RGN", marker_offset)?;
                let rgn = parse_rgn(payload, csiz, marker_offset)?;
                let mut cost = size_of::<Rgn>() as i64;
                let dup = ps.repeat(2, rgn.component_index, csiz, next)?;
                if !dup {
                    hm.rgn.push(rgn);
                    cost += 2 * size_of::<Rgn>() as i64;
                }
                ps.spend(cost, next)?;
                pos = next;
            }
            MARKER_POC => {
                let (payload, next) = read_segment(data, pos + 2, "POC", marker_offset)?;
                let poc = parse_poc(payload, csiz, marker_offset)?;
                poc_changes += poc.changes.len();
                if poc_changes >= 32 {
                    return Err(invalid!(
                        "too many POC changes ({poc_changes} >= 32) at offset {marker_offset}"
                    ));
                }
                hm.poc.push(poc);
                ps.spend(2 * size_of::<Poc>() as i64 + 2 * payload.len() as i64, next)?;
                pos = next;
            }
            MARKER_COM => {
                let (payload, next) = read_segment(data, pos + 2, "COM", marker_offset)?;
                if ps.keep {
                    // Spent before the payload is copied.
                    ps.spend(2 * size_of::<Com>() as i64 + 2 * payload.len() as i64, next)?;
                }
                // A decode keeps no COM: only check it.
                if ps.keep {
                    hm.com.push(parse_com(payload, marker_offset)?);
                } else if payload.len() < 2 {
                    return Err(invalid!("COM segment too short at offset {marker_offset}"));
                }
                pos = next;
            }
            MARKER_TLM => {
                let (payload, next) = read_segment(data, pos + 2, "TLM", marker_offset)?;
                if !allow_tlm {
                    ps.warn(|| {
                        format!("TLM marker outside main header at offset {marker_offset}, skipped")
                    });
                    pos = next;
                    continue;
                }
                let tlm = parse_tlm(payload, marker_offset, tiles_wide, tiles_high, ps)?;
                hm.tlm.push(tlm);
                ps.spend(2 * size_of::<Tlm>() as i64 + 4 * payload.len() as i64, next)?;
                pos = next;
            }
            MARKER_PLT => {
                let (payload, next) = read_segment(data, pos + 2, "PLT", marker_offset)?;
                if !allow_plt {
                    ps.warn(|| {
                        format!(
                            "PLT marker outside tile-part header at offset {marker_offset}, skipped"
                        )
                    });
                    pos = next;
                    continue;
                }
                if ps.keep {
                    // Spent before the length list is built.
                    ps.spend(2 * size_of::<Plt>() as i64 + 8 * payload.len() as i64, next)?;
                }
                let plt = parse_plt(payload, marker_offset, ps.keep)?;
                if ps.keep {
                    hm.plt.push(plt);
                }
                pos = next;
            }
            _ => {
                let r = read_segment(data, pos + 2, "", marker_offset);
                let (payload, next) = match r {
                    Ok(v) => v,
                    Err(_) => {
                        // Named only now: the name is formatted per marker.
                        read_segment(data, pos + 2, &marker_name(id), marker_offset)?;
                        return Err(invalid!("invalid segment at offset {marker_offset}"));
                    }
                };
                // Unknown markers are tolerated only in the main header.
                if !allow_tlm {
                    return Err(invalid!(
                        "unknown marker {} not allowed in tile-part header at offset {marker_offset}",
                        marker_name(id)
                    ));
                }
                let plen = payload.len();
                ps.warn(|| {
                    format!(
                        "unknown marker {} at offset {marker_offset} (length {}), skipped",
                        marker_name(id),
                        plen + 2
                    )
                });
                pos = next;
            }
        }
    }
}

/// Used when Psot == 0 ("runs to next SOT or EOC") or Psot == 12 (Kakadu's empty-SOT quirk).
/// Scanning for a raw 0xFF90/0xFFD9 byte pair is safe: bit stuffing guarantees a genuine 0xFF in
/// packet data is never followed by a byte > 0x8F.
fn find_tile_data_end(data: &[u8], data_start: usize) -> usize {
    let mut i = data_start;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (data[i + 1] == 0x90 || data[i + 1] == 0xD9) {
            return i;
        }
        i += 1;
    }
    data.len()
}

/// Parses one tile-part starting at the SOT marker id (`pos`) and returns it plus the offset of
/// the following marker. The bool reports a truncated tile-part (Psot points past the end of
/// data): the caller stops walking tile-parts but treats the parse as successful.
fn parse_tile_part(
    data: &[u8],
    pos: usize,
    csiz: u16,
    ps: &mut ParseState,
) -> Result<(TilePart, usize, bool)> {
    let seg_start = pos;
    let (sot_payload, next) = read_segment(data, pos + 2, "SOT", pos)?;
    let sot = parse_sot_segment(sot_payload, pos)?;
    let mut pos = next;

    if sot.psot == 12 {
        // Kakadu's "empty SOT" quirk: a warning, not an error.
        ps.warn(|| format!("SOT at offset {seg_start}: empty SOT marker detected (Psot==12)"));
    }

    let (hm, next) = parse_markers(data, pos, MARKER_SOD, csiz, false, true, 0, 0, ps)?;
    pos = next; // at SOD marker id
    if pos + 2 > data.len() {
        return Err(invalid!("truncated SOD marker at offset {pos}"));
    }
    pos += 2; // past SOD, start of tile-part data
    let data_start = pos;

    let data_end;
    let mut truncated = false;
    if sot.psot == 0 || sot.psot == 12 {
        data_end = find_tile_data_end(data, data_start);
        if data_end == data.len() {
            // Scanned to the end of the stream without finding a following SOT or EOC.
            truncated = true;
        }
    } else {
        let end = seg_start as u64 + sot.psot as u64;
        if end > data.len() as u64 {
            data_end = data.len();
            truncated = true;
        } else {
            data_end = end as usize;
        }
        if data_end < data_start {
            return Err(invalid!(
                "SOT Psot {} puts tile-part data out of bounds at offset {seg_start}",
                sot.psot
            ));
        }
    }

    let any = hm.cod.is_some()
        || hm.qcd.is_some()
        || !(hm.coc.is_empty()
            && hm.qcc.is_empty()
            && hm.rgn.is_empty()
            && hm.poc.is_empty()
            && hm.com.is_empty()
            && hm.plt.is_empty()
            && hm.ppt.is_empty());
    let markers = if any {
        Some(Box::new(TpMarkers {
            cod: hm.cod,
            coc: hm.coc,
            qcd: hm.qcd,
            qcc: hm.qcc,
            rgn: hm.rgn,
            poc: hm.poc,
            com: hm.com,
            plt: hm.plt,
            ppt: hm.ppt,
        }))
    } else {
        None
    };
    let tp = TilePart {
        sot,
        markers,
        data_offset: data_start,
        data_length: data_end - data_start,
    };
    Ok((tp, data_end, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16b(v: u16) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    fn u32b(v: u32) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    /// A full marker segment: 2-byte id + 2-byte length + payload.
    fn segment(id: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = u16b(id);
        out.extend(u16b(2 + payload.len() as u16));
        out.extend_from_slice(payload);
        out
    }

    fn cat(chunks: &[&[u8]]) -> Vec<u8> {
        chunks.concat()
    }

    static NEVER: AtomicBool = AtomicBool::new(false);

    fn state() -> ParseState<'static> {
        ParseState {
            cancel: &NEVER,
            markers: 0,
            warnings: Vec::new(),
            dropped: 0,
            keep: true,
            cost: 0,
            charged: 0,
            max_alloc: i64::MAX,
            seen: Vec::new(),
            hdr: 0,
            min_marker: 0xFF00,
            quants: Vec::new(),
            precincts: Vec::new(),
        }
    }

    fn lim() -> Limits {
        Limits::default()
    }

    fn fixture_siz_payload(csiz: u16) -> Vec<u8> {
        let mut p = cat(&[
            &u16b(0),
            &u32b(256),
            &u32b(256),
            &u32b(0),
            &u32b(0),
            &u32b(128),
            &u32b(128),
            &u32b(0),
            &u32b(0),
            &u16b(csiz),
        ]);
        for _ in 0..csiz {
            p.extend_from_slice(&[7, 1, 1]); // 8-bit unsigned, XR=YR=1
        }
        p
    }

    fn fixture_cod_payload(custom: bool) -> Vec<u8> {
        let mut p = cat(&[
            &[if custom { 1 } else { 0 }, 0x00], // Scod, progression order LRCP
            &u16b(1),                            // layers
            &[0x00],                             // mct
            &[0x01, 0x02, 0x02, 0x00, 0x00],     // decomp=1, xcb, ycb, style, reversible
        ]);
        if custom {
            p.extend_from_slice(&[0x44, 0x44]); // 2 resolutions, PPx=PPy=4 each
        }
        p
    }

    fn fixture_qcd_payload() -> Vec<u8> {
        // no-quantization style, 1 decomposition level -> 4 bands
        vec![0x00, 8, 16, 24, 32]
    }

    /// A minimal, fully valid one-tile-part codestream.
    fn fixture_codestream() -> Vec<u8> {
        let mut b = u16b(MARKER_SOC);
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        b.extend(segment(MARKER_COD, &fixture_cod_payload(false)));
        b.extend(segment(MARKER_QCD, &fixture_qcd_payload()));
        b.extend(segment(MARKER_COM, &cat(&[&u16b(1), b"hi"])));
        let sot_offset = b.len();
        b.extend(segment(
            MARKER_SOT,
            &cat(&[&u16b(0), &u32b(0), &[0x00, 0x01]]),
        ));
        b.extend(u16b(MARKER_SOD));
        b.extend_from_slice(&[0x01, 0x02, 0x03, 0x04]);
        let data_end = b.len();
        b.extend(u16b(MARKER_EOC));
        let psot = (data_end - sot_offset) as u32;
        b[sot_offset + 6..sot_offset + 10].copy_from_slice(&psot.to_be_bytes());
        b
    }

    /// SOC + SIZ + COD + QCD, the start of most hand-built streams.
    fn head() -> Vec<u8> {
        let mut b = u16b(MARKER_SOC);
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        b.extend(segment(MARKER_COD, &fixture_cod_payload(false)));
        b.extend(segment(MARKER_QCD, &fixture_qcd_payload()));
        b
    }

    #[test]
    fn parse_codestream_valid() {
        let cs = parse(&fixture_codestream(), &lim()).unwrap();
        assert_eq!((cs.siz.xsiz, cs.siz.csiz), (256, 1));
        assert_eq!((cs.siz.tiles_wide, cs.siz.tiles_high), (2, 2));
        assert_eq!(cs.cod.number_of_layers, 1);
        assert_eq!(cs.cod.coding_style.num_decomp_levels, 1);
        assert_eq!(cs.qcd.quantization.exponents.len(), 4);
        assert_eq!(cs.com.len(), 1);
        assert_eq!((cs.com[0].rcom, cs.com[0].data.as_slice()), (1, &b"hi"[..]));
        assert_eq!(cs.tile_parts.len(), 1);
        let tp = &cs.tile_parts[0];
        assert_eq!((tp.sot.isot, tp.sot.tnsot, tp.data_length), (0, 1, 4));
        assert!(cs.eoc);
    }

    #[test]
    fn parse_codestream_errors() {
        let valid = fixture_codestream();
        let sot = segment(MARKER_SOT, &cat(&[&u16b(0), &u32b(14), &[0, 1]]));
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("missing SOC", vec![0, 0, 0, 0]),
            ("empty", vec![]),
            (
                "SIZ not immediately after SOC",
                cat(&[
                    &u16b(MARKER_SOC),
                    &segment(MARKER_COD, &fixture_cod_payload(false)),
                ]),
            ),
            (
                "truncated after SOC+SIZ id",
                cat(&[&u16b(MARKER_SOC), &u16b(MARKER_SIZ)]),
            ),
            ("truncated whole file", valid[..valid.len() - 10].to_vec()),
            (
                "missing COD",
                cat(&[
                    &u16b(MARKER_SOC),
                    &segment(MARKER_SIZ, &fixture_siz_payload(1)),
                    &segment(MARKER_QCD, &fixture_qcd_payload()),
                    &sot,
                ]),
            ),
            (
                "missing QCD",
                cat(&[
                    &u16b(MARKER_SOC),
                    &segment(MARKER_SIZ, &fixture_siz_payload(1)),
                    &segment(MARKER_COD, &fixture_cod_payload(false)),
                    &sot,
                ]),
            ),
        ];
        for (name, data) in cases {
            assert!(parse(&data, &lim()).is_err(), "{name}");
        }
    }

    #[test]
    fn parse_siz() {
        let data = segment(MARKER_SIZ, &fixture_siz_payload(3));
        let (siz, next) = parse_siz_segment(&cat(&[&u16b(MARKER_SOC), &data]), 2).unwrap();
        assert_eq!((siz.csiz, siz.components.len()), (3, 3));
        assert_eq!(next, 2 + data.len());
        for c in &siz.components {
            assert!(c.precision == 8 && !c.signed);
        }
        // Csiz claims 2 components but only 1's worth of data follows.
        let mut p = fixture_siz_payload(1);
        p[34] = 0;
        p[35] = 2;
        assert!(
            parse_siz_segment(&cat(&[&u16b(MARKER_SOC), &segment(MARKER_SIZ, &p)]), 2).is_err()
        );
    }

    #[test]
    fn parse_cod_fields_and_errors() {
        let cod = parse_cod(&fixture_cod_payload(false), 0).unwrap();
        assert_eq!(
            (
                cod.progression_order,
                cod.number_of_layers,
                cod.multiple_component_transform
            ),
            (0, 1, 0)
        );
        let s = &cod.coding_style;
        assert_eq!(
            (
                s.num_decomp_levels,
                s.code_block_width_exp,
                s.code_block_height_exp,
                s.transformation
            ),
            (1, 4, 4, 0)
        );
        let cp = parse_cod(&fixture_cod_payload(true), 0).unwrap();
        let ps = cp.coding_style.precinct_sizes.unwrap();
        assert!(cp.uses_custom_precincts && ps.len() == 2);
        assert_eq!((ps[0].ppx, ps[0].ppy), (4, 4));

        assert!(parse_cod(&[0, 0, 0, 1], 0).is_err());
        let mut bad = fixture_cod_payload(false);
        bad[0] = 0xF0;
        assert!(parse_cod(&bad, 0).is_err(), "reserved Scod bits");
        let mut bad = fixture_cod_payload(false);
        bad[2] = 0;
        bad[3] = 0;
        assert!(parse_cod(&bad, 0).is_err(), "zero layers");
    }

    #[test]
    fn parse_coc_widths() {
        let payload = cat(&[&[2], &[0x00], &[0x01, 0x02, 0x02, 0x00, 0x00]]);
        let coc = parse_coc(&payload, 4, 0).unwrap();
        assert_eq!(
            (coc.component_index, coc.coding_style.num_decomp_levels),
            (2, 1)
        );
        assert!(
            parse_coc(&payload, 2, 0).is_err(),
            "component index out of range"
        );
        let wide = cat(&[&[0x01, 0x00], &[0x00], &[0x01, 0x02, 0x02, 0x00, 0x00]]);
        assert_eq!(parse_coc(&wide, 300, 0).unwrap().component_index, 256);
    }

    #[test]
    fn parse_qcd_styles() {
        let q = parse_quantization(&[0x20, 8, 16], "QCD", 0).unwrap();
        assert_eq!(
            (
                q.quantization_style,
                q.guard_bits,
                q.exponents.len(),
                q.exponents[0]
            ),
            (0, 1, 2, 1)
        );
        let q1 = parse_quantization(&[0x41, 0x28, 0x03], "QCD", 0).unwrap();
        assert_eq!((q1.quantization_style, q1.exponents.len()), (1, 1));
        assert!(parse_quantization(&[0x01, 0x28], "QCD", 0).is_err());
        let q2 = parse_quantization(&[0x42, 0x28, 0x03, 0x30, 0x04], "QCD", 0).unwrap();
        assert_eq!((q2.quantization_style, q2.exponents.len()), (2, 2));
        assert_eq!((q2.exponents[0], q2.mantissas[0]), (5, 0x03));
        // Styles 3..31 are reserved (T.800 Table A.28).
        assert!(parse_quantization(&[0x43, 0x28, 0x03], "QCD", 0).is_err());
    }

    #[test]
    fn parse_qcc_marker() {
        let data = cat(&[
            &u16b(MARKER_SOC),
            &segment(MARKER_QCC, &[5, 0x00, 8]),
            &u16b(0xFFFF),
        ]);
        let (hm, _) =
            parse_markers(&data, 2, 0xFFFF, 10, false, false, 0, 0, &mut state()).unwrap();
        assert_eq!(hm.qcc.len(), 1);
        assert_eq!(hm.qcc[0].component_index, 5);
    }

    #[test]
    fn parse_rgn_poc_com() {
        let rgn = parse_rgn(&[1, 0, 5], 4, 0).unwrap();
        assert_eq!((rgn.component_index, rgn.srgn, rgn.sprgn), (1, 0, 5));
        assert!(parse_rgn(&[1, 0], 4, 0).is_err());
        assert!(parse_rgn(&[9, 0, 5], 4, 0).is_err());

        let chunk = [0, 1, 0, 4, 3, 3, 2];
        let poc = parse_poc(&chunk, 4, 0).unwrap();
        assert_eq!(poc.changes.len(), 1);
        let c = poc.changes[0];
        assert_eq!(
            (c.rspoc, c.cspoc, c.lyepoc, c.repoc, c.cepoc, c.ppoc),
            (0, 1, 4, 3, 3, 2)
        );
        assert!(parse_poc(&chunk[..6], 4, 0).is_err());

        let com = parse_com(&cat(&[&u16b(1), b"hello"]), 0).unwrap();
        assert_eq!((com.rcom, com.data.as_slice()), (1, &b"hello"[..]));
        assert!(parse_com(&[0], 0).is_err());
    }

    #[test]
    fn parse_tlm_entries_and_invalid() {
        let payload = [0x00, 0x10, 0x00, 0x00, 0x10, 0x01, 0x00, 0x20];
        let mut ps = state();
        let tlm = parse_tlm(&payload, 0, 2, 1, &mut ps).unwrap();
        assert!(ps.warnings.is_empty());
        assert!(!tlm.invalid && tlm.entries.len() == 2);
        assert_eq!(tlm.entries[0], TlmEntry { ttlm: 0, ptlm: 16 });
        assert_eq!(tlm.entries[1], TlmEntry { ttlm: 1, ptlm: 32 });
        // ST == 3 is invalid -> marked invalid, no hard error.
        let bad = parse_tlm(&[0x00, 0x30], 0, 2, 1, &mut ps).unwrap();
        assert!(bad.invalid && ps.warnings.len() == 1);
        // Ttlm=9 is way out of range for a single-tile grid.
        let mut ps = state();
        let oor = parse_tlm(&[0x00, 0x10, 0x09, 0x01, 0x00], 0, 1, 1, &mut ps).unwrap();
        assert!(oor.invalid && ps.warnings.len() == 1);
    }

    fn encode_7bit(mut v: u64) -> Vec<u8> {
        let mut groups = vec![(v & 0x7F) as u8];
        v >>= 7;
        while v > 0 {
            groups.push((v & 0x7F) as u8);
            v >>= 7;
        }
        groups.reverse();
        let n = groups.len();
        for g in groups.iter_mut().take(n - 1) {
            *g |= 0x80;
        }
        groups
    }

    #[test]
    fn parse_plt_lengths_and_errors() {
        let payload = cat(&[&[0], &encode_7bit(200), &encode_7bit(5)]);
        let plt = parse_plt(&payload, 0, true).unwrap();
        assert_eq!(plt.packet_lengths, vec![200, 5]);
        assert!(
            parse_plt(&[0, 0x81], 0, true).is_err(),
            "trailing continuation"
        );
        let over = cat(&[&[0], &encode_7bit(1 << 33)]);
        assert!(parse_plt(&over, 0, true).is_err());
        let long_run = [
            0, 0x81, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x05,
        ];
        assert!(parse_plt(&long_run, 0, true).is_err());
    }

    #[test]
    fn parse_sot() {
        let payload = cat(&[&u16b(3), &u32b(1000), &[2, 5]]);
        let sot = parse_sot_segment(&payload, 0).unwrap();
        assert_eq!((sot.isot, sot.psot, sot.tpsot, sot.tnsot), (3, 1000, 2, 5));
        assert!(parse_sot_segment(&cat(&[&u16b(0), &u32b(5), &[0, 0]]), 0).is_err());
        assert!(parse_sot_segment(&cat(&[&u16b(0), &u32b(12), &[0, 0]]), 0).is_ok());
        assert!(parse_sot_segment(&payload[..7], 0).is_err());
    }

    #[test]
    fn unknown_marker_skipped_with_warning() {
        let data = cat(&[&u16b(MARKER_SOC), &segment(0xFF63, &[1, 2]), &u16b(0xFFFF)]);
        let mut ps = state();
        let (_, next) = parse_markers(&data, 2, 0xFFFF, 1, true, false, 0, 0, &mut ps).unwrap();
        assert_eq!(ps.warnings.len(), 1);
        assert_eq!(next, data.len() - 2);
        // Tolerated only in the main header.
        assert!(parse_markers(&data, 2, 0xFFFF, 1, false, true, 0, 0, &mut state()).is_err());
    }

    /// A header of millions of markers notices a cancel.
    #[test]
    fn marker_loop_checks_cancel() {
        static CANCELLED: AtomicBool = AtomicBool::new(true);
        let mut data = u16b(MARKER_SOC);
        for _ in 0..2 * CANCEL_EVERY {
            data.extend_from_slice(&u16b(0xFF30));
        }
        data.extend_from_slice(&u16b(0xFFFF));
        let mut ps = state();
        ps.cancel = &CANCELLED;
        let r = parse_markers(&data, 2, 0xFFFF, 1, true, false, 0, 0, &mut ps);
        assert!(matches!(r, Err(Error::Cancelled)));
    }

    /// 0xFF30-0xFF3F have no length field.
    #[test]
    fn delimiting_marker_has_no_length_field() {
        let data = cat(&[&u16b(MARKER_SOC), &u16b(0xFF30), &u16b(0xFFFF)]);
        let mut ps = state();
        let (_, next) = parse_markers(&data, 2, 0xFFFF, 1, true, false, 0, 0, &mut ps).unwrap();
        assert!(ps.warnings.is_empty());
        assert_eq!(next, data.len() - 2);
    }

    #[test]
    fn read_segment_truncation() {
        assert!(read_segment(&[0x00], 0, "TEST", 0).is_err());
        assert!(read_segment(&[0x00, 0x01], 0, "TEST", 0).is_err());
        assert!(read_segment(&[0x00, 0x0A], 0, "TEST", 0).is_err());
    }

    #[test]
    fn find_tile_data_end_and_index_width() {
        assert_eq!(find_tile_data_end(&[1, 2, 0xFF, 0, 3, 0xFF, 0x90], 0), 5);
        let no = [1, 0xFF, 0, 2];
        assert_eq!(find_tile_data_end(&no, 0), no.len());
        assert_eq!(component_index_width(256), 1);
        assert_eq!(component_index_width(257), 2);
    }

    #[test]
    fn empty_sot_psot_12_is_a_warning() {
        let mut b = head();
        b.extend(segment(MARKER_SOT, &cat(&[&u16b(0), &u32b(12), &[0, 1]])));
        b.extend(u16b(MARKER_SOD));
        b.extend_from_slice(&[1, 2, 3, 4]);
        b.extend(u16b(MARKER_EOC));
        let cs = parse(&b, &lim()).unwrap();
        assert!(cs.tile_parts.len() == 1 && cs.tile_parts[0].data_length == 4);
        assert_eq!(cs.warnings.len(), 1);
        assert!(cs.eoc);
    }

    #[test]
    fn duplicate_siz_and_poc_cap() {
        let mut b = u16b(MARKER_SOC);
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        assert!(parse(&b, &lim()).is_err());

        let mut b = head();
        let chunks: Vec<u8> = std::iter::repeat_n([0u8, 0, 0, 1, 0, 0, 0], 40)
            .flatten()
            .collect();
        b.extend(segment(MARKER_POC, &chunks));
        assert!(
            parse(&b, &lim()).is_err(),
            "POC accumulation over 32 changes"
        );
    }

    fn tile_part(tpsot: u8, tnsot: u8, markers: &[u8], data: &[u8]) -> Vec<u8> {
        let mut b = segment(MARKER_SOT, &cat(&[&u16b(0), &u32b(0), &[tpsot, tnsot]]));
        b.extend_from_slice(markers);
        b.extend(u16b(MARKER_SOD));
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn ppt_and_ppm_stored_and_misplaced() {
        let mut b = head();
        b.extend(tile_part(
            0,
            1,
            &segment(MARKER_PPT, &[0x00, 0xAA]),
            &[1, 2],
        ));
        b.extend(u16b(MARKER_EOC));
        let cs = parse(&b, &lim()).unwrap();
        assert_eq!(cs.tile_parts[0].ppt().len(), 1);
        assert_eq!(
            cs.tile_parts[0].ppt()[0],
            PptEntry {
                z: 0,
                data: vec![0xAA]
            }
        );

        // Zppt repeated across two tile-parts of the same tile.
        let mut b = head();
        b.extend(tile_part(0, 2, &segment(MARKER_PPT, &[5, 0xAA]), &[1, 2]));
        b.extend(tile_part(1, 2, &segment(MARKER_PPT, &[5, 0xBB]), &[3, 4]));
        b.extend(u16b(MARKER_EOC));
        assert!(parse(&b, &lim()).is_err());

        // PPM inside a tile-part header, and PPT inside the main header.
        let mut b = head();
        b.extend(tile_part(0, 1, &segment(MARKER_PPM, &[0, 0xAA]), &[1, 2]));
        b.extend(u16b(MARKER_EOC));
        assert!(parse(&b, &lim()).is_err());
        let mut b = u16b(MARKER_SOC);
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        b.extend(segment(MARKER_COD, &fixture_cod_payload(false)));
        b.extend(segment(MARKER_PPT, &[0, 0xAA]));
        b.extend(segment(MARKER_QCD, &fixture_qcd_payload()));
        b.extend(tile_part(0, 1, &[], &[1, 2]));
        b.extend(u16b(MARKER_EOC));
        assert!(parse(&b, &lim()).is_err());

        let mut b = u16b(MARKER_SOC);
        b.extend(segment(MARKER_SIZ, &fixture_siz_payload(1)));
        b.extend(segment(MARKER_PPM, &[0x00, 0xAA]));
        b.extend(segment(MARKER_COD, &fixture_cod_payload(false)));
        b.extend(segment(MARKER_QCD, &fixture_qcd_payload()));
        b.extend(tile_part(0, 1, &[], &[1, 2]));
        b.extend(u16b(MARKER_EOC));
        let cs = parse(&b, &lim()).unwrap();
        assert_eq!(
            cs.ppm,
            vec![PpmEntry {
                z: 0,
                data: vec![0xAA]
            }]
        );
    }

    #[test]
    fn tpsot_and_tnsot_consistency() {
        let mut b = head();
        b.extend(tile_part(0, 1, &[], &[1, 2]));
        b.extend(tile_part(1, 1, &[], &[3, 4])); // TPsot >= TNsot
        b.extend(u16b(MARKER_EOC));
        assert!(parse(&b, &lim()).is_err());

        let mut b = head();
        b.extend(tile_part(1, 2, &[], &[1, 2, 3])); // must start at TPsot 0
        b.extend(u16b(MARKER_EOC));
        assert!(parse(&b, &lim()).is_err());
    }

    #[test]
    fn truncated_without_eoc_is_tolerated() {
        let data = fixture_codestream();
        let cs = parse(&data[..data.len() - 3], &lim()).unwrap();
        assert!(cs.truncated && !cs.eoc);
        assert_eq!(cs.tile_parts.len(), 1);

        // Psot == 0 runs to the next SOT or EOC: none, so truncated.
        let mut b = head();
        b.extend(tile_part(0, 1, &[], &[1, 2, 3, 4]));
        let cs = parse(&b, &lim()).unwrap();
        assert!(cs.truncated && !cs.eoc);
    }

    #[test]
    fn golden_fixture() {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/gradient-16bit-97.j2k"
        ))
        .unwrap();
        let cs = parse(&data, &lim()).unwrap();
        assert_eq!((cs.siz.xsiz, cs.siz.ysiz), (128, 128));
        assert_eq!((cs.siz.tiles_wide, cs.siz.tiles_high), (1, 1));
        assert_eq!(cs.siz.components.len(), 3);
        for c in &cs.siz.components {
            assert!(c.precision == 16 && !c.signed && c.xrsiz == 1 && c.yrsiz == 1);
        }
        let cod = &cs.cod;
        assert!(!cod.uses_custom_precincts && cod.progression_order == 0);
        assert_eq!(
            (cod.number_of_layers, cod.multiple_component_transform),
            (1, 1)
        );
        let s = &cod.coding_style;
        assert_eq!(
            (
                s.num_decomp_levels,
                s.code_block_width_exp,
                s.code_block_height_exp
            ),
            (5, 6, 6)
        );
        assert_eq!((s.code_block_style, s.transformation), (0, 0));
        assert_eq!(cs.qcd.sqcd, 0x42);
        let q = &cs.qcd.quantization;
        assert_eq!((q.quantization_style, q.guard_bits), (2, 2));
        assert_eq!((q.exponents.len(), q.mantissas.len()), (16, 16));
    }

    #[test]
    fn warnings_capped() {
        let mut b = head();
        for _ in 0..200 {
            b.extend(segment(0xFF63, &[1, 2]));
        }
        b.extend(tile_part(0, 1, &[], &[1, 2]));
        b.extend(u16b(MARKER_EOC));
        let cs = parse(&b, &lim()).unwrap();
        assert_eq!(cs.warnings.len(), MAX_WARNINGS + 1);
        assert!(cs.warnings[MAX_WARNINGS].contains("100 more warnings not kept"));
    }
}
