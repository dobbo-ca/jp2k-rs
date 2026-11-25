use super::markers::*;

/// Resolves the effective quantization for a component, honoring T.800 A.6.4/A.6.5: a tile-part
/// QCC/QCD overrides the main header's, in the same precedence order as
/// [`effective_coding_style`]'s COC/COD resolution.
pub fn quant_for(cs: &Codestream, tile: u16, comp: usize) -> &Quantization {
    let mut tile_qcd: Option<&Qcd> = None;
    let mut tile_qcc: Vec<&Qcc> = Vec::new();
    for &i in cs.tile_part_index(tile) {
        let tp = &cs.tile_parts[i as usize];
        if let Some(q) = tp.qcd() {
            tile_qcd = Some(q);
        }
        tile_qcc.extend(tp.qcc().iter());
    }
    for qcc in tile_qcc {
        if qcc.component_index as usize == comp {
            return &qcc.quantization;
        }
    }
    if let Some(q) = tile_qcd {
        return &q.quantization;
    }
    for qcc in &cs.qcc {
        if qcc.component_index as usize == comp {
            return &qcc.quantization;
        }
    }
    &cs.qcd.quantization
}

/// The COD/COC found in `tile`'s tile-part headers, and whether any has a POC. T.800 A.6.1/A.6.2
/// only allow COD/COC in a tile's first tile-part, but nothing here depends on that.
pub fn tile_overrides(cs: &Codestream, tile: u16) -> (Option<&Cod>, Vec<&Coc>, bool) {
    let mut tile_cod = None;
    let mut tile_coc = Vec::new();
    let mut has_poc = false;
    for &i in cs.tile_part_index(tile) {
        let tp = &cs.tile_parts[i as usize];
        if let Some(c) = tp.cod() {
            tile_cod = Some(c);
        }
        tile_coc.extend(tp.coc().iter());
        if !tp.poc().is_empty() {
            has_poc = true;
        }
    }
    (tile_cod, tile_coc, has_poc)
}

/// Resolves T.800 A.6's precedence: a per-component COC beats a per-tile/main COD, and a
/// tile-part override beats the main header (tile COC > tile COD > main COC > main COD).
pub fn effective_coding_style<'a>(
    cs: &'a Codestream,
    tile_cod: Option<&'a Cod>,
    tile_coc: &[&'a Coc],
    comp: usize,
) -> &'a CodingStyle {
    for c in tile_coc {
        if c.component_index as usize == comp {
            return &c.coding_style;
        }
    }
    if let Some(c) = tile_cod {
        return &c.coding_style;
    }
    for c in &cs.coc {
        if c.component_index as usize == comp {
            return &c.coding_style;
        }
    }
    &cs.cod.coding_style
}

/// Resolves T.800 A.6.1's precedence for the MCT bit: a COD in the tile's first tile-part
/// overrides the main header's COD wholesale.
pub fn effective_mct(cs: &Codestream, tile_cod: Option<&Cod>) -> u8 {
    match tile_cod {
        Some(c) => c.multiple_component_transform,
        None => cs.cod.multiple_component_transform,
    }
}
