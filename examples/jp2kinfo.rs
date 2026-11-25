//! Prints the marker segments of a JPEG 2000 codestream, raw or inside a JP2 file:
//! `cargo run --example jp2kinfo file.jp2` (or the file on stdin). Port of `cmd/jp2kinfo`.
#![allow(clippy::needless_range_loop)]
use jp2k::Limits;
use jp2k::fuzz::{
    self, Coc, Cod, Codestream, CodingStyle, Com, Plt, Poc, Qcc, Qcd, Quantization, Rgn, Tlm,
};
use std::io::Read;
use std::sync::atomic::AtomicBool;

fn main() {
    if let Err(e) = run() {
        eprintln!("jp2kinfo: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut data = Vec::new();
    match std::env::args().nth(1) {
        Some(p) => data = std::fs::read(p).map_err(|e| e.to_string())?,
        None => {
            std::io::stdin()
                .read_to_end(&mut data)
                .map_err(|e| e.to_string())?;
        }
    }
    let stream: &[u8] = if data.starts_with(b"\x00\x00\x00\x0cjP  ") {
        jp2c(&data)?
    } else {
        &data
    };
    let cs = fuzz::parse(stream, &Limits::default(), &AtomicBool::new(false))
        .map_err(|e| e.to_string())?;
    print_codestream(&cs);
    Ok(())
}

/// The payload of a JP2 file's contiguous codestream box.
fn jp2c(mut data: &[u8]) -> Result<&[u8], String> {
    while data.len() >= 8 {
        let mut n = u64::from(u32::from_be_bytes([data[0], data[1], data[2], data[3]]));
        let typ = String::from_utf8_lossy(&data[4..8]).to_string();
        let mut hdr = 8u64;
        match n {
            0 => n = data.len() as u64, // box runs to the end of the file
            1 => {
                // 64-bit length follows the type
                if data.len() < 16 {
                    return Err("truncated JP2 box header".into());
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[8..16]);
                n = u64::from_be_bytes(b);
                hdr = 16;
            }
            _ => {}
        }
        if n < hdr || n > data.len() as u64 {
            return Err(format!("JP2 box {typ:?} has bad length {n}"));
        }
        if typ == "jp2c" {
            return Ok(&data[hdr as usize..n as usize]);
        }
        data = &data[n as usize..];
    }
    Err("no jp2c box in JP2 file".into())
}

fn print_codestream(cs: &Codestream) {
    println!("SOC\n");
    println!("SIZ");
    let s = &cs.siz;
    println!("  Rsiz: {}", s.rsiz);
    println!(
        "  Image: {}x{}, offset ({},{})",
        s.xsiz - s.xosiz,
        s.ysiz - s.yosiz,
        s.xosiz,
        s.yosiz
    );
    println!(
        "  Tile: {}x{}, offset ({},{})",
        s.xtsiz, s.ytsiz, s.xtosiz, s.ytosiz
    );
    println!("  Tile grid: {}x{}", s.tiles_wide, s.tiles_high);
    println!("  Components: {}", s.csiz);
    for (i, c) in s.components.iter().enumerate() {
        println!(
            "    [{i}] precision={} signed={} XRsiz={} YRsiz={}",
            c.precision, c.signed, c.xrsiz, c.yrsiz
        );
    }
    println!();

    print_cod(&cs.cod);
    cs.coc.iter().for_each(print_coc);
    print_qcd(&cs.qcd);
    cs.qcc.iter().for_each(print_qcc);
    cs.rgn.iter().for_each(print_rgn);
    cs.poc.iter().for_each(print_poc);
    cs.com.iter().for_each(print_com);
    cs.tlm.iter().for_each(print_tlm);

    for (i, tp) in cs.tile_parts.iter().enumerate() {
        println!("SOT [tile-part {i}]");
        println!(
            "  Isot={} Psot={} TPsot={} TNsot={}",
            tp.sot.isot, tp.sot.psot, tp.sot.tpsot, tp.sot.tnsot
        );
        println!();
        if let Some(c) = tp.cod() {
            print_cod(c);
        }
        tp.coc().iter().for_each(print_coc);
        if let Some(q) = tp.qcd() {
            print_qcd(q);
        }
        tp.qcc().iter().for_each(print_qcc);
        tp.rgn().iter().for_each(print_rgn);
        tp.poc().iter().for_each(print_poc);
        tp.com().iter().for_each(print_com);
        tp.plt().iter().for_each(print_plt);
        println!("SOD");
        println!("  data offset={} length={}", tp.data_offset, tp.data_length);
        println!();
    }
    if cs.eoc {
        println!("EOC");
    }
    for w in &cs.warnings {
        eprintln!("warning: {w}");
    }
}

fn print_cod(c: &Cod) {
    println!("COD");
    println!(
        "  Scod=0x{:02X} customPrecincts={} sop={} eph={}",
        c.scod, c.uses_custom_precincts, c.uses_sop, c.uses_eph
    );
    println!(
        "  progressionOrder={} layers={} mct={}",
        c.progression_order, c.number_of_layers, c.multiple_component_transform
    );
    print_coding_style(&c.coding_style);
    println!();
}

fn print_coc(c: &Coc) {
    println!("COC");
    println!(
        "  component={} Scoc=0x{:02X} customPrecincts={}",
        c.component_index, c.scoc, c.uses_custom_precincts
    );
    print_coding_style(&c.coding_style);
    println!();
}

fn print_coding_style(cs: &CodingStyle) {
    println!(
        "  decompositionLevels={} codeBlock={}x{} style=0x{:02X} transform={}",
        cs.num_decomp_levels,
        1u32 << cs.code_block_width_exp,
        1u32 << cs.code_block_height_exp,
        cs.code_block_style,
        cs.transformation
    );
    for (i, p) in cs.precinct_sizes.iter().flatten().enumerate() {
        println!("    precinct[{i}] {}x{}", 1u32 << p.ppx, 1u32 << p.ppy);
    }
}

fn print_qcd(q: &Qcd) {
    println!("QCD");
    println!("  Sqcd=0x{:02X}", q.sqcd);
    print_quantization(&q.quantization);
    println!();
}

fn print_qcc(q: &Qcc) {
    println!("QCC");
    println!("  component={} Sqcc=0x{:02X}", q.component_index, q.sqcc);
    print_quantization(&q.quantization);
    println!();
}

fn print_quantization(q: &Quantization) {
    println!(
        "  style={} guardBits={} bands={}",
        q.quantization_style,
        q.guard_bits,
        q.exponents.len()
    );
    for i in 0..q.exponents.len() {
        println!(
            "    band[{i}] exponent={} mantissa={}",
            q.exponents[i], q.mantissas[i]
        );
    }
}

fn print_rgn(r: &Rgn) {
    println!("RGN");
    println!(
        "  component={} Srgn={} SPrgn={}",
        r.component_index, r.srgn, r.sprgn
    );
    println!();
}

fn print_poc(p: &Poc) {
    println!("POC");
    for (i, c) in p.changes.iter().enumerate() {
        println!(
            "  [{i}] RS={} CS={} LYE={} RE={} CE={} P={}",
            c.rspoc, c.cspoc, c.lyepoc, c.repoc, c.cepoc, c.ppoc
        );
    }
    println!();
}

fn print_com(c: &Com) {
    println!("COM");
    println!("  Rcom={} length={}", c.rcom, c.data.len());
    println!();
}

fn print_tlm(t: &Tlm) {
    println!("TLM");
    println!(
        "  Ztlm={} Stlm=0x{:02X} invalid={} entries={}",
        t.ztlm,
        t.stlm,
        t.invalid,
        t.entries.len()
    );
    for (i, e) in t.entries.iter().enumerate() {
        println!("    [{i}] Ttlm={} Ptlm={}", e.ttlm, e.ptlm);
    }
    println!();
}

fn print_plt(p: &Plt) {
    println!("PLT");
    println!("  Zplt={} packets={}", p.zplt, p.packet_lengths.len());
    println!();
}
