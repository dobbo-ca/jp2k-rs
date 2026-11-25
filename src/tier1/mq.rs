//! MQ arithmetic decoder, T.800 Annex C.

/// One row of T.800 Table C-2: the probability estimate for a context state plus the state
/// transitions on MPS/LPS exchange: (qe, nmps, nlps, switch).
const QE_TABLE: [(u32, u8, u8, u8); 47] = [
    (0x5601, 1, 1, 1),
    (0x3401, 2, 6, 0),
    (0x1801, 3, 9, 0),
    (0x0ac1, 4, 12, 0),
    (0x0521, 5, 29, 0),
    (0x0221, 38, 33, 0),
    (0x5601, 7, 6, 1),
    (0x5401, 8, 14, 0),
    (0x4801, 9, 14, 0),
    (0x3801, 10, 14, 0),
    (0x3001, 11, 17, 0),
    (0x2401, 12, 18, 0),
    (0x1c01, 13, 20, 0),
    (0x1601, 29, 21, 0),
    (0x5601, 15, 14, 1),
    (0x5401, 16, 14, 0),
    (0x5101, 17, 15, 0),
    (0x4801, 18, 16, 0),
    (0x3801, 19, 17, 0),
    (0x3401, 20, 18, 0),
    (0x3001, 21, 19, 0),
    (0x2801, 22, 19, 0),
    (0x2401, 23, 20, 0),
    (0x2201, 24, 21, 0),
    (0x1c01, 25, 22, 0),
    (0x1801, 26, 23, 0),
    (0x1601, 27, 24, 0),
    (0x1401, 28, 25, 0),
    (0x1201, 29, 26, 0),
    (0x1101, 30, 27, 0),
    (0x0ac1, 31, 28, 0),
    (0x09c1, 32, 29, 0),
    (0x08a1, 33, 30, 0),
    (0x0521, 34, 31, 0),
    (0x0441, 35, 32, 0),
    (0x02a1, 36, 33, 0),
    (0x0221, 37, 34, 0),
    (0x0141, 38, 35, 0),
    (0x0111, 39, 36, 0),
    (0x0085, 40, 37, 0),
    (0x0049, 41, 38, 0),
    (0x0025, 42, 39, 0),
    (0x0015, 43, 40, 0),
    (0x0009, 44, 41, 0),
    (0x0005, 45, 42, 0),
    (0x0001, 45, 43, 0),
    (0x5601, 46, 46, 0),
];

/// `QE_TABLE` indexed by a packed context byte (`state << 1 | mps`), one word per state: qe in
/// bits 0..31, the packed successor after an MPS in 32..39, after an LPS (MPS switch applied) in
/// 40..47, the state byte itself in 48..55. A decision then needs one load.
pub const MQ_W: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 47 {
        let (qe, nmps, nlps, sw) = QE_TABLE[i];
        let mut mps = 0u8;
        while mps < 2 {
            let s = (i << 1) | mps as usize;
            let n0 = (nmps << 1) | mps;
            let n1 = (nlps << 1) | (mps ^ sw);
            t[s] = qe as u64 | (n0 as u64) << 32 | (n1 as u64) << 40 | (s as u64) << 48;
            mps += 1;
        }
        i += 1;
    }
    t
};

/// The MQ decoder over one code-block segment's bytes. T.800's 32-bit C register (chigh in bits
/// 16..31, clow below) is the top half of `c`; the bottom half buffers bytes read ahead, `ct`
/// bits of them counting down from where BYTEIN would put the next byte.
pub struct MqDecoder<'a> {
    data: &'a [u8],
    /// Last byte read.
    bp: usize,
    pub(super) c: u64,
    pub(super) a: u32,
    pub(super) ct: u32,
}

impl<'a> MqDecoder<'a> {
    /// Reads `data[i]`, returning 0xFF past the end of the segment. T.800 C.3.4 requires the
    /// decoder to keep feeding 1-bits once the compressed data is exhausted.
    fn at(&self, i: usize) -> u64 {
        match self.data.get(i) {
            Some(&b) => b as u64,
            None => 0xFF,
        }
    }

    /// INITDEC (T.800 C.3.5).
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = MqDecoder {
            data,
            bp: 0,
            c: 0,
            a: 0,
            ct: 0,
        };
        d.c = d.at(0) << 48;
        d.byte_in();
        d.c <<= 7;
        d.ct -= 7;
        d.fill();
        d.a = 0x8000;
        d
    }

    /// BYTEIN, T.800 C.3.4, including the 0xFF > 0x8F stuffing rule, appending below the `ct`
    /// bits already buffered.
    fn byte_in(&mut self) {
        if self.at(self.bp) == 0xFF {
            if self.at(self.bp + 1) > 0x8F {
                self.c = self.c.wrapping_add(0xFF << (40 - self.ct));
                self.ct += 8;
            } else {
                self.bp += 1;
                self.c = self.c.wrapping_add(self.at(self.bp) << (41 - self.ct));
                self.ct += 7;
            }
        } else {
            self.bp += 1;
            self.c = self.c.wrapping_add(self.at(self.bp) << (40 - self.ct));
            self.ct += 8;
        }
    }

    /// Reads ahead until more than 32 bits are buffered, enough for two renormalizations.
    fn fill(&mut self) {
        while self.ct <= 32 {
            self.byte_in();
        }
    }

    /// `fill` on register copies of `c` and `ct`, for the coding passes that hold them in locals.
    #[inline(never)]
    pub(super) fn fill_regs(&mut self, c: u64, ct: u32) -> (u64, u32) {
        self.c = c;
        self.ct = ct;
        self.fill();
        (self.c, self.ct)
    }

    /// DECODE, T.800 C.3.2, with the C.3.3 renormalization.
    #[inline]
    pub fn decode_bit(&mut self, contexts: &mut [u64; 32], cx: usize) -> u32 {
        let w = contexts[cx & 31];
        let qe = w as u32;
        let mut bit = ((w >> 48) & 1) as u32;
        self.a = self.a.wrapping_sub(qe);
        let hi = (qe as u64) << 48;
        if self.c < hi || self.a & 0x8000 == 0 {
            // Conditional exchange: which sub-interval is the LPS depends on their sizes
            // (C.3.2 Figures C.4, C.5).
            let mut lps = self.a < qe;
            if self.c < hi {
                lps = !lps;
                self.a = qe;
            } else {
                self.c = self.c.wrapping_sub(hi);
            }
            if lps {
                bit ^= 1;
                contexts[cx & 31] = MQ_W[((w >> 40) & 127) as usize];
            } else {
                contexts[cx & 31] = MQ_W[((w >> 32) & 127) as usize];
            }
            let n = (self.a.leading_zeros().wrapping_sub(16)) & 15;
            if n > self.ct {
                self.fill();
            }
            self.a <<= n;
            self.c <<= n;
            self.ct -= n;
        } else {
            self.c = self.c.wrapping_sub(hi);
        }
        bit
    }
}

/// The raw (selective-bypass) bit reader used by SPP/MRP coding passes once a code-block's
/// style enables bypass and the current bit-plane is deep enough (T.800 Annex C.13 / D.4). It
/// applies the same FF-stuffing rule as the MQ decoder, but each bit is read straight from the
/// byte stream.
pub struct RawDecoder<'a> {
    data: &'a [u8],
    bp: usize,
    c: u32,
    ct: i32,
}

impl<'a> RawDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        RawDecoder {
            data,
            bp: 0,
            c: 0,
            ct: 0,
        }
    }

    /// 0xFF past the segment end, so a raw segment that runs out of real bytes keeps feeding the
    /// artificial stuffed 0xFF sentinel.
    fn at(&self, i: usize) -> u32 {
        match self.data.get(i) {
            Some(&b) => b as u32,
            None => 0xFF,
        }
    }

    pub fn read_bit(&mut self) -> u32 {
        if self.ct == 0 {
            if self.c == 0xFF {
                if self.at(self.bp) > 0x8F {
                    self.c = 0xFF;
                    self.ct = 8;
                } else {
                    self.c = self.at(self.bp);
                    self.bp += 1;
                    self.ct = 7;
                }
            } else {
                self.c = self.at(self.bp);
                self.bp += 1;
                self.ct = 8;
            }
        }
        self.ct -= 1;
        (self.c >> self.ct) & 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transcribed T.800 Table C-2 against structural invariants and a few rows verbatim.
    #[test]
    fn qe_table_integrity() {
        assert_eq!(QE_TABLE.len(), 47);
        for (i, &(_, nmps, nlps, sw)) in QE_TABLE.iter().enumerate() {
            assert!(nmps <= 46, "row {i}: nmps {nmps} out of range");
            assert!(nlps <= 46, "row {i}: nlps {nlps} out of range");
            assert!(sw <= 1, "row {i}: switch flag {sw} not 0/1");
        }
        assert_eq!(QE_TABLE[0], (0x5601, 1, 1, 1));
        assert_eq!(QE_TABLE[1], (0x3401, 2, 6, 0));
        assert_eq!(QE_TABLE[13], (0x1601, 29, 21, 0));
        assert_eq!(QE_TABLE[45], (0x0001, 45, 43, 0)); // self-loops on NMPS
        assert_eq!(QE_TABLE[46], (0x5601, 46, 46, 0)); // self-loops on both
    }

    /// The packed word table agrees with the table it is built from.
    #[test]
    fn packed_word_matches_table() {
        for (i, &(qe, nmps, nlps, sw)) in QE_TABLE.iter().enumerate() {
            for mps in 0..2usize {
                let w = MQ_W[(i << 1) | mps];
                assert_eq!(w as u32, qe);
                assert_eq!((w >> 32) & 0xFF, ((nmps as u64) << 1) | mps as u64);
                assert_eq!(
                    (w >> 40) & 0xFF,
                    ((nlps as u64) << 1) | (mps as u64 ^ sw as u64)
                );
                assert_eq!((w >> 48) & 0xFF, ((i << 1) | mps) as u64);
            }
        }
    }

    /// Once the segment runs out, T.800 C.3.4 requires the decoder to keep decoding (feeding
    /// 1-bits) rather than panic or read out of bounds.
    #[test]
    fn decoder_no_panic_past_end() {
        let mut mq = MqDecoder::new(&[]);
        let mut ctxs = [MQ_W[0]; 32];
        for i in 0..10000 {
            mq.decode_bit(&mut ctxs, i % 19);
        }
    }

    /// A lone 0xFF followed by a byte > 0x8F is the marker pattern and must not advance `bp`.
    #[test]
    fn decoder_ff_stuffing() {
        let mut mq = MqDecoder::new(&[0xFF, 0x90, 0x00]);
        let before = mq.bp;
        mq.byte_in();
        assert_eq!(mq.bp, before);
    }

    /// 10k decisions from pseudo-random bytes must never panic or loop forever.
    #[test]
    fn decoder_random_bytes() {
        let mut x: u64 = 0x2545F4914F6CDD1D;
        for len in [0usize, 1, 2, 3, 7, 64, 1000] {
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect();
            let mut mq = MqDecoder::new(&data);
            let mut ctxs = [MQ_W[0]; 32];
            for i in 0..10000 {
                mq.decode_bit(&mut ctxs, i % 19);
            }
        }
    }

    #[test]
    fn raw_decoder_stuffing() {
        // After 0xFF the next byte carries 7 bits.
        let mut r = RawDecoder::new(&[0xFF, 0x7F]);
        let bits: Vec<u32> = (0..15).map(|_| r.read_bit()).collect();
        assert_eq!(&bits[..8], &[1; 8]);
        assert_eq!(&bits[8..], &[1; 7]);
        // Past the end it keeps reading 1-bits.
        let mut r = RawDecoder::new(&[]);
        assert!((0..64).all(|_| r.read_bit() == 1));
    }
}
