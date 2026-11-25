//! The T.800 B.10.2 tag tree (OpenJPEG tgt.c), used for both code-block inclusion and
//! zero-bitplane signalling. State persists across packets: it lives on the precinct for the
//! life of the tile.

use crate::error::{Error, Result};

/// Where a tag tree reads its bits from: the packet bit reader, or a plain queue in the tests.
pub trait BitSource {
    fn read_bit(&mut self) -> Result<u32>;
}

#[derive(Clone, Copy, Debug)]
struct Node {
    value: i32,
    low: i32,
    /// -1 for the root.
    parent: i32,
}

#[derive(Debug, Default)]
pub struct TagTree {
    nodes: Vec<Node>,
    /// 0 when w or h is 0: an empty precinct has no code-blocks.
    num_leaves: usize,
}

/// More levels than any tree `node_count` accepts can have (its nodes fit an i32).
const MAX_LEVELS: usize = 64;

impl TagTree {
    /// Builds the node/parent topology for a `w` x `h` leaf grid (see `reset`).
    pub fn new(w: usize, h: usize) -> Result<TagTree> {
        let mut t = TagTree::default();
        t.reset(w, h)?;
        Ok(t)
    }

    /// The heap its nodes hold.
    pub(crate) fn heap(&self) -> usize {
        crate::limits::held_by(&self.nodes)
    }

    /// Rebuilds the tree for a `w` x `h` leaf grid in its own nodes' memory, all values unknown.
    /// The parent-linking loop is a direct port of tgt.c's index arithmetic. The caller has
    /// charged `size_for(w, h)` to its budget; this checks the grid's arithmetic and reserves
    /// the nodes fallibly (allocating only when the tree had fewer).
    pub fn reset(&mut self, w: usize, h: usize) -> Result<()> {
        self.clear();
        if w == 0 || h == 0 {
            return Ok(());
        }
        let numnodes = Self::node_count(w, h)?;
        let (mut nplh, mut nplv) = ([0usize; MAX_LEVELS], [0usize; MAX_LEVELS]);
        nplh[0] = w;
        nplv[0] = h;
        let mut nl = 1;
        loop {
            let (lh, lv) = (nplh[nl - 1], nplv[nl - 1]);
            if nl == MAX_LEVELS {
                return Err(limit!("tag tree grid {w}x{h} is too large"));
            }
            nplh[nl] = lh.div_ceil(2);
            nplv[nl] = lv.div_ceil(2);
            nl += 1;
            if lh * lv <= 1 {
                break;
            }
        }
        let numlvls = nl - 1;

        let nodes = &mut self.nodes;
        nodes
            .try_reserve_exact(numnodes)
            .map_err(|_| Error::OutOfMemory)?;
        nodes.resize(
            numnodes,
            Node {
                value: 999,
                low: 0,
                parent: -1,
            },
        );

        let mut node_idx = 0usize;
        let mut parent_idx = w * h;
        let mut parent0_idx = parent_idx;
        for i in 0..numlvls.saturating_sub(1) {
            for j in 0..nplv[i] {
                let mut k = nplh[i] as isize;
                loop {
                    k -= 1;
                    if k < 0 {
                        break;
                    }
                    nodes[node_idx].parent = parent_idx as i32;
                    node_idx += 1;
                    k -= 1;
                    if k >= 0 {
                        nodes[node_idx].parent = parent_idx as i32;
                        node_idx += 1;
                    }
                    parent_idx += 1;
                }
                if j & 1 == 1 || j == nplv[i] - 1 {
                    parent0_idx = parent_idx;
                } else {
                    parent_idx = parent0_idx;
                    parent0_idx += nplh[i];
                }
            }
        }
        self.num_leaves = w * h;
        Ok(())
    }

    /// Empties the tree, keeping its memory.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.num_leaves = 0;
    }

    /// Whether the tree has no leaves (it is cleared, or for an empty grid).
    pub fn is_empty(&self) -> bool {
        self.num_leaves == 0
    }

    /// The nodes of a tree for a `w` x `h` grid (0 for an empty grid), or a limit error when the
    /// count overflows or its parent links would not fit an i32.
    fn node_count(w: usize, h: usize) -> Result<usize> {
        let (mut lw, mut lh, mut nodes) = (w, h, 0usize);
        while lw > 0 && lh > 0 {
            let n = lw
                .checked_mul(lh)
                .ok_or_else(|| limit!("tag tree grid {w}x{h} is too large"))?;
            nodes = nodes
                .checked_add(n)
                .ok_or_else(|| limit!("tag tree grid {w}x{h} is too large"))?;
            if n <= 1 {
                break;
            }
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
        }
        if nodes > i32::MAX as usize {
            return Err(limit!("tag tree grid {w}x{h} is too large"));
        }
        Ok(nodes)
    }

    /// Heap bytes of the nodes of a tree for a `w` x `h` grid, counted as `new` does, or a limit
    /// error when the grid is too large to count.
    pub fn size_for(w: usize, h: usize) -> Result<usize> {
        Self::node_count(w, h)?
            .checked_add(8)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Node>()))
            .ok_or_else(|| limit!("tag tree grid {w}x{h} is too large"))
    }

    /// Runs one threshold step of the tag-tree recurrence for `leaf`, reading bits until the
    /// leaf's value is resolved against `threshold` or is known to be >= it. Inclusion uses
    /// `threshold = layer + 1`; zero-bitplane decoding walks once at its final threshold.
    pub fn decode<B: BitSource>(
        &mut self,
        leaf: usize,
        threshold: i32,
        br: &mut B,
    ) -> Result<bool> {
        if leaf >= self.num_leaves {
            return Err(invalid!(
                "tag tree leaf {leaf} out of range ({} leaves)",
                self.num_leaves
            ));
        }
        // A tree of up to 2^20 leaves is under 22 levels deep.
        let mut stack = [0usize; 64];
        let mut sp = 0usize;
        let mut node = leaf;
        while self.nodes[node].parent != -1 {
            stack[sp] = node;
            sp += 1;
            node = self.nodes[node].parent as usize;
        }
        let mut low = 0i32;
        loop {
            let n = &mut self.nodes[node];
            if low > n.low {
                n.low = low;
            } else {
                low = n.low;
            }
            while low < threshold && low < n.value {
                let bit = br.read_bit()?;
                if bit == 1 {
                    n.value = low;
                } else {
                    low += 1;
                }
            }
            n.low = low;
            if sp == 0 {
                break;
            }
            sp -= 1;
            node = stack[sp];
        }
        Ok(self.nodes[node].value < threshold)
    }

    pub fn value(&self, leaf: usize) -> i32 {
        self.nodes[leaf].value
    }

    /// Skips leaves already proven absent by the root.
    pub fn excluded(&self, threshold: i32) -> bool {
        match self.nodes.last() {
            None => true,
            Some(root) => root.value == 999 && root.low >= threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial sequential bit source, separate from the real packet bit reader (which also
    /// handles FF stuffing).
    struct BitQueue {
        bits: Vec<u32>,
        pos: usize,
    }

    impl BitSource for BitQueue {
        fn read_bit(&mut self) -> Result<u32> {
            let b = self.bits[self.pos];
            self.pos += 1;
            Ok(b)
        }
    }

    /// Each node's true value bottom-up (min over its leaf subtree), straight from the topology
    /// `TagTree::new` builds, so the test encoder is driven by the same parent links production
    /// code uses.
    fn tree_true_values(t: &TagTree, leaf_values: &[i32]) -> Vec<i32> {
        let mut vals = vec![999; t.nodes.len()];
        vals[..t.num_leaves].copy_from_slice(&leaf_values[..t.num_leaves]);
        for i in 0..t.nodes.len() {
            let p = t.nodes[i].parent;
            if p == -1 {
                continue;
            }
            if vals[i] < vals[p as usize] {
                vals[p as usize] = vals[i];
            }
        }
        vals
    }

    impl TagTree {
        /// The write-side dual of `decode`: same traversal and low/value bookkeeping, but it
        /// knows each node's true value up front and emits the bit the real encoder would have
        /// written, instead of reading one (T.800 B.10.2).
        fn encode_leaf(
            &mut self,
            leaf: usize,
            threshold: i32,
            true_val: &[i32],
            out: &mut Vec<u32>,
        ) -> bool {
            let mut stack = Vec::new();
            let mut node = leaf;
            while self.nodes[node].parent != -1 {
                stack.push(node);
                node = self.nodes[node].parent as usize;
            }
            let mut low = 0;
            loop {
                let n = &mut self.nodes[node];
                if low > n.low {
                    n.low = low;
                } else {
                    low = n.low;
                }
                while low < threshold && low < n.value {
                    if low == true_val[node] {
                        out.push(1);
                        n.value = low;
                    } else {
                        out.push(0);
                        low += 1;
                    }
                }
                n.low = low;
                match stack.pop() {
                    None => break,
                    Some(x) => node = x,
                }
            }
            self.nodes[node].value < threshold
        }
    }

    /// A 3x2 tree with known leaf values: every leaf's inclusion bit (threshold = value + 1) is
    /// encoded into one bitstream, which a fresh tree then decodes.
    #[test]
    fn inclusion_round_trip() {
        let (w, h) = (3, 2);
        let leaf_values = [2, 0, 1, 3, 0, 2];
        let mut enc = TagTree::new(w, h).unwrap();
        let true_val = tree_true_values(&enc, &leaf_values);
        let mut bits = Vec::new();
        for leaf in 0..w * h {
            assert!(enc.encode_leaf(leaf, leaf_values[leaf] + 1, &true_val, &mut bits));
        }
        let mut dec = TagTree::new(w, h).unwrap();
        let mut q = BitQueue {
            bits: bits.clone(),
            pos: 0,
        };
        for leaf in 0..w * h {
            assert!(
                dec.decode(leaf, leaf_values[leaf] + 1, &mut q).unwrap(),
                "leaf {leaf}"
            );
        }
        assert_eq!(q.pos, bits.len());
    }

    #[test]
    fn below_threshold() {
        let (w, h) = (4, 4);
        let mut leaf_values: Vec<i32> = (0..w * h).map(|i| (i % 5) as i32).collect();
        leaf_values[6] = 5;
        let mut enc = TagTree::new(w, h).unwrap();
        let true_val = tree_true_values(&enc, &leaf_values);
        let mut bits = Vec::new();
        assert!(!enc.encode_leaf(6, 2, &true_val, &mut bits));
        let mut dec = TagTree::new(w, h).unwrap();
        let mut q = BitQueue {
            bits: bits.clone(),
            pos: 0,
        };
        assert!(!dec.decode(6, 2, &mut q).unwrap());
        assert_eq!(q.pos, bits.len());
    }

    /// The zero-bitplane usage pattern (repeated decode calls with threshold 0, 1, 2, ... until
    /// true) resolves to the encoded value while consuming the same bits.
    #[test]
    fn zero_bitplane_loop() {
        let (w, h) = (3, 2);
        let leaf_values = [2, 0, 1, 3, 0, 2];
        let leaf = 3;
        let mut enc = TagTree::new(w, h).unwrap();
        let true_val = tree_true_values(&enc, &leaf_values);
        let mut bits = Vec::new();
        enc.encode_leaf(leaf, leaf_values[leaf] + 1, &true_val, &mut bits);
        let mut dec = TagTree::new(w, h).unwrap();
        let mut q = BitQueue {
            bits: bits.clone(),
            pos: 0,
        };
        let mut i = 0;
        while !dec.decode(leaf, i, &mut q).unwrap() {
            i += 1;
            assert!(i <= 74, "zero-bitplane loop did not resolve");
        }
        // The loop's first check is decode(leaf, 0), which never resolves, so it overshoots.
        assert_eq!(i, leaf_values[leaf] + 1);
        assert_eq!(q.pos, bits.len());
    }

    #[test]
    fn empty_grid_and_leaf_out_of_range() {
        let mut tree = TagTree::new(0, 3).unwrap();
        assert_eq!(tree.num_leaves, 0);
        let mut q = BitQueue {
            bits: vec![0; 4],
            pos: 0,
        };
        assert!(tree.decode(0, 1, &mut q).is_err());
        let mut tree = TagTree::new(2, 2).unwrap();
        assert!(tree.decode(4, 1, &mut q).is_err());
    }

    /// The excluded shortcut must resolve every leaf, and consume the same bits, as a full walk.
    #[test]
    fn excluded_shortcut_matches_full_walk() {
        let (w, h, layers) = (8usize, 7usize, 6i32);
        let n = w * h;
        let leaf_values: Vec<i32> = (0..n as i32).map(|i| (i * 5 + 3) % (layers + 2)).collect();
        let mut enc = TagTree::new(w, h).unwrap();
        let true_val = tree_true_values(&enc, &leaf_values);
        let mut bits = Vec::new();
        let mut ever_enc = vec![false; n];
        for layer in 0..layers {
            for leaf in 0..n {
                if ever_enc[leaf] {
                    continue;
                }
                if enc.encode_leaf(leaf, layer + 1, &true_val, &mut bits) {
                    ever_enc[leaf] = true;
                }
            }
        }
        let mut full = TagTree::new(w, h).unwrap();
        let mut short = TagTree::new(w, h).unwrap();
        let mut q_full = BitQueue {
            bits: bits.clone(),
            pos: 0,
        };
        let mut q_short = BitQueue {
            bits: bits.clone(),
            pos: 0,
        };
        let mut ever_full = vec![false; n];
        let mut ever_short = vec![false; n];
        for layer in 0..layers {
            let threshold = layer + 1;
            for leaf in 0..n {
                assert_eq!(
                    ever_full[leaf], ever_short[leaf],
                    "layer {layer} leaf {leaf}"
                );
                if ever_full[leaf] {
                    continue;
                }
                let got_full = full.decode(leaf, threshold, &mut q_full).unwrap();
                ever_full[leaf] = got_full;
                let got_short = if short.excluded(threshold) {
                    false
                } else {
                    short.decode(leaf, threshold, &mut q_short).unwrap()
                };
                ever_short[leaf] = got_short;
                assert_eq!(got_full, got_short, "layer {layer} leaf {leaf}");
            }
            assert_eq!(q_full.pos, q_short.pos, "layer {layer}");
        }
        assert!(q_full.pos == bits.len() && q_short.pos == bits.len());
    }

    /// `size_for` charges every node `new` allocates, on narrow grids too.
    #[test]
    fn size_for_covers_all_nodes() {
        for (w, h) in [
            (1, 1),
            (1, 4096),
            (4096, 1),
            (3, 5),
            (7, 7),
            (64, 33),
            (1, 2),
        ] {
            let t = TagTree::new(w, h).unwrap();
            assert!(
                TagTree::size_for(w, h).unwrap() >= t.nodes.len() * std::mem::size_of::<Node>(),
                "{w}x{h}"
            );
        }
    }

    /// A grid whose node count overflows is an error, not an allocation or a panic.
    #[test]
    fn huge_grids_are_rejected() {
        assert!(TagTree::new(1 << 20, 1 << 20).is_err());
        assert!(TagTree::new(usize::MAX, 2).is_err());
        assert!(TagTree::size_for(usize::MAX, usize::MAX).is_err());
        assert!(TagTree::size_for(1 << 20, 1 << 20).is_err());
    }
}
