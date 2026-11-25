package tier2

import "fmt"

// tagTree implements the T.800 B.10.2 tag tree (OpenJPEG tgt.c), used for
// both code-block inclusion and zero-bitplane signalling. State is
// persistent across packets: it lives on the precinct for the life of the
// tile, not per packet.
type tagTreeNode struct {
	value  int32
	low    int32
	parent int // -1 for the root
}

type tagTree struct {
	nodes     []tagTreeNode
	numLeaves int // 0 when w or h is 0: an empty precinct has no code-blocks
}

// newTagTree builds the node/parent topology for a w x h leaf grid. The
// parent-linking loop is a direct port of tgt.c's index arithmetic: a
// "parent = (x>>1, y>>1)" shortcut only matches it if levels are laid out
// identically, and the odd-row reuse clause is easy to get backwards.
func newTagTree(w, h int) *tagTree {
	if w <= 0 || h <= 0 {
		return &tagTree{}
	}

	// Stack-backed: region decodes build thousands of small trees.
	var hb, vb [34]int
	nplh := append(hb[:0], w)
	nplv := append(vb[:0], h)
	numnodes := 0
	for {
		n := nplh[len(nplh)-1] * nplv[len(nplv)-1]
		nplh = append(nplh, (nplh[len(nplh)-1]+1)/2)
		nplv = append(nplv, (nplv[len(nplv)-1]+1)/2)
		numnodes += n
		if n <= 1 {
			break
		}
	}
	numlvls := len(nplh) - 1

	nodes := make([]tagTreeNode, numnodes)
	for i := range nodes {
		nodes[i] = tagTreeNode{value: 999, low: 0, parent: -1}
	}

	nodeIdx := 0
	parentIdx := w * h
	parent0Idx := parentIdx
	for i := 0; i < numlvls-1; i++ {
		for j := 0; j < nplv[i]; j++ {
			k := nplh[i]
			for {
				k--
				if k < 0 {
					break
				}
				nodes[nodeIdx].parent = parentIdx
				nodeIdx++
				k--
				if k >= 0 {
					nodes[nodeIdx].parent = parentIdx
					nodeIdx++
				}
				parentIdx++
			}
			if j&1 == 1 || j == nplv[i]-1 {
				parent0Idx = parentIdx
			} else {
				parentIdx = parent0Idx
				parent0Idx += nplh[i]
			}
		}
	}

	return &tagTree{nodes: nodes, numLeaves: w * h}
}

// decode runs one threshold step of the tag-tree recurrence for leaf,
// reading bits via readBit until the leaf's value is resolved against
// threshold or is known to be >= it. Inclusion uses threshold = layer+1;
// zero-bitplane decoding calls this in a loop with increasing threshold
// until it returns true. low/value/parent state persists in t.nodes across
// calls, which is what makes multi-layer, multi-call decoding correct.
func (t *tagTree) decode(leaf int, threshold int32, readBit func() (int, error)) (bool, error) {
	if leaf < 0 || leaf >= t.numLeaves {
		return false, fmt.Errorf("jp2k: tag tree leaf %d out of range (%d leaves)", leaf, t.numLeaves)
	}

	// Deep enough for any tree maxCodeBlocksPerPrecinct allows, so the
	// walk stays off the heap; append still copes with more.
	var buf [24]int
	stack := buf[:0]
	node := leaf
	for t.nodes[node].parent != -1 {
		stack = append(stack, node)
		node = t.nodes[node].parent
	}

	low := int32(0)
	for {
		n := &t.nodes[node]
		if low > n.low {
			n.low = low
		} else {
			low = n.low
		}
		for low < threshold && low < n.value {
			bit, err := readBit()
			if err != nil {
				return false, err
			}
			if bit == 1 {
				n.value = low
			} else {
				low++
			}
		}
		n.low = low
		if len(stack) == 0 {
			break
		}
		node = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
	}

	return t.nodes[node].value < threshold, nil
}

// excluded skips leaves already proven absent by root.
func (t *tagTree) excluded(threshold int32) bool {
	if len(t.nodes) == 0 {
		return true
	}
	root := &t.nodes[len(t.nodes)-1]
	return root.value == 999 && root.low >= threshold
}
