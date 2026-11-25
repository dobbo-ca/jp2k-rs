//go:build !purego

#include "textflag.h"

// Vector FADD/FMUL/FMLA are WORD-encoded: the go 1.22 assembler has no
// mnemonics for them. FMLA is fused, matching what the Go compiler emits
// for liftGo on arm64.

// func liftNEON(dst, d, u, v *float32, n int, c float32)
TEXT ·liftNEON(SB), NOSPLIT, $0-44
	MOVD  dst+0(FP), R0
	MOVD  d+8(FP), R4
	MOVD  u+16(FP), R1
	MOVD  v+24(FP), R2
	MOVD  n+32(FP), R3
	FMOVS c+40(FP), F3
	VDUP  V3.S[0], V20.S4

	CMP $16, R3
	BLT loop

// V20 holds c; V0-V7, V16-V19 data.
l16:
	VLD1.P 64(R4), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4e30d484 // FADD V4.4S, V4.4S, V16.4S
	WORD $0x4e31d4a5 // FADD V5.4S, V5.4S, V17.4S
	WORD $0x4e32d4c6 // FADD V6.4S, V6.4S, V18.4S
	WORD $0x4e33d4e7 // FADD V7.4S, V7.4S, V19.4S
	WORD $0x4e34cc80 // FMLA V0.4S, V4.4S, V20.4S
	WORD $0x4e34cca1 // FMLA V1.4S, V5.4S, V20.4S
	WORD $0x4e34ccc2 // FMLA V2.4S, V6.4S, V20.4S
	WORD $0x4e34cce3 // FMLA V3.4S, V7.4S, V20.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUB    $16, R3
	CMP    $16, R3
	BGE    l16
	CBZ    R3, ldone

loop:
	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R2), [V2.S4]
	WORD   $0x4e22d421 // FADD V1.4S, V1.4S, V2.4S
	WORD   $0x4e34cc20 // FMLA V0.4S, V1.4S, V20.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    loop
ldone:
	RET

// func scaleNEON(dst, src *float32, n int, k float32)
TEXT ·scaleNEON(SB), NOSPLIT, $0-28
	MOVD  dst+0(FP), R0
	MOVD  src+8(FP), R4
	MOVD  n+16(FP), R3
	FMOVS k+24(FP), F3
	VDUP  V3.S[0], V20.S4

	CMP $16, R3
	BLT sloop

s16:
	VLD1.P 64(R4), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD $0x6e34dc00 // FMUL V0.4S, V0.4S, V20.4S
	WORD $0x6e34dc21 // FMUL V1.4S, V1.4S, V20.4S
	WORD $0x6e34dc42 // FMUL V2.4S, V2.4S, V20.4S
	WORD $0x6e34dc63 // FMUL V3.4S, V3.4S, V20.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUB    $16, R3
	CMP    $16, R3
	BGE    s16
	CBZ    R3, sdone

sloop:
	VLD1.P 16(R4), [V0.S4]
	WORD   $0x6e34dc00 // FMUL V0.4S, V0.4S, V20.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    sloop
sdone:
	RET

// FCVTNS rounds half to even and saturates to int32, as sampleF's guards
// and conversion do. NaN lanes are masked to 0, as sampleF gives them.

// func packF16NEON(dst *byte, bpp int, src *float32, n int, lo, hi, shift, sl, sr int32)
TEXT ·packF16NEON(SB), NOSPLIT, $0-52
	MOVD  dst+0(FP), R0
	MOVD  bpp+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  n+24(FP), R3
	MOVW  lo+32(FP), R4
	VDUP  R4, V20.S4
	MOVW  hi+36(FP), R4
	VDUP  R4, V21.S4
	MOVW  shift+40(FP), R4
	VDUP  R4, V22.S4
	MOVW  sl+44(FP), R4
	VDUP  R4, V23.S4
	MOVW  sr+48(FP), R4
	VDUP  R4, V24.S4
	CMP   $2, R1
	BEQ   ploop2

ploop:
	VLD1.P 16(R2), [V0.S4]
	WORD   $0x4e20e405 // FCMEQ V5.4S, V0.4S, V0.4S
	WORD   $0x4e21a800 // FCVTNS V0.4S, V0.4S
	WORD   $0x4eb46400 // SMAX V0.4S, V0.4S, V20.4S
	WORD   $0x4eb56c00 // SMIN V0.4S, V0.4S, V21.4S
	VADD   V22.S4, V0.S4, V0.S4
	VAND   V5.B16, V0.B16, V0.B16
	WORD   $0x6eb74401 // USHL V1.4S, V0.4S, V23.4S
	WORD   $0x6eb84402 // USHL V2.4S, V0.4S, V24.4S
	VORR   V2.B16, V1.B16, V1.B16
	WORD   $0x0e612821 // XTN V1.4H, V1.4S
	VREV16 V1.B8, V1.B8
	VMOV   V1.H[0], R6
	MOVH   R6, (R0)
	ADD    R1, R0
	VMOV   V1.H[1], R6
	MOVH   R6, (R0)
	ADD    R1, R0
	VMOV   V1.H[2], R6
	MOVH   R6, (R0)
	ADD    R1, R0
	VMOV   V1.H[3], R6
	MOVH   R6, (R0)
	ADD    R1, R0
	SUBS   $4, R3
	BNE    ploop
	RET

// Gray16 planes are contiguous: one 8-byte store.
ploop2:
	VLD1.P 16(R2), [V0.S4]
	WORD   $0x4e20e405 // FCMEQ V5.4S, V0.4S, V0.4S
	WORD   $0x4e21a800 // FCVTNS V0.4S, V0.4S
	WORD   $0x4eb46400 // SMAX V0.4S, V0.4S, V20.4S
	WORD   $0x4eb56c00 // SMIN V0.4S, V0.4S, V21.4S
	VADD   V22.S4, V0.S4, V0.S4
	VAND   V5.B16, V0.B16, V0.B16
	WORD   $0x6eb74401 // USHL V1.4S, V0.4S, V23.4S
	WORD   $0x6eb84402 // USHL V2.4S, V0.4S, V24.4S
	VORR   V2.B16, V1.B16, V1.B16
	WORD   $0x0e612821 // XTN V1.4H, V1.4S
	VREV16 V1.B8, V1.B8
	FMOVD  F1, (R0)
	ADD    $8, R0
	SUBS   $4, R3
	BNE    ploop2
	RET

// func pack16NEON(dst *byte, src *float32, n int)
TEXT ·pack16NEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R2
	MOVD n+16(FP), R3
	MOVD $0x8000, R4
	VDUP R4, V20.H8
p16loop:
	VLD1.P 64(R2), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD   $0x4e20e406 // FCMEQ V6.4S, V0.4S, V0.4S
	WORD   $0x4e21e427 // FCMEQ V7.4S, V1.4S, V1.4S
	WORD   $0x4e22e448 // FCMEQ V8.4S, V2.4S, V2.4S
	WORD   $0x4e23e469 // FCMEQ V9.4S, V3.4S, V3.4S
	WORD   $0x0e6128ca // XTN V10.4H, V6.4S
	WORD   $0x4e6128ea // XTN2 V10.8H, V7.4S
	WORD   $0x0e61290b // XTN V11.4H, V8.4S
	WORD   $0x4e61292b // XTN2 V11.8H, V9.4S
	WORD   $0x4e21a800 // FCVTNS V0.4S, V0.4S
	WORD   $0x4e21a821 // FCVTNS V1.4S, V1.4S
	WORD   $0x4e21a842 // FCVTNS V2.4S, V2.4S
	WORD   $0x4e21a863 // FCVTNS V3.4S, V3.4S
	WORD   $0x0e614804 // SQXTN V4.4H, V0.4S
	WORD   $0x4e614824 // SQXTN2 V4.8H, V1.4S
	WORD   $0x0e614845 // SQXTN V5.4H, V2.4S
	WORD   $0x4e614865 // SQXTN2 V5.8H, V3.4S
	VEOR   V20.B16, V4.B16, V4.B16
	VEOR   V20.B16, V5.B16, V5.B16
	VAND   V10.B16, V4.B16, V4.B16
	VAND   V11.B16, V5.B16, V5.B16
	VREV16 V4.B16, V4.B16
	VREV16 V5.B16, V5.B16
	VST1.P [V4.B16, V5.B16], 32(R0)
	SUBS   $16, R3
	BNE    p16loop
	RET

// func zipNEON(dst, a, b *float32, n int)
TEXT ·zipNEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD n+24(FP), R3
zloop:
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 16(R2), [V1.S4]
	VST2.P [V0.S4, V1.S4], 32(R0)
	SUBS   $4, R3
	BNE    zloop
	RET

// func liftNNEON(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)
// Strides in bytes; n a multiple of 4.
TEXT ·liftNNEON(SB), NOSPLIT, $0-76
	MOVD  dst+0(FP), R0
	MOVD  d+8(FP), R4
	MOVD  u+16(FP), R1
	MOVD  v+24(FP), R2
	MOVD  n+32(FP), R5
	MOVD  rows+40(FP), R6
	MOVD  sDst+48(FP), R7
	MOVD  sD+56(FP), R8
	MOVD  sUV+64(FP), R9
	FMOVS c+72(FP), F3
	VDUP  V3.S[0], V20.S4
	LSL   $2, R5, R10
	SUB   R10, R7, R7
	SUB   R10, R8, R8
	SUB   R10, R9, R9

nrow:
	MOVD  R5, R3

	CMP $16, R3
	BLT nloop

n16:
	VLD1.P 64(R4), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R1), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R2), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4e30d484 // FADD V4.4S, V4.4S, V16.4S
	WORD $0x4e31d4a5 // FADD V5.4S, V5.4S, V17.4S
	WORD $0x4e32d4c6 // FADD V6.4S, V6.4S, V18.4S
	WORD $0x4e33d4e7 // FADD V7.4S, V7.4S, V19.4S
	WORD $0x4e34cc80 // FMLA V0.4S, V4.4S, V20.4S
	WORD $0x4e34cca1 // FMLA V1.4S, V5.4S, V20.4S
	WORD $0x4e34ccc2 // FMLA V2.4S, V6.4S, V20.4S
	WORD $0x4e34cce3 // FMLA V3.4S, V7.4S, V20.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUB    $16, R3
	CMP    $16, R3
	BGE    n16
	CBZ    R3, nnext

nloop:
	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R2), [V2.S4]
	WORD   $0x4e22d421 // FADD V1.4S, V1.4S, V2.4S
	WORD   $0x4e34cc20 // FMLA V0.4S, V1.4S, V20.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    nloop
nnext:
	ADD    R7, R0
	ADD    R8, R4
	ADD    R9, R1
	ADD    R9, R2
	SUBS   $1, R6
	BNE    nrow
	RET

// func scaleNNEON(dst, src *float32, n, rows, sDst, sSrc int, k float32)
// Strides in bytes; n a multiple of 4.
TEXT ·scaleNNEON(SB), NOSPLIT, $0-52
	MOVD  dst+0(FP), R0
	MOVD  src+8(FP), R4
	MOVD  n+16(FP), R5
	MOVD  rows+24(FP), R6
	MOVD  sDst+32(FP), R7
	MOVD  sSrc+40(FP), R8
	FMOVS k+48(FP), F3
	VDUP  V3.S[0], V20.S4
	MOVD  $8, R11
	MUL   R8, R11, R12
	LSL   $2, R5, R10
	SUB   R10, R7, R7
	SUB   R10, R8, R8

snrow:
	MOVD  R5, R3
	ADD   R4, R12, R14
	PRFM  (R14), PLDL1KEEP
	PRFM  64(R14), PLDL1KEEP
	PRFM  128(R14), PLDL1KEEP
	PRFM  192(R14), PLDL1KEEP

	CMP $16, R3
	BLT snloop

sn16:
	VLD1.P 64(R4), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD $0x6e34dc00 // FMUL V0.4S, V0.4S, V20.4S
	WORD $0x6e34dc21 // FMUL V1.4S, V1.4S, V20.4S
	WORD $0x6e34dc42 // FMUL V2.4S, V2.4S, V20.4S
	WORD $0x6e34dc63 // FMUL V3.4S, V3.4S, V20.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUB    $16, R3
	CMP    $16, R3
	BGE    sn16
	CBZ    R3, snnext

snloop:
	VLD1.P 16(R4), [V0.S4]
	WORD   $0x6e34dc00 // FMUL V0.4S, V0.4S, V20.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    snloop
snnext:
	ADD    R7, R0
	ADD    R8, R4
	SUBS   $1, R6
	BNE    snrow
	RET

// func dequantFNEON(dst *float32, mag *int32, flags *uint32, n int, sh uint32, bNo, bPi, d float32)
// Matches Finish's dequant loop: flags is pre-shifted by sh, bit 21
// picks bNo/bPi, bit 19 is the sign, bit 4 gates significance.
TEXT ·dequantFNEON(SB), NOSPLIT, $0-48
	MOVD  dst+0(FP), R0
	MOVD  mag+8(FP), R1
	MOVD  flags+16(FP), R2
	MOVD  n+24(FP), R3
	MOVW  sh+32(FP), R4
	NEG   R4, R4
	VDUP  R4, V16.S4 // -sh, per-lane dynamic shift
	FMOVS bNo+36(FP), F17
	VDUP  V17.S[0], V17.S4
	FMOVS bPi+40(FP), F18
	VDUP  V18.S[0], V18.S4
	FMOVS d+44(FP), F19
	VDUP  V19.S[0], V19.S4
	MOVW  $0x200000, R5 // t1Pi, bit 21
	VDUP  R5, V21.S4
	MOVW  $0x80000000, R6
	VDUP  R6, V22.S4 // sign bit, bit 31
	MOVW  $0x10, R7 // t1Sig, bit 4
	VDUP  R7, V23.S4

dqloop:
	VLD1.P 16(R2), [V0.S4]      // flags
	VUSHL  V16.S4, V0.S4, V0.S4 // f >>= sh
	VLD1.P 16(R1), [V1.S4]      // mag
	VSCVTF V1.S4, V1.S4         // float32(mag)
	VCMTST V21.S4, V0.S4, V2.S4 // bit21 mask
	VMOV   V17.B16, V4.B16
	VBIT   V2.B16, V18.B16, V4.B16 // V4 = bit21 ? bPi : bNo
	WORD   $0x4e24d421           // FADD V1.4S, V1.4S, V4.4S
	WORD   $0x6e33dc21           // FMUL V1.4S, V1.4S, V19.4S
	VSHL   $12, V0.S4, V3.S4     // bit19 -> bit31
	VAND   V22.B16, V3.B16, V3.B16
	VEOR   V3.B16, V1.B16, V1.B16 // apply sign
	VCMTST V23.S4, V0.S4, V5.S4   // bit4 mask
	VAND   V5.B16, V1.B16, V1.B16 // zero insignificant lanes
	VST1.P [V1.S4], 16(R0)
	SUBS   $4, R3
	BNE    dqloop
	RET

// func packJJ16NEON(dst *byte, src *float32, n int, shift, hi int32)
// n a multiple of 8.
TEXT ·packJJ16NEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP), R0
	MOVD   src+8(FP), R2
	MOVD   n+16(FP), R3
	MOVW   shift+24(FP), R4
	VDUP   R4, V22.S4
	MOVW   hi+28(FP), R4
	VDUP   R4, V23.S4
	MOVW   $0x80000000, R4
	VDUP   R4, V20.S4 // sign bit
	MOVW   $0xbf000000, R4
	VDUP   R4, V21.S4 // -0.5
	VEOR   V24.B16, V24.B16, V24.B16

jjloop:
	VLD1.P 32(R2), [V0.S4, V1.S4]
	WORD   $0x4ea0c802 // FCMGT V2.4S, V0.4S, #0.0
	WORD   $0x4ea0c823 // FCMGT V3.4S, V1.4S, #0.0
	VAND   V20.B16, V2.B16, V2.B16
	VAND   V20.B16, V3.B16, V3.B16
	VEOR   V21.B16, V2.B16, V2.B16 // +0.5 or -0.5
	VEOR   V21.B16, V3.B16, V3.B16
	WORD   $0x4e22d400 // FADD V0.4S, V0.4S, V2.4S
	WORD   $0x4e23d421 // FADD V1.4S, V1.4S, V3.4S
	WORD   $0x4ea1b800 // FCVTZS V0.4S, V0.4S
	WORD   $0x4ea1b821 // FCVTZS V1.4S, V1.4S
	VADD   V22.S4, V0.S4, V0.S4
	VADD   V22.S4, V1.S4, V1.S4
	WORD   $0x4eb86400 // SMAX V0.4S, V0.4S, V24.4S
	WORD   $0x4eb86421 // SMAX V1.4S, V1.4S, V24.4S
	WORD   $0x4eb76c00 // SMIN V0.4S, V0.4S, V23.4S
	WORD   $0x4eb76c21 // SMIN V1.4S, V1.4S, V23.4S
	WORD   $0x0e612804 // XTN V4.4H, V0.4S
	WORD   $0x4e612824 // XTN2 V4.8H, V1.4S
	VREV16 V4.B16, V4.B16
	VST1.P [V4.B16], 16(R0)
	SUBS   $8, R3
	BNE    jjloop
	RET

// JJ2000 lifts: no FMLA, n%4 == 0.

// func jjLpfNEON(dst, l, h0, h1 *float32, n int, kl, kh, d float32)
TEXT ·jjLpfNEON(SB), NOSPLIT, $0-52
	MOVD  dst+0(FP), R0
	MOVD  l+8(FP), R1
	MOVD  h0+16(FP), R2
	MOVD  h1+24(FP), R3
	MOVD  n+32(FP), R4
	FMOVS kl+40(FP), F20
	VDUP  V20.S[0], V20.S4
	FMOVS kh+44(FP), F21
	VDUP  V21.S[0], V21.S4
	FMOVS d+48(FP), F22
	VDUP  V22.S[0], V22.S4

lpfloop:
	VLD1.P 16(R2), [V0.S4]
	VLD1.P 16(R3), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	WORD   $0x4e21d400 // FADD V0.4S, V0.4S, V1.4S
	WORD   $0x6e36dc00 // FMUL V0.4S, V0.4S, V22.4S
	WORD   $0x6e35fc00 // FDIV V0.4S, V0.4S, V21.4S
	WORD   $0x6e34fc42 // FDIV V2.4S, V2.4S, V20.4S
	WORD   $0x4ea0d442 // FSUB V2.4S, V2.4S, V0.4S
	VST1.P [V2.S4], 16(R0)
	SUBS   $4, R4
	BNE    lpfloop
	RET

// func jjHpfNEON(dst, h, u, v *float32, n int, kh, c float32)
TEXT ·jjHpfNEON(SB), NOSPLIT, $0-48
	MOVD  dst+0(FP), R0
	MOVD  h+8(FP), R1
	MOVD  u+16(FP), R2
	MOVD  v+24(FP), R3
	MOVD  n+32(FP), R4
	FMOVS kh+40(FP), F21
	VDUP  V21.S[0], V21.S4
	FMOVS c+44(FP), F22
	VDUP  V22.S[0], V22.S4

hpfloop:
	VLD1.P 16(R2), [V0.S4]
	VLD1.P 16(R3), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	WORD   $0x4e21d400 // FADD V0.4S, V0.4S, V1.4S
	WORD   $0x6e36dc00 // FMUL V0.4S, V0.4S, V22.4S
	WORD   $0x6e35fc42 // FDIV V2.4S, V2.4S, V21.4S
	WORD   $0x4ea0d442 // FSUB V2.4S, V2.4S, V0.4S
	VST1.P [V2.S4], 16(R0)
	SUBS   $4, R4
	BNE    hpfloop
	RET

// func jjSubNEON(dst, s, u, v *float32, n int, c float32)
TEXT ·jjSubNEON(SB), NOSPLIT, $0-44
	MOVD  dst+0(FP), R0
	MOVD  s+8(FP), R1
	MOVD  u+16(FP), R2
	MOVD  v+24(FP), R3
	MOVD  n+32(FP), R4
	FMOVS c+40(FP), F22
	VDUP  V22.S[0], V22.S4

subloop:
	VLD1.P 16(R2), [V0.S4]
	VLD1.P 16(R3), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	WORD   $0x4e21d400 // FADD V0.4S, V0.4S, V1.4S
	WORD   $0x6e36dc00 // FMUL V0.4S, V0.4S, V22.4S
	WORD   $0x4ea0d442 // FSUB V2.4S, V2.4S, V0.4S
	VST1.P [V2.S4], 16(R0)
	SUBS   $4, R4
	BNE    subloop
	RET

// func jjDivNEON(x *float32, n int, k float32)
TEXT ·jjDivNEON(SB), NOSPLIT, $0-20
	MOVD  x+0(FP), R0
	MOVD  n+8(FP), R4
	FMOVS k+16(FP), F22
	VDUP  V22.S[0], V22.S4

divloop:
	VLD1   (R0), [V0.S4]
	WORD   $0x6e36fc00 // FDIV V0.4S, V0.4S, V22.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R4
	BNE    divloop
	RET

// Separate rounding: FMUL then FADD/FSUB, never FMLA/FMLS.

// func liftSepNEON(dst, d, u, v *float32, n int, c float32)
TEXT ·liftSepNEON(SB), NOSPLIT, $0-44
	MOVD  dst+0(FP), R0
	MOVD  d+8(FP), R4
	MOVD  u+16(FP), R1
	MOVD  v+24(FP), R2
	MOVD  n+32(FP), R3
	FMOVS c+40(FP), F3
	VDUP  V3.S[0], V20.S4

lsloop:
	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R2), [V2.S4]
	WORD   $0x4e22d421 // FADD V1.4S, V1.4S, V2.4S
	WORD   $0x6e34dc21 // FMUL V1.4S, V1.4S, V20.4S
	WORD   $0x4e21d400 // FADD V0.4S, V0.4S, V1.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    lsloop
	RET

// func liftNSepNEON(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)
// Strides in bytes; n a multiple of 4.
TEXT ·liftNSepNEON(SB), NOSPLIT, $0-76
	MOVD  dst+0(FP), R0
	MOVD  d+8(FP), R4
	MOVD  u+16(FP), R1
	MOVD  v+24(FP), R2
	MOVD  n+32(FP), R5
	MOVD  rows+40(FP), R6
	MOVD  sDst+48(FP), R7
	MOVD  sD+56(FP), R8
	MOVD  sUV+64(FP), R9
	FMOVS c+72(FP), F3
	VDUP  V3.S[0], V20.S4
	LSL   $2, R5, R10
	SUB   R10, R7, R7
	SUB   R10, R8, R8
	SUB   R10, R9, R9

lnsrow:
	MOVD R5, R3

lnsloop:
	VLD1.P 16(R4), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R2), [V2.S4]
	WORD   $0x4e22d421 // FADD V1.4S, V1.4S, V2.4S
	WORD   $0x6e34dc21 // FMUL V1.4S, V1.4S, V20.4S
	WORD   $0x4e21d400 // FADD V0.4S, V0.4S, V1.4S
	VST1.P [V0.S4], 16(R0)
	SUBS   $4, R3
	BNE    lnsloop
	ADD    R7, R0
	ADD    R8, R4
	ADD    R9, R1
	ADD    R9, R2
	SUBS   $1, R6
	BNE    lnsrow
	RET

// ICT: c0, c1, c2 in place, n a multiple of 4. V20-V23 hold
// crr, cbg, crg, cbb. FMLS is fused: d - n*m = fma(-n, m, d).

// func ictFusedNEON(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)
TEXT ·ictFusedNEON(SB), NOSPLIT, $0-48
	MOVD  c0+0(FP), R0
	MOVD  c1+8(FP), R1
	MOVD  c2+16(FP), R2
	MOVD  n+24(FP), R3
	FMOVS crr+32(FP), F20
	VDUP  V20.S[0], V20.S4
	FMOVS cbg+36(FP), F21
	VDUP  V21.S[0], V21.S4
	FMOVS crg+40(FP), F22
	VDUP  V22.S[0], V22.S4
	FMOVS cbb+44(FP), F23
	VDUP  V23.S[0], V23.S4

ictfloop:
	VLD1 (R0), [V0.S4] // y
	VLD1 (R1), [V1.S4] // u
	VLD1 (R2), [V2.S4] // v
	VMOV V0.B16, V3.B16
	VMOV V0.B16, V4.B16
	WORD $0x4e34cc40 // FMLA V0.4S, V2.4S, V20.4S
	WORD $0x4eb5cc23 // FMLS V3.4S, V1.4S, V21.4S
	WORD $0x4eb6cc43 // FMLS V3.4S, V2.4S, V22.4S
	WORD $0x4e37cc24 // FMLA V4.4S, V1.4S, V23.4S
	VST1.P [V0.S4], 16(R0)
	VST1.P [V3.S4], 16(R1)
	VST1.P [V4.S4], 16(R2)
	SUBS   $4, R3
	BNE    ictfloop
	RET

// func ictSepNEON(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)
TEXT ·ictSepNEON(SB), NOSPLIT, $0-48
	MOVD  c0+0(FP), R0
	MOVD  c1+8(FP), R1
	MOVD  c2+16(FP), R2
	MOVD  n+24(FP), R3
	FMOVS crr+32(FP), F20
	VDUP  V20.S[0], V20.S4
	FMOVS cbg+36(FP), F21
	VDUP  V21.S[0], V21.S4
	FMOVS crg+40(FP), F22
	VDUP  V22.S[0], V22.S4
	FMOVS cbb+44(FP), F23
	VDUP  V23.S[0], V23.S4

ictsloop:
	VLD1 (R0), [V0.S4] // y
	VLD1 (R1), [V1.S4] // u
	VLD1 (R2), [V2.S4] // v
	VMOV V0.B16, V3.B16
	VMOV V0.B16, V4.B16
	WORD $0x6e34dc45 // FMUL V5.4S, V2.4S, V20.4S
	WORD $0x4e25d400 // FADD V0.4S, V0.4S, V5.4S
	WORD $0x6e35dc26 // FMUL V6.4S, V1.4S, V21.4S
	WORD $0x4ea6d463 // FSUB V3.4S, V3.4S, V6.4S
	WORD $0x6e36dc47 // FMUL V7.4S, V2.4S, V22.4S
	WORD $0x4ea7d463 // FSUB V3.4S, V3.4S, V7.4S
	WORD $0x6e37dc30 // FMUL V16.4S, V1.4S, V23.4S
	WORD $0x4e30d484 // FADD V4.4S, V4.4S, V16.4S
	VST1.P [V0.S4], 16(R0)
	VST1.P [V3.S4], 16(R1)
	VST1.P [V4.S4], 16(R2)
	SUBS   $4, R3
	BNE    ictsloop
	RET
