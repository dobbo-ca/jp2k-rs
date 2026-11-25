//go:build !purego

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// func liftAVX2(dst, d, u, v *float32, n int, c float32)
TEXT ·liftAVX2(SB), NOSPLIT, $0-44
	MOVQ         dst+0(FP), DI
	MOVQ         d+8(FP), BX
	MOVQ         u+16(FP), SI
	MOVQ         v+24(FP), DX
	MOVQ         n+32(FP), CX
	VBROADCASTSS c+40(FP), Y3
	MOVQ         CX, R8
	ANDQ         $7, R8  // scalar tail length
	ANDQ         $-8, CX // whole vectors
	JZ           liftail

liftavx:
	VMOVUPS     (SI), Y1
	VADDPS      (DX), Y1, Y1
	VMOVUPS     (BX), Y0
	VFMADD231PS Y3, Y1, Y0 // Y0 += Y1*Y3, rounded once like liftGo
	VMOVUPS     Y0, (DI)
	ADDQ        $32, SI
	ADDQ        $32, DX
	ADDQ        $32, BX
	ADDQ        $32, DI
	SUBQ        $8, CX
	JNZ         liftavx

liftail:
	TESTQ R8, R8
	JZ    liftdone

liftscalar:
	VMOVSS       (SI), X1
	VADDSS       (DX), X1, X1
	VMOVSS       (BX), X0
	VFMADD231SS  X3, X1, X0 // one rounding, equal to fma32
	VMOVSS       X0, (DI)
	ADDQ         $4, SI
	ADDQ         $4, DX
	ADDQ         $4, BX
	ADDQ         $4, DI
	DECQ         R8
	JNZ          liftscalar

liftdone:
	VZEROUPPER
	RET

// func scaleSSE2(dst, src *float32, n int, k float32)
TEXT ·scaleSSE2(SB), NOSPLIT, $0-28
	MOVQ   dst+0(FP), DI
	MOVQ   src+8(FP), SI
	MOVQ   n+16(FP), CX
	MOVSS  k+24(FP), X3
	SHUFPS $0, X3, X3

scalesse:
	MOVUPS (SI), X0
	MULPS  X3, X0
	MOVUPS X0, (DI)
	ADDQ   $16, SI
	ADDQ   $16, DI
	SUBQ   $4, CX
	JNZ    scalesse
	RET

// func scaleAVX2(dst, src *float32, n int, k float32)
TEXT ·scaleAVX2(SB), NOSPLIT, $0-28
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VBROADCASTSS k+24(FP), Y3

scaleavx:
	VMULPS  (SI), Y3, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     scaleavx
	VZEROUPPER
	RET

// func packF16SSE2(dst *byte, bpp int, src *float32, n int, lo, hi float32, shift int32, sl, sr uint32)
TEXT ·packF16SSE2(SB), NOSPLIT, $0-52
	MOVQ   dst+0(FP), DI
	MOVQ   bpp+8(FP), BX
	MOVQ   src+16(FP), SI
	MOVQ   n+24(FP), CX
	MOVSS  lo+32(FP), X4
	SHUFPS $0, X4, X4
	MOVSS  hi+36(FP), X5
	SHUFPS $0, X5, X5
	MOVL   shift+40(FP), AX
	MOVL   AX, X6
	PSHUFL $0, X6, X6
	MOVL   sl+44(FP), AX
	MOVL   AX, X7
	MOVL   sr+48(FP), AX
	MOVL   AX, X8
	CMPQ   BX, $2
	JEQ    packsse2

packsse:
	MOVUPS   (SI), X0
	MAXPS    X4, X0 // NaN takes lo, matching sampleF's 0 on amd64
	MINPS    X5, X0
	CVTPS2PL X0, X0
	PADDL    X6, X0
	MOVO     X0, X1
	PSLLL    X7, X0
	PSRLL    X8, X1
	POR      X1, X0
	MOVO     X0, X1 // byte-swap each dword's low word
	PSLLL    $8, X1
	PSRLL    $8, X0
	POR      X1, X0
	PEXTRW   $0, X0, AX
	MOVW     AX, (DI)
	ADDQ     BX, DI
	PEXTRW   $2, X0, AX
	MOVW     AX, (DI)
	ADDQ     BX, DI
	PEXTRW   $4, X0, AX
	MOVW     AX, (DI)
	ADDQ     BX, DI
	PEXTRW   $6, X0, AX
	MOVW     AX, (DI)
	ADDQ     BX, DI
	ADDQ     $16, SI
	SUBQ     $4, CX
	JNZ      packsse
	RET

// Gray16 planes are contiguous: one 8-byte store.
packsse2:
	MOVUPS   (SI), X0
	MAXPS    X4, X0
	MINPS    X5, X0
	CVTPS2PL X0, X0
	PADDL    X6, X0
	MOVO     X0, X1
	PSLLL    X7, X0
	PSRLL    X8, X1
	POR      X1, X0
	MOVO     X0, X1
	PSLLL    $8, X1
	PSRLL    $8, X0
	POR      X1, X0
	PSHUFLW  $0x08, X0, X0 // words 0,2,4,6 to 0..3
	PSHUFHW  $0x08, X0, X0
	PSHUFL   $0x08, X0, X0
	MOVQ     X0, (DI)
	ADDQ     $8, DI
	ADDQ     $16, SI
	SUBQ     $4, CX
	JNZ      packsse2
	RET

// func zipSSE2(dst, a, b *float32, n int)
TEXT ·zipSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	MOVQ n+24(FP), CX

zipsse:
	MOVUPS   (SI), X0
	MOVUPS   (DX), X1
	MOVAPS   X0, X2
	UNPCKLPS X1, X0
	UNPCKHPS X1, X2
	MOVUPS   X0, (DI)
	MOVUPS   X2, 16(DI)
	ADDQ     $16, SI
	ADDQ     $16, DX
	ADDQ     $32, DI
	SUBQ     $4, CX
	JNZ      zipsse
	RET

// func liftNAVX2(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)
// Strides in bytes; n >= 1, rows >= 1.
TEXT ·liftNAVX2(SB), NOSPLIT, $0-76
	MOVQ         dst+0(FP), DI
	MOVQ         d+8(FP), BX
	MOVQ         u+16(FP), SI
	MOVQ         v+24(FP), DX
	MOVQ         n+32(FP), R8
	SHLQ         $2, R8 // byte length of n floats
	MOVQ         R8, R13
	ANDQ         $-32, R13 // byte length of the whole vectors
	MOVQ         rows+40(FP), R9
	MOVQ         sDst+48(FP), R10
	MOVQ         sD+56(FP), R11
	MOVQ         sUV+64(FP), R12
	VBROADCASTSS c+72(FP), Y3

nrow:
	MOVQ $0, AX
	CMPQ AX, R13
	JAE  ntail

ncol:
	VMOVUPS     (SI)(AX*1), Y1
	VADDPS      (DX)(AX*1), Y1, Y1
	VMOVUPS     (BX)(AX*1), Y0
	VFMADD231PS Y3, Y1, Y0
	VMOVUPS     Y0, (DI)(AX*1)
	ADDQ        $32, AX
	CMPQ        AX, R13
	JB          ncol

ntail:
	CMPQ AX, R8
	JAE  nnext

nscalar:
	VMOVSS      (SI)(AX*1), X1
	VADDSS      (DX)(AX*1), X1, X1
	VMOVSS      (BX)(AX*1), X0
	VFMADD231SS X3, X1, X0
	VMOVSS      X0, (DI)(AX*1)
	ADDQ        $4, AX
	CMPQ        AX, R8
	JB          nscalar

nnext:
	ADDQ        R10, DI
	ADDQ        R11, BX
	ADDQ        R12, SI
	ADDQ        R12, DX
	DECQ        R9
	JNZ         nrow
	VZEROUPPER
	RET

// Byte swap within each 16-bit word.
DATA bswap16<>+0(SB)/8, $0x0607040502030001
DATA bswap16<>+8(SB)/8, $0x0e0f0c0d0a0b0809
DATA bswap16<>+16(SB)/8, $0x0607040502030001
DATA bswap16<>+24(SB)/8, $0x0e0f0c0d0a0b0809
GLOBL bswap16<>(SB), RODATA|NOPTR, $32

// func packJJ16AVX2(dst *byte, src *float32, n int, shift, hi int32)
// n a multiple of 16.
TEXT ·packJJ16AVX2(SB), NOSPLIT, $0-32
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VBROADCASTSS shift+24(FP), Y10 // int32 bits
	VBROADCASTSS hi+28(FP), Y11
	MOVL         $0x80000000, AX
	MOVL         AX, X12
	VPBROADCASTD X12, Y12 // sign bit
	MOVL         $0xbf000000, AX
	MOVL         AX, X13
	VPBROADCASTD X13, Y13 // -0.5
	VXORPS       Y14, Y14, Y14
	VMOVDQU      bswap16<>(SB), Y15

jjloop:
	VMOVUPS    (SI), Y0
	VMOVUPS    32(SI), Y1
	VCMPPS     $1, Y0, Y14, Y2 // 0 < v
	VCMPPS     $1, Y1, Y14, Y3
	VANDPS     Y12, Y2, Y2
	VANDPS     Y12, Y3, Y3
	VXORPS     Y13, Y2, Y2     // +0.5 or -0.5
	VXORPS     Y13, Y3, Y3
	VADDPS     Y2, Y0, Y0
	VADDPS     Y3, Y1, Y1
	VCMPPS     $7, Y0, Y0, Y2  // not NaN
	VCMPPS     $7, Y1, Y1, Y3
	VCVTTPS2DQ Y0, Y0
	VCVTTPS2DQ Y1, Y1
	VPAND      Y2, Y0, Y0
	VPAND      Y3, Y1, Y1
	VPADDD     Y10, Y0, Y0
	VPADDD     Y10, Y1, Y1
	VPMINSD    Y11, Y0, Y0
	VPMINSD    Y11, Y1, Y1
	VPACKUSDW  Y1, Y0, Y0      // lanes interleave; VPERMQ fixes
	VPERMQ     $0xd8, Y0, Y0
	VPSHUFB    Y15, Y0, Y0
	VMOVDQU    Y0, (DI)
	ADDQ       $64, SI
	ADDQ       $32, DI
	SUBQ       $16, CX
	JNZ        jjloop
	VZEROUPPER
	RET

// JJ2000 lifts: no FMA, n%8 == 0.

// func jjLpfAVX2(dst, l, h0, h1 *float32, n int, kl, kh, d float32)
TEXT ·jjLpfAVX2(SB), NOSPLIT, $0-52
	MOVQ         dst+0(FP), DI
	MOVQ         l+8(FP), SI
	MOVQ         h0+16(FP), DX
	MOVQ         h1+24(FP), BX
	MOVQ         n+32(FP), CX
	VBROADCASTSS kl+40(FP), Y13
	VBROADCASTSS kh+44(FP), Y14
	VBROADCASTSS d+48(FP), Y15

lpfloop:
	VMOVUPS (DX), Y0
	VADDPS  (BX), Y0, Y0
	VMULPS  Y15, Y0, Y0
	VDIVPS  Y14, Y0, Y0
	VMOVUPS (SI), Y1
	VDIVPS  Y13, Y1, Y1
	VSUBPS  Y0, Y1, Y1
	VMOVUPS Y1, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DX
	ADDQ    $32, BX
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     lpfloop
	VZEROUPPER
	RET

// func jjHpfAVX2(dst, h, u, v *float32, n int, kh, c float32)
TEXT ·jjHpfAVX2(SB), NOSPLIT, $0-48
	MOVQ         dst+0(FP), DI
	MOVQ         h+8(FP), SI
	MOVQ         u+16(FP), DX
	MOVQ         v+24(FP), BX
	MOVQ         n+32(FP), CX
	VBROADCASTSS kh+40(FP), Y14
	VBROADCASTSS c+44(FP), Y15

hpfloop:
	VMOVUPS (DX), Y0
	VADDPS  (BX), Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS (SI), Y1
	VDIVPS  Y14, Y1, Y1
	VSUBPS  Y0, Y1, Y1
	VMOVUPS Y1, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DX
	ADDQ    $32, BX
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     hpfloop
	VZEROUPPER
	RET

// func jjSubAVX2(dst, s, u, v *float32, n int, c float32)
TEXT ·jjSubAVX2(SB), NOSPLIT, $0-44
	MOVQ         dst+0(FP), DI
	MOVQ         s+8(FP), SI
	MOVQ         u+16(FP), DX
	MOVQ         v+24(FP), BX
	MOVQ         n+32(FP), CX
	VBROADCASTSS c+40(FP), Y15

subloop:
	VMOVUPS (DX), Y0
	VADDPS  (BX), Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS (SI), Y1
	VSUBPS  Y0, Y1, Y1
	VMOVUPS Y1, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DX
	ADDQ    $32, BX
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     subloop
	VZEROUPPER
	RET

// func jjDivAVX2(x *float32, n int, k float32)
TEXT ·jjDivAVX2(SB), NOSPLIT, $0-20
	MOVQ         x+0(FP), DI
	MOVQ         n+8(FP), CX
	VBROADCASTSS k+16(FP), Y15

divloop:
	VMOVUPS (DI), Y0
	VDIVPS  Y15, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     divloop
	VZEROUPPER
	RET

// Separate rounding: product rounded, then the sum. No FMA.

// func liftSepAVX2(dst, d, u, v *float32, n int, c float32)
TEXT ·liftSepAVX2(SB), NOSPLIT, $0-44
	MOVQ         dst+0(FP), DI
	MOVQ         d+8(FP), BX
	MOVQ         u+16(FP), SI
	MOVQ         v+24(FP), DX
	MOVQ         n+32(FP), CX
	VBROADCASTSS c+40(FP), Y3
	MOVQ         CX, R8
	ANDQ         $7, R8  // scalar tail length
	ANDQ         $-8, CX // whole vectors
	JZ           lsepail

lsepavx:
	VMOVUPS (SI), Y1
	VADDPS  (DX), Y1, Y1
	VMULPS  Y3, Y1, Y1
	VADDPS  (BX), Y1, Y1
	VMOVUPS Y1, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DX
	ADDQ    $32, BX
	ADDQ    $32, DI
	SUBQ    $8, CX
	JNZ     lsepavx

lsepail:
	TESTQ R8, R8
	JZ    lsepdone

lsepscalar:
	VMOVSS (SI), X1
	VADDSS (DX), X1, X1
	VMULSS X3, X1, X1
	VADDSS (BX), X1, X1
	VMOVSS X1, (DI)
	ADDQ   $4, SI
	ADDQ   $4, DX
	ADDQ   $4, BX
	ADDQ   $4, DI
	DECQ   R8
	JNZ    lsepscalar

lsepdone:
	VZEROUPPER
	RET

// func liftNSepAVX2(dst, d, u, v *float32, n, rows, sDst, sD, sUV int, c float32)
// Strides in bytes; n >= 1, rows >= 1.
TEXT ·liftNSepAVX2(SB), NOSPLIT, $0-76
	MOVQ         dst+0(FP), DI
	MOVQ         d+8(FP), BX
	MOVQ         u+16(FP), SI
	MOVQ         v+24(FP), DX
	MOVQ         n+32(FP), R8
	SHLQ         $2, R8 // byte length of n floats
	MOVQ         R8, R13
	ANDQ         $-32, R13 // byte length of the whole vectors
	MOVQ         rows+40(FP), R9
	MOVQ         sDst+48(FP), R10
	MOVQ         sD+56(FP), R11
	MOVQ         sUV+64(FP), R12
	VBROADCASTSS c+72(FP), Y3

nsrow:
	MOVQ $0, AX
	CMPQ AX, R13
	JAE  nstail

nscol:
	VMOVUPS (SI)(AX*1), Y1
	VADDPS  (DX)(AX*1), Y1, Y1
	VMULPS  Y3, Y1, Y1
	VADDPS  (BX)(AX*1), Y1, Y1
	VMOVUPS Y1, (DI)(AX*1)
	ADDQ    $32, AX
	CMPQ    AX, R13
	JB      nscol

nstail:
	CMPQ AX, R8
	JAE  nsnext

nsscalar:
	VMOVSS (SI)(AX*1), X1
	VADDSS (DX)(AX*1), X1, X1
	VMULSS X3, X1, X1
	VADDSS (BX)(AX*1), X1, X1
	VMOVSS X1, (DI)(AX*1)
	ADDQ   $4, AX
	CMPQ   AX, R8
	JB     nsscalar

nsnext:
	ADDQ    R10, DI
	ADDQ    R11, BX
	ADDQ    R12, SI
	ADDQ    R12, DX
	DECQ    R9
	JNZ     nsrow
	VZEROUPPER
	RET

// ICT: c0, c1, c2 in place, n a multiple of 8. All three vectors load
// before any store.

// func ictFusedAVX2(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)
TEXT ·ictFusedAVX2(SB), NOSPLIT, $0-48
	MOVQ         c0+0(FP), DI
	MOVQ         c1+8(FP), SI
	MOVQ         c2+16(FP), DX
	MOVQ         n+24(FP), CX
	VBROADCASTSS crr+32(FP), Y8
	VBROADCASTSS cbg+36(FP), Y9
	VBROADCASTSS crg+40(FP), Y10
	VBROADCASTSS cbb+44(FP), Y11

ictfloop:
	VMOVUPS      (DI), Y0 // y
	VMOVUPS      (SI), Y1 // u
	VMOVUPS      (DX), Y2 // v
	VMOVAPS      Y0, Y3
	VMOVAPS      Y0, Y4
	VFMADD231PS  Y8, Y2, Y0  // y + v*crr
	VFNMADD231PS Y9, Y1, Y3  // y - u*cbg
	VFNMADD231PS Y10, Y2, Y3 // then - v*crg
	VFMADD231PS  Y11, Y1, Y4 // y + u*cbb
	VMOVUPS      Y0, (DI)
	VMOVUPS      Y3, (SI)
	VMOVUPS      Y4, (DX)
	ADDQ         $32, DI
	ADDQ         $32, SI
	ADDQ         $32, DX
	SUBQ         $8, CX
	JNZ          ictfloop
	VZEROUPPER
	RET

// func ictSepAVX2(c0, c1, c2 *float32, n int, crr, cbg, crg, cbb float32)
TEXT ·ictSepAVX2(SB), NOSPLIT, $0-48
	MOVQ         c0+0(FP), DI
	MOVQ         c1+8(FP), SI
	MOVQ         c2+16(FP), DX
	MOVQ         n+24(FP), CX
	VBROADCASTSS crr+32(FP), Y8
	VBROADCASTSS cbg+36(FP), Y9
	VBROADCASTSS crg+40(FP), Y10
	VBROADCASTSS cbb+44(FP), Y11

ictsloop:
	VMOVUPS (DI), Y0 // y
	VMOVUPS (SI), Y1 // u
	VMOVUPS (DX), Y2 // v
	VMULPS  Y8, Y2, Y5
	VADDPS  Y5, Y0, Y5 // y + v*crr
	VMULPS  Y9, Y1, Y6
	VSUBPS  Y6, Y0, Y3 // y - u*cbg
	VMULPS  Y10, Y2, Y7
	VSUBPS  Y7, Y3, Y3 // then - v*crg
	VMULPS  Y11, Y1, Y4
	VADDPS  Y4, Y0, Y4 // y + u*cbb
	VMOVUPS Y5, (DI)
	VMOVUPS Y3, (SI)
	VMOVUPS Y4, (DX)
	ADDQ    $32, DI
	ADDQ    $32, SI
	ADDQ    $32, DX
	SUBQ    $8, CX
	JNZ     ictsloop
	VZEROUPPER
	RET

// func dequantFAVX2(dst *float32, mag *int32, flags *uint32, n int, sh uint32, bNo, bPi, d float32)
// n a multiple of 8. Matches dequantFGo: flags >>= sh, bit 21 picks
// bNo/bPi, bit 19 is the sign, bit 4 gates significance. No FMA.
TEXT ·dequantFAVX2(SB), NOSPLIT, $0-48
	MOVQ         dst+0(FP), DI
	MOVQ         mag+8(FP), SI
	MOVQ         flags+16(FP), DX
	MOVQ         n+24(FP), CX
	MOVL         sh+32(FP), AX
	VMOVD        AX, X4 // shift count, upper bits zero
	VBROADCASTSS bNo+36(FP), Y8
	VBROADCASTSS bPi+40(FP), Y9
	VBROADCASTSS d+44(FP), Y10
	MOVL         $0x80000000, AX
	VMOVD        AX, X11
	VPBROADCASTD X11, Y11 // sign bit

dqloop:
	VMOVDQU   (DX), Y0
	VPSRLD    X4, Y0, Y0     // f >>= sh
	VCVTDQ2PS (SI), Y1       // float32(mag), nearest-even
	VPSLLD    $10, Y0, Y2    // bit 21 to bit 31
	VBLENDVPS Y2, Y9, Y8, Y3 // bit 31 set: bPi, else bNo
	VADDPS    Y3, Y1, Y1
	VMULPS    Y10, Y1, Y1
	VPSLLD    $12, Y0, Y2    // bit 19 to bit 31
	VPAND     Y11, Y2, Y2
	VXORPS    Y2, Y1, Y1     // sign
	VPSLLD    $27, Y0, Y2    // bit 4 to bit 31
	VPSRAD    $31, Y2, Y2
	VPAND     Y2, Y1, Y1     // zero insignificant lanes
	VMOVUPS   Y1, (DI)
	ADDQ      $32, DX
	ADDQ      $32, SI
	ADDQ      $32, DI
	SUBQ      $8, CX
	JNZ       dqloop
	VZEROUPPER
	RET
