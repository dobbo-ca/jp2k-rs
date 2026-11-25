package dsp

// The arm64 compiler fuses this into one FMADDS (TestFMA32 checks).
func fma32(a, b, c float32) float32 { return a*b + c }
