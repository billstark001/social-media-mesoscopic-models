package numerics

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
)

// TridiagonalSystem stores an immutable tridiagonal operator. Backends may
// copy and factor its diagonals using their preferred batched solver.
type TridiagonalSystem struct {
	Size     int
	Lower    []float64
	Diagonal []float64
	Upper    []float64
}

func NewTridiagonalSystem(lower, diagonal, upper []float64) *TridiagonalSystem {
	return &TridiagonalSystem{
		Size: len(diagonal), Lower: append([]float64(nil), lower...),
		Diagonal: append([]float64(nil), diagonal...), Upper: append([]float64(nil), upper...),
	}
}

func solveTridiagonalBatch(dst, source []float64, system *TridiagonalSystem, rightHandSides int) {
	copy(dst, source)
	lower := append([]float64(nil), system.Lower...)
	diagonal := append([]float64(nil), system.Diagonal...)
	upper := append([]float64(nil), system.Upper...)
	ok := lapack64.Gtsv(
		blas.NoTrans,
		lapack64.Tridiagonal{N: system.Size, DL: lower, D: diagonal, DU: upper},
		blas64.General{Rows: system.Size, Cols: rightHandSides, Stride: rightHandSides, Data: dst},
	)
	if !ok {
		panic("numerics: singular tridiagonal system")
	}
}

func transportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem) {
	size := system.Size
	solveTridiagonalBatch(dst, source, system, size)
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			scratch[right*size+left] = dst[left*size+right]
		}
	}
	solveTridiagonalBatch(dst, scratch, system, size)
	for right := 0; right < size; right++ {
		for left := 0; left < size; left++ {
			scratch[left*size+right] = dst[right*size+left]
		}
	}
	copy(dst, scratch)
}

func tensorIndex(channel, left, center, right, size int) int {
	return channel*size*size*size + (left*size+center)*size + right
}

func tensorLine(channel, left, center, right, size, axis int) int {
	switch axis {
	case 0:
		return (channel*size+center)*size + right
	case 1:
		return (channel*size+left)*size + right
	default:
		return (channel*size+left)*size + center
	}
}

func packTensorAxis(packed, canonical []float64, size, channels, axis int) {
	rightHandSides := channels * size * size
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					coordinate := [...]int{left, center, right}[axis]
					line := tensorLine(channel, left, center, right, size, axis)
					packed[coordinate*rightHandSides+line] = canonical[tensorIndex(channel, left, center, right, size)]
				}
			}
		}
	}
}

func unpackTensorAxis(canonical, packed []float64, size, channels, axis int) {
	rightHandSides := channels * size * size
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					coordinate := [...]int{left, center, right}[axis]
					line := tensorLine(channel, left, center, right, size, axis)
					canonical[tensorIndex(channel, left, center, right, size)] = packed[coordinate*rightHandSides+line]
				}
			}
		}
	}
}

func transportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int) {
	size := system.Size
	rightHandSides := channels * size * size
	packTensorAxis(scratch, source, size, channels, 0)
	solveTridiagonalBatch(dst, scratch, system, rightHandSides)
	unpackTensorAxis(scratch, dst, size, channels, 0)
	packTensorAxis(dst, scratch, size, channels, 1)
	solveTridiagonalBatch(scratch, dst, system, rightHandSides)
	unpackTensorAxis(dst, scratch, size, channels, 1)
	packTensorAxis(scratch, dst, size, channels, 2)
	solveTridiagonalBatch(dst, scratch, system, rightHandSides)
	unpackTensorAxis(scratch, dst, size, channels, 2)
	copy(dst, scratch)
}

func (b pureGoBackend) ApplyTridiagonal(dst, source []float64, system *TridiagonalSystem) {
	solveTridiagonalBatch(dst, source, system, 1)
}

func (b pureGoBackend) TransportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem) {
	transportTridiagonalMatrix(dst, scratch, source, system)
}

func (b pureGoBackend) TransportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int) {
	transportTridiagonalTensor3(dst, scratch, source, system, channels)
}
