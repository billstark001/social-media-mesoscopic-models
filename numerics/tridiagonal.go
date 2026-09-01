package numerics

import (
	"errors"
	"fmt"
	"math"
)

type tridiagonalFactor interface {
	Solve(dst, source []float64, rightHandSides int) error
}

// TridiagonalSystem stores one immutable factorization. Repeated vector,
// matrix, and tensor transports reuse it instead of copying and refactoring
// the diagonals for each axis.
type TridiagonalSystem struct {
	Size   int
	factor tridiagonalFactor
}

func NewTridiagonalSystem(lower, diagonal, upper []float64) (*TridiagonalSystem, error) {
	return ActiveBackend.FactorTridiagonal(lower, diagonal, upper)
}

type pureTridiagonalFactor struct {
	size            int
	lower, diagonal []float64
	upper, upper2   []float64
	pivots          []int
}

func validateTridiagonal(lower, diagonal, upper []float64) error {
	if len(diagonal) == 0 {
		return errors.New("tridiagonal system must not be empty")
	}
	if len(lower) != len(diagonal)-1 || len(upper) != len(diagonal)-1 {
		return fmt.Errorf("tridiagonal dimensions lower=%d diagonal=%d upper=%d", len(lower), len(diagonal), len(upper))
	}
	for _, values := range [][]float64{lower, diagonal, upper} {
		for _, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return errors.New("tridiagonal coefficients must be finite")
			}
		}
	}
	return nil
}

func factorPureTridiagonal(lower, diagonal, upper []float64) (*pureTridiagonalFactor, error) {
	if err := validateTridiagonal(lower, diagonal, upper); err != nil {
		return nil, err
	}
	n := len(diagonal)
	factor := &pureTridiagonalFactor{
		size: n, lower: append([]float64(nil), lower...),
		diagonal: append([]float64(nil), diagonal...), upper: append([]float64(nil), upper...),
		upper2: make([]float64, max(n-2, 0)), pivots: make([]int, n),
	}
	for index := range factor.pivots {
		factor.pivots[index] = index
	}
	for index := 0; index < n-1; index++ {
		if math.Abs(factor.diagonal[index]) >= math.Abs(factor.lower[index]) {
			if factor.diagonal[index] != 0 {
				multiplier := factor.lower[index] / factor.diagonal[index]
				factor.lower[index] = multiplier
				factor.diagonal[index+1] -= multiplier * factor.upper[index]
			}
			continue
		}
		multiplier := factor.diagonal[index] / factor.lower[index]
		factor.diagonal[index] = factor.lower[index]
		factor.lower[index] = multiplier
		temporary := factor.upper[index]
		factor.upper[index] = factor.diagonal[index+1]
		factor.diagonal[index+1] = temporary - multiplier*factor.diagonal[index+1]
		factor.pivots[index] = index + 1
		if index < n-2 {
			factor.upper2[index] = factor.upper[index+1]
			factor.upper[index+1] = -multiplier * factor.upper[index+1]
		}
	}
	for index, pivot := range factor.diagonal {
		if pivot == 0 || math.IsNaN(pivot) || math.IsInf(pivot, 0) {
			return nil, fmt.Errorf("singular tridiagonal system at pivot %d", index)
		}
	}
	return factor, nil
}

func (factor *pureTridiagonalFactor) Solve(dst, source []float64, rightHandSides int) error {
	expected, err := CheckedProduct("tridiagonal right-hand sides", factor.size, rightHandSides)
	if err != nil {
		return err
	}
	if len(source) != expected || len(dst) != expected {
		return fmt.Errorf("tridiagonal batch length source=%d destination=%d, expected %d", len(source), len(dst), expected)
	}
	copy(dst, source)
	for row := 0; row < factor.size-1; row++ {
		if factor.pivots[row] == row {
			for column := 0; column < rightHandSides; column++ {
				offset := column * factor.size
				dst[offset+row+1] -= factor.lower[row] * dst[offset+row]
			}
		} else {
			for column := 0; column < rightHandSides; column++ {
				offset := column * factor.size
				temporary := dst[offset+row]
				dst[offset+row] = dst[offset+row+1]
				dst[offset+row+1] = temporary - factor.lower[row]*dst[offset+row]
			}
		}
	}
	last := factor.size - 1
	for column := 0; column < rightHandSides; column++ {
		offset := column * factor.size
		dst[offset+last] /= factor.diagonal[last]
	}
	if factor.size > 1 {
		row := factor.size - 2
		for column := 0; column < rightHandSides; column++ {
			offset := column * factor.size
			dst[offset+row] = (dst[offset+row] -
				factor.upper[row]*dst[offset+row+1]) / factor.diagonal[row]
		}
	}
	for row := factor.size - 3; row >= 0; row-- {
		for column := 0; column < rightHandSides; column++ {
			offset := column * factor.size
			dst[offset+row] = (dst[offset+row] -
				factor.upper[row]*dst[offset+row+1] -
				factor.upper2[row]*dst[offset+row+2]) / factor.diagonal[row]
		}
	}
	return nil
}

func (b pureGoBackend) FactorTridiagonal(lower, diagonal, upper []float64) (*TridiagonalSystem, error) {
	factor, err := factorPureTridiagonal(lower, diagonal, upper)
	if err != nil {
		return nil, err
	}
	return &TridiagonalSystem{Size: len(diagonal), factor: factor}, nil
}

func solveTridiagonalBatch(dst, source []float64, system *TridiagonalSystem, rightHandSides int) error {
	if system == nil || system.factor == nil {
		return errors.New("tridiagonal system is not initialized")
	}
	return system.factor.Solve(dst, source, rightHandSides)
}

func transportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem) error {
	size := system.Size
	// Pack columns as contiguous right-hand sides for LAPACK's native
	// column-major representation.
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			scratch[right*size+left] = source[left*size+right]
		}
	}
	if err := solveTridiagonalBatch(dst, scratch, system, size); err != nil {
		return err
	}
	for right := 0; right < size; right++ {
		for left := 0; left < size; left++ {
			scratch[left*size+right] = dst[right*size+left]
		}
	}
	// Each canonical row is now one contiguous RHS, yielding right
	// multiplication by the transposed inverse.
	return solveTridiagonalBatch(dst, scratch, system, size)
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
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					coordinate := [...]int{left, center, right}[axis]
					line := tensorLine(channel, left, center, right, size, axis)
					packed[line*size+coordinate] = canonical[tensorIndex(channel, left, center, right, size)]
				}
			}
		}
	}
}

func unpackTensorAxis(canonical, packed []float64, size, channels, axis int) {
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					coordinate := [...]int{left, center, right}[axis]
					line := tensorLine(channel, left, center, right, size, axis)
					canonical[tensorIndex(channel, left, center, right, size)] = packed[line*size+coordinate]
				}
			}
		}
	}
}

func transportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int) error {
	size := system.Size
	rightHandSides := channels * size * size
	packTensorAxis(scratch, source, size, channels, 0)
	if err := solveTridiagonalBatch(dst, scratch, system, rightHandSides); err != nil {
		return err
	}
	unpackTensorAxis(scratch, dst, size, channels, 0)
	packTensorAxis(dst, scratch, size, channels, 1)
	if err := solveTridiagonalBatch(scratch, dst, system, rightHandSides); err != nil {
		return err
	}
	unpackTensorAxis(dst, scratch, size, channels, 1)
	packTensorAxis(scratch, dst, size, channels, 2)
	if err := solveTridiagonalBatch(dst, scratch, system, rightHandSides); err != nil {
		return err
	}
	unpackTensorAxis(scratch, dst, size, channels, 2)
	copy(dst, scratch)
	return nil
}

func (b pureGoBackend) ApplyTridiagonal(dst, source []float64, system *TridiagonalSystem) error {
	return solveTridiagonalBatch(dst, source, system, 1)
}

func (b pureGoBackend) TransportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem) error {
	return transportTridiagonalMatrix(dst, scratch, source, system)
}

func (b pureGoBackend) TransportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int) error {
	return transportTridiagonalTensor3(dst, scratch, source, system, channels)
}
