package numerics

import (
	"errors"
	"fmt"
	"math"
)

const sparseTransitionTolerance = 1e-15

type TransitionEntry struct {
	Index  int
	Weight float64
}

// SparseTransition is a row-stochastic source-to-destination operator. Dense
// and Rows are two representations of the same pruned and renormalized
// operator, so choosing a backend never changes the numerical model.
type SparseTransition struct {
	Size     int
	Rows     [][]TransitionEntry
	Dense    []float64
	NonZeros int
}

func DenseToSparse(dense []float64, size int) (*SparseTransition, error) {
	length, err := CheckedProduct("transition matrix", size, size)
	if err != nil {
		return nil, err
	}
	if len(dense) != length {
		return nil, fmt.Errorf("transition matrix length %d, expected %d", len(dense), length)
	}
	rows := make([][]TransitionEntry, size)
	canonical := make([]float64, length)
	nonZeros := 0
	for source := 0; source < size; source++ {
		input := dense[source*size : (source+1)*size]
		total := 0.0
		for target, weight := range input {
			if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
				return nil, fmt.Errorf("invalid transition weight at (%d,%d): %g", source, target, weight)
			}
			if weight > sparseTransitionTolerance {
				total += weight
			}
		}
		if total <= 0 || math.IsInf(total, 0) {
			return nil, fmt.Errorf("transition row %d has no finite positive mass", source)
		}
		inverse := 1 / total
		for target, weight := range input {
			if weight <= sparseTransitionTolerance {
				continue
			}
			weight *= inverse
			rows[source] = append(rows[source], TransitionEntry{Index: target, Weight: weight})
			canonical[source*size+target] = weight
			nonZeros++
		}
	}
	return &SparseTransition{Size: size, Rows: rows, Dense: canonical, NonZeros: nonZeros}, nil
}

func NewSparseTransition(size int, rows [][]TransitionEntry) (*SparseTransition, error) {
	length, err := CheckedProduct("transition matrix", size, size)
	if err != nil {
		return nil, err
	}
	if len(rows) != size {
		return nil, fmt.Errorf("transition row count %d, expected %d", len(rows), size)
	}
	dense := make([]float64, length)
	for source, row := range rows {
		if len(row) == 0 {
			return nil, fmt.Errorf("transition row %d is empty", source)
		}
		for _, entry := range row {
			if entry.Index < 0 || entry.Index >= size {
				return nil, fmt.Errorf("transition target %d outside [0,%d)", entry.Index, size)
			}
			if math.IsNaN(entry.Weight) || math.IsInf(entry.Weight, 0) || entry.Weight < 0 {
				return nil, errors.New("transition weights must be finite and nonnegative")
			}
			dense[source*size+entry.Index] += entry.Weight
		}
	}
	return DenseToSparse(dense, size)
}

func (transition *SparseTransition) preferSparse() bool {
	if transition == nil || transition.Size <= 0 {
		return true
	}
	// Reserve dense BLAS for matrices above 25% density. Valid transitions have
	// already passed the request budget, so this int64 product cannot overflow.
	return int64(transition.NonZeros)*4 <= int64(transition.Size)*int64(transition.Size)
}
