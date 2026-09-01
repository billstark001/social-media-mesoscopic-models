package numerics

// Backend owns the model-independent linear algebra kernels. Implementations
// must permit dst to be distinct from every input.
type Backend interface {
	Name() string
	Sandwich(dst, scratch, matrix, transition []float64, size int)
	TransportTensor3(dst, scratch1, scratch2, tensor, transition []float64, size int)
	MultiplyABT(dst, left, right []float64, size int)
	WeightedGram(dst, scratch, matrix, weights []float64, size int)
	ApplyDenseTransitionBatch(dst, source, transition []float64, size, columns int)
	TransportMatrixChannels(dst, scratch, source, transition []float64, size, channels int)
	ApplyTransition(dst, source []float64, transition *SparseTransition)
	SandwichTransition(dst, scratch, matrix []float64, transition *SparseTransition)
	TransportTransitionTensor3(dst, scratch, tensor []float64, transition *SparseTransition, channels int)
	FactorTridiagonal(lower, diagonal, upper []float64) (*TridiagonalSystem, error)
	ApplyTridiagonal(dst, source []float64, system *TridiagonalSystem) error
	TransportTridiagonalMatrix(dst, scratch, source []float64, system *TridiagonalSystem) error
	TransportTridiagonalTensor3(dst, scratch, source []float64, system *TridiagonalSystem, channels int) error
}

type pureGoBackend struct {
	name string
}

func (b pureGoBackend) Name() string { return b.name }

func (b pureGoBackend) MultiplyABT(dst, left, right []float64, size int) {
	clear(dst)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			for inner := 0; inner < size; inner++ {
				dst[row*size+column] += left[row*size+inner] * right[column*size+inner]
			}
		}
	}
}

// WeightedGram computes matrix*diag(weights)*matrix^T.
func (b pureGoBackend) WeightedGram(dst, _ []float64, matrix, weights []float64, size int) {
	clear(dst)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			total := 0.0
			for inner, weight := range weights {
				total += matrix[row*size+inner] * weight * matrix[column*size+inner]
			}
			dst[row*size+column] = total
		}
	}
}

// ApplyDenseTransitionBatch applies transition^T to a row-major size x columns
// matrix, sharing the same transition across every right-hand side.
func (b pureGoBackend) ApplyDenseTransitionBatch(dst, source, transition []float64, size, columns int) {
	clear(dst)
	for from := 0; from < size; from++ {
		input := source[from*columns : (from+1)*columns]
		for to := 0; to < size; to++ {
			weight := transition[from*size+to]
			if weight == 0 {
				continue
			}
			output := dst[to*columns : (to+1)*columns]
			for column, value := range input {
				output[column] += weight * value
			}
		}
	}
}

// TransportMatrixChannels computes T^T*M_c*T for matrices stored as
// source[(left*size+right)*channels+channel].
func (b pureGoBackend) TransportMatrixChannels(dst, scratch, source, transition []float64, size, channels int) {
	clear(dst)
	clear(scratch)
	index := func(left, right, channel int) int { return (left*size+right)*channels + channel }
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			for to := 0; to < size; to++ {
				weight := transition[left*size+to]
				if weight == 0 {
					continue
				}
				for channel := 0; channel < channels; channel++ {
					scratch[index(to, right, channel)] += weight * source[index(left, right, channel)]
				}
			}
		}
	}
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			for to := 0; to < size; to++ {
				weight := transition[right*size+to]
				if weight == 0 {
					continue
				}
				for channel := 0; channel < channels; channel++ {
					dst[index(left, to, channel)] += scratch[index(left, right, channel)] * weight
				}
			}
		}
	}
}

// Sandwich computes transition^T * matrix * transition for row-major square
// matrices. The loop ordering keeps the inner operations contiguous.
func (b pureGoBackend) Sandwich(dst, scratch, matrix, transition []float64, size int) {
	clear(scratch)
	clear(dst)
	for source := 0; source < size; source++ {
		for left := 0; left < size; left++ {
			weight := transition[source*size+left]
			if weight == 0 {
				continue
			}
			matrixRow := matrix[source*size : (source+1)*size]
			outRow := scratch[left*size : (left+1)*size]
			for target, value := range matrixRow {
				outRow[target] += weight * value
			}
		}
	}
	for left := 0; left < size; left++ {
		outRow := dst[left*size : (left+1)*size]
		for middle := 0; middle < size; middle++ {
			value := scratch[left*size+middle]
			if value == 0 {
				continue
			}
			transitionRow := transition[middle*size : (middle+1)*size]
			for right, weight := range transitionRow {
				outRow[right] += value * weight
			}
		}
	}
}

// TransportTensor3 applies the same row-stochastic transition to all three
// opinion labels of tensor[i,r,j].
func (b pureGoBackend) TransportTensor3(dst, scratch1, scratch2, tensor, transition []float64, size int) {
	clear(scratch1)
	clear(scratch2)
	clear(dst)
	index := func(i, r, j int) int { return (i*size+r)*size + j }
	for i := 0; i < size; i++ {
		for a := 0; a < size; a++ {
			weight := transition[i*size+a]
			if weight == 0 {
				continue
			}
			for r := 0; r < size; r++ {
				input := tensor[index(i, r, 0) : index(i, r, 0)+size]
				output := scratch1[index(a, r, 0) : index(a, r, 0)+size]
				for j, value := range input {
					output[j] += weight * value
				}
			}
		}
	}
	for a := 0; a < size; a++ {
		for r := 0; r < size; r++ {
			for middle := 0; middle < size; middle++ {
				weight := transition[r*size+middle]
				if weight == 0 {
					continue
				}
				input := scratch1[index(a, r, 0) : index(a, r, 0)+size]
				output := scratch2[index(a, middle, 0) : index(a, middle, 0)+size]
				for j, value := range input {
					output[j] += weight * value
				}
			}
		}
	}
	for a := 0; a < size; a++ {
		for middle := 0; middle < size; middle++ {
			output := dst[index(a, middle, 0) : index(a, middle, 0)+size]
			for j := 0; j < size; j++ {
				value := scratch2[index(a, middle, j)]
				if value == 0 {
					continue
				}
				transitionRow := transition[j*size : (j+1)*size]
				for c, weight := range transitionRow {
					output[c] += value * weight
				}
			}
		}
	}
}

var ActiveBackend Backend = newDenseBackend()
