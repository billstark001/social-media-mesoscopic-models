package numerics

func (b pureGoBackend) ApplyTransition(dst, source []float64, transition *SparseTransition) {
	clear(dst)
	for from, mass := range source {
		for _, entry := range transition.Rows[from] {
			dst[entry.Index] += mass * entry.Weight
		}
	}
}

func (b pureGoBackend) SandwichTransition(dst, scratch, matrix []float64, transition *SparseTransition) {
	size := transition.Size
	clear(dst)
	clear(scratch)
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			value := matrix[left*size+right]
			for _, entry := range transition.Rows[left] {
				scratch[entry.Index*size+right] += value * entry.Weight
			}
		}
	}
	for left := 0; left < size; left++ {
		for right := 0; right < size; right++ {
			value := scratch[left*size+right]
			for _, entry := range transition.Rows[right] {
				dst[left*size+entry.Index] += value * entry.Weight
			}
		}
	}
}

func (b pureGoBackend) TransportTransitionTensor3(dst, scratch, tensor []float64, transition *SparseTransition, channels int) {
	size := transition.Size
	block := size * size * size
	clear(dst)
	clear(scratch)
	index := func(channel, left, center, right int) int {
		return channel*block + (left*size+center)*size + right
	}
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					value := tensor[index(channel, left, center, right)]
					for _, entry := range transition.Rows[left] {
						dst[index(channel, entry.Index, center, right)] += value * entry.Weight
					}
				}
			}
		}
	}
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					value := dst[index(channel, left, center, right)]
					for _, entry := range transition.Rows[center] {
						scratch[index(channel, left, entry.Index, right)] += value * entry.Weight
					}
				}
			}
		}
	}
	clear(dst)
	for channel := 0; channel < channels; channel++ {
		for left := 0; left < size; left++ {
			for center := 0; center < size; center++ {
				for right := 0; right < size; right++ {
					value := scratch[index(channel, left, center, right)]
					for _, entry := range transition.Rows[right] {
						dst[index(channel, left, center, entry.Index)] += value * entry.Weight
					}
				}
			}
		}
	}
}
