package numerics

import (
	"errors"
	"math"
	"math/rand/v2"
)

const ProbabilityEpsilon = 1e-14

func Clamp(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func Sum(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total
}

func NormalizedCopy(values []float64) ([]float64, error) {
	result := append([]float64(nil), values...)
	total := 0.0
	for _, value := range result {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return nil, errors.New("values must be finite and nonnegative")
		}
		total += value
	}
	if total <= 0 || math.IsInf(total, 0) {
		return nil, errors.New("values must have a finite positive sum")
	}
	for i := range result {
		result[i] /= total
	}
	return result, nil
}

func NormalizeInPlace(values []float64, fallback []float64) {
	if len(values) == 0 {
		return
	}
	total := 0.0
	for i, value := range values {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			values[i] = 0
		}
		total += values[i]
	}
	if total <= ProbabilityEpsilon {
		clear(values)
		if len(fallback) == len(values) {
			copy(values, fallback)
		}
		total = Sum(values)
	}
	if total <= ProbabilityEpsilon {
		uniform := 1 / float64(len(values))
		for i := range values {
			values[i] = uniform
		}
		return
	}
	for i := range values {
		values[i] /= total
	}
}

// sampleBinomial uses inverse transform with the smaller of p and 1-p.
// For the finite populations used here it is exact and avoids a dependency on
// an RNG implementation outside math/rand/v2.
func SampleBinomial(n int, p float64, rng *rand.Rand) int {
	if n <= 0 || p <= 0 {
		return 0
	}
	if p >= 1 {
		return n
	}
	complement := false
	if p > 0.5 {
		p = 1 - p
		complement = true
	}
	q := 1 - p
	probability := math.Pow(q, float64(n))
	if probability == 0 {
		pmf := BinomialPMF(n, p)
		u := rng.Float64()
		cumulative := 0.0
		for k, mass := range pmf {
			cumulative += mass
			if u <= cumulative || k == n {
				if complement {
					return n - k
				}
				return k
			}
		}
	}
	u := rng.Float64()
	cumulative := probability
	k := 0
	for u > cumulative && k < n {
		k++
		probability *= float64(n-k+1) / float64(k) * p / q
		cumulative += probability
	}
	if complement {
		return n - k
	}
	return k
}

func BinomialPMF(n int, p float64) []float64 {
	result := make([]float64, n+1)
	if p <= 0 {
		result[0] = 1
		return result
	}
	if p >= 1 {
		result[n] = 1
		return result
	}
	q := 1 - p
	mode := int(math.Floor(float64(n+1) * p))
	result[mode] = 1
	for k := mode; k > 0; k-- {
		result[k-1] = result[k] * float64(k) / float64(n-k+1) * q / p
	}
	for k := mode; k < n; k++ {
		result[k+1] = result[k] * float64(n-k) / float64(k+1) * p / q
	}
	// The recurrence is scaled to one at the mode, avoiding underflow in the
	// extreme tail that would occur when starting from q^n.
	NormalizeInPlace(result, nil)
	return result
}

func SampleMultinomial(n int, probabilities []float64, rng *rand.Rand, output []int) {
	clear(output)
	if n <= 0 {
		return
	}
	remainingN := n
	remainingP := Sum(probabilities)
	for i := 0; i < len(probabilities)-1; i++ {
		if remainingN == 0 {
			return
		}
		p := 0.0
		if remainingP > ProbabilityEpsilon {
			p = Clamp(probabilities[i]/remainingP, 0, 1)
		}
		value := SampleBinomial(remainingN, p, rng)
		output[i] = value
		remainingN -= value
		remainingP -= probabilities[i]
	}
	output[len(probabilities)-1] = remainingN
}

func PoissonPMF(mean float64, maximum int) []float64 {
	result := make([]float64, maximum+1)
	if mean <= 0 {
		result[0] = 1
		return result
	}
	result[0] = math.Exp(-mean)
	total := result[0]
	for k := 1; k < maximum; k++ {
		result[k] = result[k-1] * mean / float64(k)
		total += result[k]
	}
	result[maximum] = math.Max(1-total, 0)
	NormalizeInPlace(result, nil)
	return result
}

func DepositLinear(row, axis []float64, value, weight float64) {
	if weight <= 0 {
		return
	}
	value = Clamp(value, axis[0], axis[len(axis)-1])
	if value <= axis[0] {
		row[0] += weight
		return
	}
	last := len(axis) - 1
	if value >= axis[last] {
		row[last] += weight
		return
	}
	lo, hi := 0, last
	for hi-lo > 1 {
		mid := (lo + hi) / 2
		if axis[mid] <= value {
			lo = mid
		} else {
			hi = mid
		}
	}
	fraction := (value - axis[lo]) / (axis[hi] - axis[lo])
	row[lo] += weight * (1 - fraction)
	row[hi] += weight * fraction
}

func QuantileNormal(index, count int) float64 {
	p := (float64(index) + 0.5) / float64(count)
	return math.Sqrt2 * math.Erfinv(2*p-1)
}

func MixInto(dst, current, target []float64, targetWeight float64) {
	targetWeight = Clamp(targetWeight, 0, 1)
	for i := range dst {
		dst[i] = (1-targetWeight)*current[i] + targetWeight*target[i]
		if dst[i] < 0 && dst[i] > -1e-12 {
			dst[i] = 0
		}
	}
}

func FiniteSlice(values []float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}
