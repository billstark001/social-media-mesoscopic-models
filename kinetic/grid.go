package kinetic

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type gridGeometry struct {
	Axis               []float64
	Dx                 float64
	Concordance        []float64
	Displacement       []float64
	DisplacementSecond []float64
}

func gaussLegendre(count int) ([]float64, []float64) {
	nodes := make([]float64, count)
	weights := make([]float64, count)
	for index := 0; index < (count+1)/2; index++ {
		root := math.Cos(math.Pi * (float64(index) + 0.75) / (float64(count) + 0.5))
		derivative := 0.0
		for iteration := 0; iteration < 64; iteration++ {
			previous, polynomial := 1.0, root
			if count == 1 {
				previous = 1
			}
			for degree := 2; degree <= count; degree++ {
				next := ((2*float64(degree)-1)*root*polynomial - (float64(degree)-1)*previous) / float64(degree)
				previous, polynomial = polynomial, next
			}
			derivative = float64(count) * (root*polynomial - previous) / (root*root - 1)
			nextRoot := root - polynomial/derivative
			if math.Abs(nextRoot-root) <= 2e-16 {
				root = nextRoot
				break
			}
			root = nextRoot
		}
		weight := 2 / ((1 - root*root) * derivative * derivative)
		nodes[index], nodes[count-1-index] = -root, root
		weights[index], weights[count-1-index] = weight, weight
	}
	return nodes, weights
}

func uniqueSorted(values []float64) []float64 {
	sort.Float64s(values)
	result := values[:0]
	for _, value := range values {
		if len(result) == 0 || math.Abs(value-result[len(result)-1]) > 1e-15 {
			result = append(result, value)
		}
	}
	return result
}

func newGrid(request RunRequest) (*gridGeometry, error) {
	size := request.OpinionBins
	dx := (request.Initial.OpinionMax - request.Initial.OpinionMin) / float64(size)
	axis := make([]float64, size)
	for index := range axis {
		axis[index] = request.Initial.OpinionMin + (float64(index)+0.5)*dx
	}
	result := &gridGeometry{
		Axis: axis, Dx: dx,
		Concordance:        make([]float64, size*size),
		Displacement:       make([]float64, size*size),
		DisplacementSecond: make([]float64, size*size),
	}
	mode := strings.ToLower(strings.TrimSpace(request.ConfidenceMode))
	if mode == "center" {
		for source := 0; source < size; source++ {
			for target := 0; target < size; target++ {
				difference := axis[target] - axis[source]
				if math.Abs(difference) <= request.Dynamics.Tolerance {
					index := source*size + target
					result.Concordance[index] = 1
					result.Displacement[index] = difference
					result.DisplacementSecond[index] = difference * difference
				}
			}
		}
		return result, nil
	}
	if mode != "cell_average" {
		return nil, fmt.Errorf("unsupported confidence mode %q", request.ConfidenceMode)
	}
	lower := make([]float64, size)
	upper := make([]float64, size)
	for index := range axis {
		lower[index] = axis[index] - 0.5*dx
		upper[index] = axis[index] + 0.5*dx
	}
	epsilon := request.Dynamics.Tolerance
	quadratureNodes, quadratureWeights := gaussLegendre(request.Resolution.ConfidenceQuadraturePoints)
	for source := 0; source < size; source++ {
		for target := 0; target < size; target++ {
			breakpoints := []float64{lower[source], upper[source]}
			for _, point := range []float64{
				lower[target] - epsilon,
				upper[target] - epsilon,
				lower[target] + epsilon,
				upper[target] + epsilon,
			} {
				if lower[source] < point && point < upper[source] {
					breakpoints = append(breakpoints, point)
				}
			}
			breakpoints = uniqueSorted(breakpoints)
			pair := source*size + target
			for segment := 0; segment+1 < len(breakpoints); segment++ {
				left, right := breakpoints[segment], breakpoints[segment+1]
				half := 0.5 * (right - left)
				midpoint := 0.5 * (right + left)
				for quadrature := range quadratureNodes {
					x := midpoint + half*quadratureNodes[quadrature]
					clippedLower := math.Max(lower[target], x-epsilon)
					clippedUpper := math.Min(upper[target], x+epsilon)
					if clippedUpper <= clippedLower {
						continue
					}
					weight := half * quadratureWeights[quadrature] / (dx * dx)
					length := clippedUpper - clippedLower
					result.Concordance[pair] += weight * length
					result.Displacement[pair] += weight * (0.5*(clippedUpper*clippedUpper-clippedLower*clippedLower) - x*length)
					result.DisplacementSecond[pair] += weight *
						(math.Pow(clippedUpper-x, 3) - math.Pow(clippedLower-x, 3)) / 3
				}
			}
			result.Concordance[pair] = math.Min(math.Max(result.Concordance[pair], 0), 1)
		}
	}
	return result, nil
}
