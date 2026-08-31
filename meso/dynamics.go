package meso

import (
	"fmt"
	"math"
	"smp-meso/config"
	"smp-meso/numerics"
	"strings"
)

func depositNormal(row, axis []float64, mean, variance, weight float64, points int) {
	if weight <= 0 {
		return
	}
	if variance <= 1e-16 || points == 1 {
		numerics.DepositLinear(row, axis, mean, weight)
		return
	}
	standardDeviation := math.Sqrt(variance)
	share := weight / float64(points)
	for index := 0; index < points; index++ {
		value := mean + standardDeviation*numerics.QuantileNormal(index, points)
		numerics.DepositLinear(row, axis, value, share)
	}
}

func concordanceMask(axis []float64, center, tolerance float64) []bool {
	result := make([]bool, len(axis))
	for i, value := range axis {
		result[i] = math.Abs(value-center) <= tolerance
	}
	return result
}

func hkTransitionRow(
	state *State,
	request config.RunRequest,
	source int,
	neighborRow, recommendationRow, output []float64,
) {
	mask := concordanceMask(state.Axis, state.Axis[source], request.Dynamics.Tolerance)
	pN, meanN, varianceN := conditionalMoments(neighborRow, state.Axis, mask)
	pR, meanR, varianceR := conditionalMoments(recommendationRow, state.Axis, mask)
	pmfN := numerics.BinomialPMF(state.Degree, pN)
	pmfR := numerics.BinomialPMF(state.Recommendations, pR)
	availableConcordant := 0.0
	if state.Rho[source] > numerics.ProbabilityEpsilon {
		for target, concordant := range mask {
			if concordant {
				availableConcordant += state.Candidate[state.matrixIndex(source, target)] / state.Rho[source]
			}
		}
	}
	for countN, probabilityN := range pmfN {
		if probabilityN < 1e-15 {
			continue
		}
		for countR, probabilityR := range pmfR {
			weight := probabilityN * probabilityR
			if weight < 1e-15 {
				continue
			}
			count := countN + countR
			if count == 0 {
				output[source] += weight
				continue
			}
			targetMean := (float64(countN)*meanN + float64(countR)*meanR) / float64(count)
			finitePopulation := 1.0
			if countR > 0 && availableConcordant > 1 {
				finitePopulation = math.Max((availableConcordant-float64(countR))/(availableConcordant-1), 0)
			}
			averageVariance := (float64(countN)*varianceN +
				float64(countR)*varianceR*finitePopulation) / float64(count*count)
			destinationMean := (1-request.Dynamics.Influence)*state.Axis[source] +
				request.Dynamics.Influence*targetMean
			destinationVariance := request.Dynamics.Influence * request.Dynamics.Influence * averageVariance
			depositNormal(output, state.Axis, destinationMean, destinationVariance, weight,
				request.Resolution.OpinionQuadrature)
		}
	}
}

func deffuantTransitionRow(
	state *State,
	request config.RunRequest,
	source int,
	neighborRow, recommendationRow, output []float64,
) {
	mask := concordanceMask(state.Axis, state.Axis[source], request.Dynamics.Tolerance)
	pN, _, _ := conditionalMoments(neighborRow, state.Axis, mask)
	pR, _, _ := conditionalMoments(recommendationRow, state.Axis, mask)
	pmfN := numerics.BinomialPMF(state.Degree, pN)
	pmfR := numerics.BinomialPMF(state.Recommendations, pR)
	for countN, probabilityN := range pmfN {
		for countR, probabilityR := range pmfR {
			weight := probabilityN * probabilityR
			count := countN + countR
			if weight < 1e-15 {
				continue
			}
			if count == 0 {
				output[source] += weight
				continue
			}
			if countN > 0 && pN > numerics.ProbabilityEpsilon {
				typeWeight := weight * float64(countN) / float64(count)
				for target, probability := range neighborRow {
					if !mask[target] || probability <= 0 {
						continue
					}
					destination := state.Axis[source] + request.Dynamics.Influence*(state.Axis[target]-state.Axis[source])
					numerics.DepositLinear(output, state.Axis, destination, typeWeight*probability/pN)
				}
			}
			if countR > 0 && pR > numerics.ProbabilityEpsilon {
				typeWeight := weight * float64(countR) / float64(count)
				for target, probability := range recommendationRow {
					if !mask[target] || probability <= 0 {
						continue
					}
					destination := state.Axis[source] + request.Dynamics.Influence*(state.Axis[target]-state.Axis[source])
					numerics.DepositLinear(output, state.Axis, destination, typeWeight*probability/pR)
				}
			}
		}
	}
}

// TransitionKernel constructs the synchronous node-bin transition law.
func TransitionKernel(state *State, request config.RunRequest, neighbors, recommendations []float64) ([]float64, error) {
	transition := make([]float64, state.Bins*state.Bins)
	for source := 0; source < state.Bins; source++ {
		neighborRow := neighbors[source*state.Bins : (source+1)*state.Bins]
		recommendationRow := recommendations[source*state.Bins : (source+1)*state.Bins]
		output := transition[source*state.Bins : (source+1)*state.Bins]
		switch strings.ToLower(strings.TrimSpace(request.Dynamics.Type)) {
		case "hk":
			hkTransitionRow(state, request, source, neighborRow, recommendationRow, output)
		case "deffuant":
			deffuantTransitionRow(state, request, source, neighborRow, recommendationRow, output)
		default:
			return nil, fmt.Errorf("unsupported dynamics %q", request.Dynamics.Type)
		}
		fallback := make([]float64, state.Bins)
		fallback[source] = 1
		numerics.NormalizeInPlace(output, fallback)
	}
	return transition, nil
}
