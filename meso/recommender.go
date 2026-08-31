package meso

import (
	"fmt"
	"math"
	"smp-meso/config"
	"smp-meso/numerics"
	"strings"
)

func candidateFallback(state *State, source int) []float64 {
	row := append([]float64(nil), state.Candidate[source*state.Bins:(source+1)*state.Bins]...)
	numerics.NormalizeInPlace(row, state.Rho)
	return row
}

func expectedOpinionWeight(base float64, recommender config.RecommenderConfig) float64 {
	if recommender.NoiseStd <= 0 {
		return math.Pow(math.Max(base, 0), recommender.Steepness)
	}
	total := 0.0
	for index := 0; index < recommender.NoiseQuadraturePoints; index++ {
		noise := numerics.QuantileNormal(index, recommender.NoiseQuadraturePoints) * recommender.NoiseStd
		value := math.Max(base*(1-2*noise)+noise, 0)
		total += math.Pow(value, recommender.Steepness)
	}
	return total / float64(recommender.NoiseQuadraturePoints)
}

// RecommendationKernel returns a row-stochastic endpoint-bin kernel. It is
// the exact projection of the retained C/S_zeta or Xi coordinates; finite-list
// dependence beyond the retained histogram remains a closure.
func RecommendationKernel(state *State, request config.RunRequest) ([]float64, error) {
	result := make([]float64, state.Bins*state.Bins)
	recommenderType := strings.ToLower(strings.TrimSpace(request.Recommender.Type))
	for source := 0; source < state.Bins; source++ {
		fallback := candidateFallback(state, source)
		row := result[source*state.Bins : (source+1)*state.Bins]
		switch recommenderType {
		case "random":
			copy(row, fallback)
		case "opinion_random":
			for target := 0; target < state.Bins; target++ {
				difference := math.Abs(state.Axis[source] - state.Axis[target])
				base := 0.0
				if request.Recommender.OpinionTolerance > 0 {
					base = math.Max(1-difference/request.Recommender.OpinionTolerance, 0)
				} else if difference == 0 {
					base = 1
				}
				row[target] = state.Candidate[state.matrixIndex(source, target)] *
					expectedOpinionWeight(base, request.Recommender)
			}
			numerics.NormalizeInPlace(row, fallback)
			for target := range row {
				row[target] = (1-request.Recommender.RandomRatio)*row[target] +
					request.Recommender.RandomRatio*fallback[target]
			}
		case "structure_random":
			copy(row, state.Score[source*state.Bins:(source+1)*state.Bins])
			numerics.NormalizeInPlace(row, fallback)
			for target := range row {
				row[target] = (1-request.Recommender.RandomRatio)*row[target] +
					request.Recommender.RandomRatio*fallback[target]
			}
		default:
			return nil, fmt.Errorf("unsupported recommender %q", request.Recommender.Type)
		}
		numerics.NormalizeInPlace(row, fallback)
	}
	return result, nil
}

func conditionalMoments(probabilities, axis []float64, mask []bool) (mass, mean, variance float64) {
	for i, probability := range probabilities {
		if mask == nil || mask[i] {
			mass += probability
			mean += probability * axis[i]
		}
	}
	if mass <= numerics.ProbabilityEpsilon {
		return 0, 0, 0
	}
	mean /= mass
	for i, probability := range probabilities {
		if mask == nil || mask[i] {
			difference := axis[i] - mean
			variance += probability * difference * difference
		}
	}
	variance /= mass
	return numerics.Clamp(mass, 0, 1), mean, math.Max(variance, 0)
}
