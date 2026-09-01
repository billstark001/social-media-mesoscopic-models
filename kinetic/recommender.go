package kinetic

import (
	"math"
	"smp-meso/numerics"
)

type recommenderPlan func(*state, []float64, []float64) []float64

func planRecommender(request RunRequest) recommenderPlan {
	switch normalize(request.Recommender.Type) {
	case "random":
		return randomRecommendations
	case "opinion_random":
		return opinionRecommendations
	case "structure_random_l0":
		return structureL0Recommendations
	case "structure_random_l1":
		return structureL1Recommendations
	default:
		panic("validated recommender was not dispatched")
	}
}

func randomKernel(rho []float64, size int) []float64 {
	result := make([]float64, size*size)
	for source := 0; source < size; source++ {
		copy(result[source*size:(source+1)*size], rho)
	}
	return result
}

func randomRecommendations(current *state, _ []float64, _ []float64) []float64 {
	return randomKernel(current.Rho, current.request.OpinionBins)
}

func blendRecommendationRows(weighted, random []float64, ratio float64, size int) {
	for source := 0; source < size; source++ {
		row := weighted[source*size : (source+1)*size]
		normalizeRow(row, random[source*size:(source+1)*size])
		for target := range row {
			row[target] = (1-ratio)*row[target] + ratio*random[source*size+target]
		}
	}
}

func opinionRecommendations(current *state, _ []float64, _ []float64) []float64 {
	size := current.request.OpinionBins
	random := randomKernel(current.Rho, size)
	result := make([]float64, size*size)
	tolerance := current.request.Recommender.OpinionTolerance
	steepness := current.request.Recommender.Steepness
	for source, x := range current.grid.Axis {
		for target, y := range current.grid.Axis {
			score := math.Max(1-math.Abs(y-x)/tolerance, 0)
			result[source*size+target] = math.Pow(score, steepness) * current.Rho[target]
		}
	}
	blendRecommendationRows(result, random, current.request.Recommender.RandomRatio, size)
	return result
}

func structureL0Recommendations(current *state, neighbors, _ []float64) []float64 {
	size := current.request.OpinionBins
	random := randomKernel(current.Rho, size)
	result := make([]float64, size*size)
	steepness := current.request.Recommender.Steepness
	overlap := make([]float64, size*size)
	numerics.ActiveBackend.MultiplyABT(overlap, neighbors, neighbors, size)
	for source := 0; source < size; source++ {
		for target := 0; target < size; target++ {
			score := overlap[source*size+target]
			result[source*size+target] = math.Pow(math.Max(score, 0), steepness) * current.Rho[target]
		}
	}
	blendRecommendationRows(result, random, current.request.Recommender.RandomRatio, size)
	return result
}

// cappedPoissonPower returns E[min(C, scoreMax)^steepness] for a Poisson
// common-neighbor count. It is exact at steepness=1 as scoreMax grows and,
// unlike a power of the mean, retains score-count variance.
func cappedPoissonPower(mean, steepness float64, scoreMax int) float64 {
	if mean <= 0 {
		return 0
	}
	probability := math.Exp(-mean)
	cumulative := probability
	moment := 0.0
	for score := 1; score < scoreMax; score++ {
		probability *= mean / float64(score)
		cumulative += probability
		moment += probability * math.Pow(float64(score), steepness)
	}
	tail := math.Max(1-cumulative, 0)
	return moment + tail*math.Pow(float64(scoreMax), steepness)
}

func structureL1Recommendations(current *state, _ []float64, structuralScore []float64) []float64 {
	size := current.request.OpinionBins
	random := randomKernel(current.Rho, size)
	result := make([]float64, size*size)
	for source := 0; source < size; source++ {
		for target := 0; target < size; target++ {
			candidateMass := current.Rho[source] * current.Rho[target]
			if candidateMass > 1e-15 {
				// Wedge masses are ordered-wedge counts per agent, whereas a
				// source/target opinion block contains N*rho_i*rho_j candidate
				// pairs per agent. Dividing by that finite-population candidate
				// mass gives the mean integer common-neighbor score of one pair.
				mean := math.Max(
					structuralScore[source*size+target]/
						(float64(current.request.Population)*candidateMass),
					0,
				)
				result[source*size+target] = candidateMass * cappedPoissonPower(
					mean, current.request.Recommender.Steepness, current.request.Resolution.ScoreMax,
				)
			}
		}
	}
	blendRecommendationRows(result, random, current.request.Recommender.RandomRatio, size)
	return result
}
