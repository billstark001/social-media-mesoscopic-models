package kinetic

import "math"

type exposureChannel struct {
	Kernel             []float64
	ConcordantMass     []float64
	MeanDisplacement   []float64
	SecondDisplacement []float64
}

type fields struct {
	Neighbors       exposureChannel
	Recommendations exposureChannel
	RewiringFlux    []float64
}

func conditionalNeighbors(current *state) []float64 {
	size := current.request.OpinionBins
	result := make([]float64, size*size)
	degree := float64(current.request.OutDegree)
	for source := 0; source < size; source++ {
		row := result[source*size : (source+1)*size]
		denominator := degree * current.Rho[source]
		if denominator > 1e-15 {
			for target := range row {
				row[target] = current.Edge[source*size+target] / denominator
			}
		}
		normalizeRow(row, current.Rho)
	}
	return result
}

func channelMoments(current *state, kernel []float64) exposureChannel {
	size := current.request.OpinionBins
	result := exposureChannel{
		Kernel: kernel, ConcordantMass: make([]float64, size),
		MeanDisplacement: make([]float64, size), SecondDisplacement: make([]float64, size),
	}
	for source := 0; source < size; source++ {
		mass, first, second := 0.0, 0.0, 0.0
		for target := 0; target < size; target++ {
			pair := source*size + target
			weight := kernel[pair]
			mass += current.grid.Concordance[pair] * weight
			first += current.grid.Displacement[pair] * weight
			second += current.grid.DisplacementSecond[pair] * weight
		}
		result.ConcordantMass[source] = clamp(mass, 0, 1)
		if mass > 1e-15 {
			result.MeanDisplacement[source] = first / mass
			result.SecondDisplacement[source] = second / mass
		}
	}
	return result
}

func computeFields(current *state, recommend recommenderPlan, structuralScore []float64) fields {
	size := current.request.OpinionBins
	neighbors := conditionalNeighbors(current)
	recommendations := recommend(current, neighbors, structuralScore)
	neighborChannel := channelMoments(current, neighbors)
	recommendationChannel := channelMoments(current, recommendations)
	flux := make([]float64, size*size)
	degree := current.request.OutDegree
	recommendationCount := current.request.RecommendationCount
	for source := 0; source < size; source++ {
		discordantProbability := clamp(1-neighborChannel.ConcordantMass[source], 0, 1)
		concordantRecommendation := recommendationChannel.ConcordantMass[source]
		eligibility := (1 - math.Pow(1-discordantProbability, float64(degree))) *
			(1 - math.Pow(1-concordantRecommendation, float64(recommendationCount)))
		eventRate := current.request.Dynamics.RewiringRate * current.Rho[source] * eligibility
		for target := 0; target < size; target++ {
			pair := source*size + target
			loss, gain := 0.0, 0.0
			if discordantProbability > 1e-15 {
				loss = (1 - current.grid.Concordance[pair]) * neighbors[pair] / discordantProbability
			}
			if concordantRecommendation > 1e-15 {
				gain = current.grid.Concordance[pair] * recommendations[pair] / concordantRecommendation
			}
			flux[pair] = eventRate * (gain - loss)
		}
	}
	return fields{Neighbors: neighborChannel, Recommendations: recommendationChannel, RewiringFlux: flux}
}
