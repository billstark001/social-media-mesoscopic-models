package kinetic

import "smp-meso/numerics"

type incrementMoments struct {
	Mean   []float64
	Second []float64
}

type momentBuilder func(*state, fields) incrementMoments

func planMomentBuilder(request RunRequest) momentBuilder {
	switch normalize(request.Dynamics.Type) {
	case "hk":
		return hkIncrementMoments
	case "deffuant":
		return deffuantIncrementMoments
	default:
		panic("validated dynamics was not dispatched")
	}
}

// hkIncrementMoments retains both sources of finite-exposure randomness:
// random concordant counts and the variance of each finite sample mean.
func hkIncrementMoments(current *state, values fields) incrementMoments {
	size := current.request.OpinionBins
	result := incrementMoments{Mean: make([]float64, size), Second: make([]float64, size)}
	for source := 0; source < size; source++ {
		neighborPMF := numerics.BinomialPMF(current.request.OutDegree, values.Neighbors.ConcordantMass[source])
		recommendationPMF := numerics.BinomialPMF(current.request.RecommendationCount, values.Recommendations.ConcordantMass[source])
		neighborMean := values.Neighbors.MeanDisplacement[source]
		recommendationMean := values.Recommendations.MeanDisplacement[source]
		neighborVariance := max(values.Neighbors.SecondDisplacement[source]-neighborMean*neighborMean, 0)
		recommendationVariance := max(values.Recommendations.SecondDisplacement[source]-recommendationMean*recommendationMean, 0)
		for neighborCount, neighborProbability := range neighborPMF {
			for recommendationCount, recommendationProbability := range recommendationPMF {
				totalCount := neighborCount + recommendationCount
				if totalCount == 0 {
					continue
				}
				probability := neighborProbability * recommendationProbability
				count := float64(totalCount)
				mean := (float64(neighborCount)*neighborMean + float64(recommendationCount)*recommendationMean) / count
				variance := (float64(neighborCount)*neighborVariance + float64(recommendationCount)*recommendationVariance) / (count * count)
				result.Mean[source] += probability * mean
				result.Second[source] += probability * (variance + mean*mean)
			}
		}
	}
	return result
}

// deffuantIncrementMoments chooses one item uniformly from the finite set of
// concordant visible items, preserving the C=0 no-update event.
func deffuantIncrementMoments(current *state, values fields) incrementMoments {
	size := current.request.OpinionBins
	result := incrementMoments{Mean: make([]float64, size), Second: make([]float64, size)}
	for source := 0; source < size; source++ {
		neighborPMF := numerics.BinomialPMF(current.request.OutDegree, values.Neighbors.ConcordantMass[source])
		recommendationPMF := numerics.BinomialPMF(current.request.RecommendationCount, values.Recommendations.ConcordantMass[source])
		neighborCoefficient, recommendationCoefficient := 0.0, 0.0
		for neighborCount, neighborProbability := range neighborPMF {
			for recommendationCount, recommendationProbability := range recommendationPMF {
				totalCount := neighborCount + recommendationCount
				if totalCount == 0 {
					continue
				}
				probability := neighborProbability * recommendationProbability
				neighborCoefficient += probability * float64(neighborCount) / float64(totalCount)
				recommendationCoefficient += probability * float64(recommendationCount) / float64(totalCount)
			}
		}
		result.Mean[source] = neighborCoefficient*values.Neighbors.MeanDisplacement[source] +
			recommendationCoefficient*values.Recommendations.MeanDisplacement[source]
		result.Second[source] = neighborCoefficient*values.Neighbors.SecondDisplacement[source] +
			recommendationCoefficient*values.Recommendations.SecondDisplacement[source]
	}
	return result
}
