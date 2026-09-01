package kinetic

import (
	"math"
	"smp-meso/numerics"
)

type transitionBuilder func(*state, fields) (*numerics.SparseTransition, error)

func planMeasureTransition(request RunRequest) transitionBuilder {
	var reference func(*state, fields) []float64
	switch normalize(request.Dynamics.Type) {
	case "hk":
		reference = hkReferenceTransition
	case "deffuant":
		reference = deffuantReferenceTransition
	default:
		panic("validated dynamics was not dispatched")
	}
	return func(current *state, values fields) (*numerics.SparseTransition, error) {
		return numerics.DenseToSparse(reference(current, values), current.request.OpinionBins), nil
	}
}

func quadratureNodes(count int) []float64 {
	result := make([]float64, count)
	if count == 1 {
		return result
	}
	mean := 0.0
	for index := range result {
		result[index] = numerics.QuantileNormal(index, count)
		mean += result[index]
	}
	mean /= float64(count)
	variance := 0.0
	for index := range result {
		result[index] -= mean
		variance += result[index] * result[index]
	}
	variance /= float64(count)
	scale := 1 / math.Sqrt(variance)
	for index := range result {
		result[index] *= scale
	}
	return result
}

func depositPoint(row []float64, axis []float64, value, mass float64) {
	if mass <= 0 {
		return
	}
	last := len(axis) - 1
	if value <= axis[0] {
		row[0] += mass
		return
	}
	if value >= axis[last] {
		row[last] += mass
		return
	}
	coordinate := (value - axis[0]) / (axis[1] - axis[0])
	lower := int(math.Floor(coordinate))
	fraction := clamp(coordinate-float64(lower), 0, 1)
	row[lower] += mass * (1 - fraction)
	row[lower+1] += mass * fraction
}

func depositNormal(row []float64, axis, nodes []float64, mean, variance, mass float64) {
	if variance <= 1e-18 || len(nodes) == 1 {
		depositPoint(row, axis, mean, mass)
		return
	}
	standardDeviation := math.Sqrt(variance)
	weight := mass / float64(len(nodes))
	for _, node := range nodes {
		depositPoint(row, axis, mean+standardDeviation*node, weight)
	}
}

func hkReferenceTransition(current *state, values fields) []float64 {
	size := current.request.OpinionBins
	result := make([]float64, size*size)
	nodes := quadratureNodes(current.request.Resolution.OpinionQuadraturePoints)
	degreePMF := make([][]float64, size)
	recommendationPMF := make([][]float64, size)
	for source := 0; source < size; source++ {
		degreePMF[source] = numerics.BinomialPMF(current.request.OutDegree, values.Neighbors.ConcordantMass[source])
		recommendationPMF[source] = numerics.BinomialPMF(current.request.RecommendationCount, values.Recommendations.ConcordantMass[source])
	}
	alpha := current.request.Dt * current.request.Dynamics.Influence
	for source, opinion := range current.grid.Axis {
		row := result[source*size : (source+1)*size]
		neighborMean := values.Neighbors.MeanDisplacement[source]
		recommendationMean := values.Recommendations.MeanDisplacement[source]
		neighborVariance := math.Max(values.Neighbors.SecondDisplacement[source]-neighborMean*neighborMean, 0)
		recommendationVariance := math.Max(values.Recommendations.SecondDisplacement[source]-recommendationMean*recommendationMean, 0)
		for neighborCount, neighborProbability := range degreePMF[source] {
			for recommendationCount, recommendationProbability := range recommendationPMF[source] {
				probability := neighborProbability * recommendationProbability
				totalCount := neighborCount + recommendationCount
				if totalCount == 0 {
					depositPoint(row, current.grid.Axis, opinion, probability)
					continue
				}
				count := float64(totalCount)
				meanDisplacement := (float64(neighborCount)*neighborMean + float64(recommendationCount)*recommendationMean) / count
				variance := (float64(neighborCount)*neighborVariance + float64(recommendationCount)*recommendationVariance) / (count * count)
				depositNormal(row, current.grid.Axis, nodes, opinion+alpha*meanDisplacement, alpha*alpha*variance, probability)
			}
		}
		normalizeRow(row, nil)
	}
	return result
}

func addDeffuantChannel(row []float64, current *state, channel exposureChannel, source int, coefficient float64, nodes []float64) {
	if coefficient <= 0 || channel.ConcordantMass[source] <= 1e-15 {
		return
	}
	size := current.request.OpinionBins
	alpha := current.request.Dt * current.request.Dynamics.Influence
	for target := 0; target < size; target++ {
		pair := source*size + target
		concordance := current.grid.Concordance[pair]
		pairMass := concordance * channel.Kernel[pair] / channel.ConcordantMass[source]
		if pairMass <= 0 {
			continue
		}
		meanDisplacement := current.grid.Displacement[pair] / concordance
		second := current.grid.DisplacementSecond[pair] / concordance
		variance := math.Max(second-meanDisplacement*meanDisplacement, 0)
		depositNormal(row, current.grid.Axis, nodes,
			current.grid.Axis[source]+alpha*meanDisplacement,
			alpha*alpha*variance, coefficient*pairMass)
	}
}

func deffuantReferenceTransition(current *state, values fields) []float64 {
	size := current.request.OpinionBins
	result := make([]float64, size*size)
	nodes := quadratureNodes(current.request.Resolution.OpinionQuadraturePoints)
	for source := 0; source < size; source++ {
		neighborPMF := numerics.BinomialPMF(current.request.OutDegree, values.Neighbors.ConcordantMass[source])
		recommendationPMF := numerics.BinomialPMF(current.request.RecommendationCount, values.Recommendations.ConcordantMass[source])
		stay, neighborCoefficient, recommendationCoefficient := 0.0, 0.0, 0.0
		for neighborCount, neighborProbability := range neighborPMF {
			for recommendationCount, recommendationProbability := range recommendationPMF {
				probability := neighborProbability * recommendationProbability
				totalCount := neighborCount + recommendationCount
				if totalCount == 0 {
					stay += probability
					continue
				}
				neighborCoefficient += probability * float64(neighborCount) / float64(totalCount)
				recommendationCoefficient += probability * float64(recommendationCount) / float64(totalCount)
			}
		}
		row := result[source*size : (source+1)*size]
		depositPoint(row, current.grid.Axis, current.grid.Axis[source], stay)
		addDeffuantChannel(row, current, values.Neighbors, source, neighborCoefficient, nodes)
		addDeffuantChannel(row, current, values.Recommendations, source, recommendationCoefficient, nodes)
		normalizeRow(row, nil)
	}
	return result
}
