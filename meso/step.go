package meso

import (
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
)

type StepDiagnostics struct {
	RewiringEvents int
	ExpectedMoves  float64
	MaxRhoChange   float64
}

func correlatedIntersection(left, right, correlation float64) float64 {
	independent := left * right
	if correlation >= 0 {
		return independent + correlation*(math.Min(left, right)-independent)
	}
	lower := math.Max(0, left+right-1)
	return independent + (-correlation)*(lower-independent)
}

func applyComponentAmbiguity(state *State, request config.RunRequest, kernel []float64, componentMix float64) {
	if math.Abs(componentMix) <= numerics.ProbabilityEpsilon {
		return
	}
	for source := 0; source < state.Bins; source++ {
		row := kernel[source*state.Bins : (source+1)*state.Bins]
		fallback := candidateFallback(state, source)
		for target := range row {
			concordant := math.Abs(state.Axis[source]-state.Axis[target]) <= request.Dynamics.Tolerance
			if concordant {
				row[target] *= 1 + componentMix
			} else {
				row[target] *= 1 - componentMix
			}
		}
		numerics.NormalizeInPlace(row, fallback)
	}
}

func inferredBridgeFraction(state *State, source, target int) float64 {
	index := state.matrixIndex(source, target)
	if state.Edge[index] <= numerics.ProbabilityEpsilon {
		return 0
	}
	if state.Layer >= LayerTopology {
		return numerics.Clamp(state.Bridges[index]/state.Edge[index], 0, 1)
	}
	adjacency := state.undirectedAdjacencyProbabilities()
	return 1 / (1 + state.scoreMean(source, target, adjacency))
}

func rewiringEligibility(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
	neighbors, recommendations []float64,
) []float64 {
	result := make([]float64, state.Bins)
	for source := 0; source < state.Bins; source++ {
		if state.Rho[source] <= numerics.ProbabilityEpsilon {
			continue
		}
		if state.Layer >= LayerHistogram {
			result[source] = state.histogramEligibility(source)
		} else {
			pDiscordant := 0.0
			pConcordantRecommendation := 0.0
			for target := 0; target < state.Bins; target++ {
				if math.Abs(state.Axis[source]-state.Axis[target]) <= request.Dynamics.Tolerance {
					pConcordantRecommendation += recommendations[state.matrixIndex(source, target)]
				} else {
					pDiscordant += neighbors[state.matrixIndex(source, target)]
				}
			}
			left := 1 - math.Pow(1-numerics.Clamp(pDiscordant, 0, 1), float64(state.Degree))
			right := 1 - math.Pow(1-numerics.Clamp(pConcordantRecommendation, 0, 1), float64(state.Recommendations))
			result[source] = correlatedIntersection(left, right, profile.EligibilityCorrelation)
		}
		if state.Layer >= LayerTopology {
			smallMass := 0.0
			for sizeBin := 0; sizeBin < state.ComponentSizeBins-1; sizeBin++ {
				smallMass += state.Components[state.componentIndex(source, sizeBin)]
			}
			isolatedFraction := numerics.Clamp(smallMass/state.Rho[source], 0, 1)
			result[source] *= 1 - isolatedFraction*(1-request.Recommender.RandomRatio)
		}
		result[source] = numerics.Clamp(result[source], 0, 1)
	}
	return result
}

func sampleLossWithoutReplacement(counts []int, total int, weights []float64, rng *rand.Rand) []int {
	result := make([]int, len(counts))
	remaining := append([]int(nil), counts...)
	for draw := 0; draw < total; draw++ {
		weightTotal := 0.0
		for i, count := range remaining {
			if count > 0 {
				weightTotal += float64(count) * weights[i]
			}
		}
		uniformFallback := weightTotal <= numerics.ProbabilityEpsilon
		if uniformFallback {
			for _, count := range remaining {
				weightTotal += float64(count)
			}
			if weightTotal <= numerics.ProbabilityEpsilon {
				break
			}
		}
		threshold := rng.Float64() * weightTotal
		accumulator := 0.0
		selected := -1
		for i, count := range remaining {
			if count <= 0 {
				continue
			}
			weight := weights[i]
			if uniformFallback {
				weight = 1
			}
			accumulator += float64(count) * weight
			if threshold < accumulator {
				selected = i
				break
			}
		}
		if selected < 0 {
			break
		}
		remaining[selected]--
		result[selected]++
	}
	return result
}

func rewire(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
	neighbors, recommendations []float64,
	rng *rand.Rand,
) ([]float64, []float64, int) {
	output := append([]float64(nil), state.Edge...)
	eligibility := rewiringEligibility(state, request, profile, neighbors, recommendations)
	totalEvents := 0
	nodeCounts := make([]int, state.Bins)
	for i, mass := range state.Rho {
		nodeCounts[i] = int(math.Round(float64(state.Population) * mass))
	}
	difference := state.Population
	for _, count := range nodeCounts {
		difference -= count
	}
	if difference != 0 {
		best := 0
		for i := range state.Rho {
			if state.Rho[i] > state.Rho[best] {
				best = i
			}
		}
		nodeCounts[best] += difference
	}
	for source, nodeCount := range nodeCounts {
		requested := numerics.SampleBinomial(nodeCount,
			numerics.Clamp(request.Dynamics.RewiringRate*eligibility[source], 0, 1), rng)
		if requested == 0 {
			continue
		}
		available := make([]int, state.Bins)
		lossWeight := make([]float64, state.Bins)
		availableTotal := 0
		gainProbability := make([]float64, state.Bins)
		for target := 0; target < state.Bins; target++ {
			concordant := math.Abs(state.Axis[source]-state.Axis[target]) <= request.Dynamics.Tolerance
			index := state.matrixIndex(source, target)
			if !concordant {
				available[target] = int(math.Floor(float64(state.Population)*math.Max(output[index], 0) + 1e-9))
				availableTotal += available[target]
				bridgeFraction := inferredBridgeFraction(state, source, target)
				lossWeight[target] = math.Max(1+profile.BridgeBias*(2*bridgeFraction-1), 0)
			} else {
				gainProbability[target] = recommendations[index]
			}
		}
		actual := min(requested, availableTotal)
		if actual <= 0 || numerics.Sum(gainProbability) <= numerics.ProbabilityEpsilon {
			continue
		}
		loss := sampleLossWithoutReplacement(available, actual, lossWeight, rng)
		numerics.NormalizeInPlace(gainProbability, state.Rho)
		gain := make([]int, state.Bins)
		numerics.SampleMultinomial(actual, gainProbability, rng, gain)
		for target := 0; target < state.Bins; target++ {
			output[state.matrixIndex(source, target)] += float64(gain[target]-loss[target]) / float64(state.Population)
		}
		totalEvents += actual
	}
	return output, eligibility, totalEvents
}

func edgeChangeByCenter(state *State, before, after []float64) ([]float64, float64) {
	centerChange := make([]float64, state.Bins)
	for center := 0; center < state.Bins; center++ {
		denominator, changed := 0.0, 0.0
		for endpoint := 0; endpoint < state.Bins; endpoint++ {
			beforeMass := before[state.matrixIndex(endpoint, center)] + before[state.matrixIndex(center, endpoint)]
			afterMass := after[state.matrixIndex(endpoint, center)] + after[state.matrixIndex(center, endpoint)]
			denominator += beforeMass
			changed += math.Abs(afterMass - beforeMass)
		}
		if denominator > numerics.ProbabilityEpsilon {
			centerChange[center] = numerics.Clamp(changed/denominator, 0, 1)
		}
	}
	global := 0.0
	for center, change := range centerChange {
		global += state.Rho[center] * change
	}
	return centerChange, numerics.Clamp(global, 0, 1)
}

func adjustPersistence(value, ambiguity float64) float64 {
	value = numerics.Clamp(value, 0, 1)
	if ambiguity >= 0 {
		return value + ambiguity*(1-value)
	}
	return value * (1 + ambiguity)
}

func independentTarget(state *State, request config.RunRequest, profile ClosureProfile, edge []float64) (*State, error) {
	target := newEmptyState(request, state.Layer)
	copy(target.Rho, state.Rho)
	copy(target.Edge, edge)
	target.rebuildCandidate()
	target.rebuildScoreState(profile.ScoreAvailability)
	if state.Layer >= LayerHistogram {
		if err := target.rebuildHistogram(request, profile.EligibilityCorrelation); err != nil {
			return nil, err
		}
	}
	if state.Layer >= LayerTopology {
		target.rebuildTopology(request)
	}
	return target, nil
}

func updateRewiredCoordinates(
	state, target *State,
	request config.RunRequest,
	profile ClosureProfile,
	centerChange []float64,
	globalChange float64,
) {
	globalPersistence := adjustPersistence(math.Pow(1-globalChange, 2), profile.MotifPersistence)
	var previousWedge []float64
	if state.Layer >= LayerWedge {
		previousWedge = state.Wedge
		updated := make([]float64, len(state.Wedge))
		for center, change := range centerChange {
			persistence := adjustPersistence(math.Pow(1-change, 2), profile.MotifPersistence)
			targetWeight := request.Closure.MotifRelaxation * (1 - persistence)
			for i := 0; i < state.Bins; i++ {
				for j := 0; j < state.Bins; j++ {
					index := state.wedgeIndex(i, center, j)
					updated[index] = (1-targetWeight)*state.Wedge[index] + targetWeight*target.Wedge[index]
				}
			}
		}
		state.Wedge = updated
	}
	if state.Layer >= LayerCandidate {
		targetWeight := request.Closure.CandidateRelaxation * (1 - globalPersistence)
		updated := make([]float64, len(state.Xi))
		numerics.MixInto(updated, state.Xi, target.Xi, targetWeight)
		state.Xi = updated
		state.projectXi()
	} else if state.Layer >= LayerWedge {
		// W resolves how rewiring changes the first structural-score moment.
		// For zeta=1 this is an exact projection. For higher powers, preserve
		// the old conditional shape factor S_zeta / S_1^zeta; if a pair gains
		// its first mass, borrow that factor from the independent target.
		for i := 0; i < state.Bins; i++ {
			for j := 0; j < state.Bins; j++ {
				index := state.matrixIndex(i, j)
				oldFirst, newFirst := 0.0, state.wedgeFirstMoment(i, j)
				for center := 0; center < state.Bins; center++ {
					oldFirst += previousWedge[state.wedgeIndex(i, center, j)]
				}
				if math.Abs(state.Steepness-1) <= 1e-12 {
					state.Score[index] = newFirst
				} else if oldFirst > numerics.ProbabilityEpsilon {
					state.Score[index] *= math.Pow(newFirst/oldFirst, state.Steepness)
				} else if newFirst > numerics.ProbabilityEpsilon {
					targetFirst := target.wedgeFirstMoment(i, j)
					if targetFirst > numerics.ProbabilityEpsilon {
						state.Score[index] = target.Score[index] *
							math.Pow(newFirst/targetFirst, state.Steepness)
					} else {
						state.Score[index] = 0
					}
				} else {
					state.Score[index] = 0
				}
			}
		}
	} else {
		targetWeight := request.Closure.MotifRelaxation * (1 - globalPersistence)
		updated := make([]float64, len(state.Score))
		numerics.MixInto(updated, state.Score, target.Score, targetWeight)
		state.Score = updated
	}
	if state.Layer >= LayerHistogram {
		updated := make([]float64, len(state.Histogram))
		numerics.MixInto(updated, state.Histogram, target.Histogram,
			request.Closure.HistogramRelaxation*(1-globalPersistence))
		state.Histogram = updated
	}
	if state.Layer >= LayerTopology {
		updatedComponents := make([]float64, len(state.Components))
		updatedBridges := make([]float64, len(state.Bridges))
		weight := request.Closure.TopologyRelaxation * (1 - globalPersistence)
		numerics.MixInto(updatedComponents, state.Components, target.Components, weight)
		numerics.MixInto(updatedBridges, state.Bridges, target.Bridges, weight)
		state.Components, state.Bridges = updatedComponents, updatedBridges
	}
}

func sampleNodeTransition(state *State, transition []float64, rng *rand.Rand) ([]float64, []float64, []int) {
	counts := make([]int, state.Bins)
	nextCounts := make([]int, state.Bins)
	sampled := make([]float64, state.Bins*state.Bins)
	rowCounts := make([]int, state.Bins)
	for source, mass := range state.Rho {
		counts[source] = int(math.Round(float64(state.Population) * mass))
	}
	difference := state.Population
	for _, count := range counts {
		difference -= count
	}
	if difference != 0 {
		best := 0
		for i := range state.Rho {
			if state.Rho[i] > state.Rho[best] {
				best = i
			}
		}
		counts[best] += difference
	}
	for source, count := range counts {
		if count <= 0 {
			sampled[state.matrixIndex(source, source)] = 1
			continue
		}
		row := transition[source*state.Bins : (source+1)*state.Bins]
		numerics.SampleMultinomial(count, row, rng, rowCounts)
		for target, value := range rowCounts {
			nextCounts[target] += value
			sampled[state.matrixIndex(source, target)] = float64(value) / float64(count)
		}
	}
	rho := make([]float64, state.Bins)
	for i, count := range nextCounts {
		rho[i] = float64(count) / float64(state.Population)
	}
	return rho, sampled, nextCounts
}

func sampleEdgeBlocks(state *State, rho []float64, nodeCounts []int, expected []float64, rng *rand.Rand) []float64 {
	result := make([]float64, len(expected))
	counts := make([]int, state.Bins)
	for source, nodeCount := range nodeCounts {
		total := state.Degree * nodeCount
		if total <= 0 {
			continue
		}
		probabilities := append([]float64(nil), expected[source*state.Bins:(source+1)*state.Bins]...)
		numerics.NormalizeInPlace(probabilities, rho)
		numerics.SampleMultinomial(total, probabilities, rng, counts)
		for target, value := range counts {
			result[state.matrixIndex(source, target)] = float64(value) / float64(state.Population)
		}
	}
	return result
}

func transportHistogram(state *State, transition []float64) []float64 {
	result := make([]float64, len(state.Histogram))
	for source := 0; source < state.Bins; source++ {
		for target := 0; target < state.Bins; target++ {
			weight := transition[state.matrixIndex(source, target)]
			if weight == 0 {
				continue
			}
			for k := 0; k <= state.Degree; k++ {
				for d := 0; d <= state.Degree; d++ {
					for c := 0; c < state.AvailabilityBins; c++ {
						result[state.histogramIndex(target, k, d, c)] +=
							weight * state.Histogram[state.histogramIndex(source, k, d, c)]
					}
				}
			}
		}
	}
	return result
}

func transportXi(state *State, transition []float64) []float64 {
	result := make([]float64, len(state.Xi))
	matrix := make([]float64, state.Bins*state.Bins)
	scratch := make([]float64, len(matrix))
	transported := make([]float64, len(matrix))
	for available := 0; available < 2; available++ {
		for score := 0; score <= state.ScoreMax; score++ {
			for i := 0; i < state.Bins; i++ {
				for j := 0; j < state.Bins; j++ {
					matrix[state.matrixIndex(i, j)] = state.Xi[state.xiIndex(i, j, available, score)]
				}
			}
			numerics.ActiveBackend.Sandwich(transported, scratch, matrix, transition, state.Bins)
			for i := 0; i < state.Bins; i++ {
				for j := 0; j < state.Bins; j++ {
					result[state.xiIndex(i, j, available, score)] = transported[state.matrixIndex(i, j)]
				}
			}
		}
	}
	return result
}

func transportComponents(state *State, transition []float64) []float64 {
	result := make([]float64, len(state.Components))
	for source := 0; source < state.Bins; source++ {
		for target := 0; target < state.Bins; target++ {
			weight := transition[state.matrixIndex(source, target)]
			for sizeBin := 0; sizeBin < state.ComponentSizeBins; sizeBin++ {
				result[state.componentIndex(target, sizeBin)] +=
					weight * state.Components[state.componentIndex(source, sizeBin)]
			}
		}
	}
	return result
}

// Step advances one closed synchronous mesoscopic path.
func Step(state *State, request config.RunRequest, profile ClosureProfile, rng *rand.Rand) (StepDiagnostics, error) {
	recommendations, err := RecommendationKernel(state, request)
	if err != nil {
		return StepDiagnostics{}, err
	}
	applyComponentAmbiguity(state, request, recommendations, profile.ComponentMix)
	neighbors := state.neighborKernel()
	transition, err := TransitionKernel(state, request, neighbors, recommendations)
	if err != nil {
		return StepDiagnostics{}, err
	}
	edgeRewired, _, rewiringEvents := rewire(state, request, profile, neighbors, recommendations, rng)
	centerChange, globalChange := edgeChangeByCenter(state, state.Edge, edgeRewired)
	target, err := independentTarget(state, request, profile, edgeRewired)
	if err != nil {
		return StepDiagnostics{}, err
	}
	updateRewiredCoordinates(state, target, request, profile, centerChange, globalChange)

	rhoNext, sampledTransition, nextCounts := sampleNodeTransition(state, transition, rng)
	matrixScratch := make([]float64, state.Bins*state.Bins)
	edgeExpected := make([]float64, state.Bins*state.Bins)
	numerics.ActiveBackend.Sandwich(edgeExpected, matrixScratch, edgeRewired, sampledTransition, state.Bins)
	edgeNext := sampleEdgeBlocks(state, rhoNext, nextCounts, edgeExpected, rng)

	scoreTransported := make([]float64, len(state.Score))
	numerics.ActiveBackend.Sandwich(scoreTransported, matrixScratch, state.Score, sampledTransition, state.Bins)
	candidateTransported := make([]float64, len(state.Candidate))
	numerics.ActiveBackend.Sandwich(candidateTransported, matrixScratch, state.Candidate, sampledTransition, state.Bins)

	var wedgeTransported []float64
	if state.Layer >= LayerWedge {
		wedgeTransported = make([]float64, len(state.Wedge))
		scratch1 := make([]float64, len(state.Wedge))
		scratch2 := make([]float64, len(state.Wedge))
		numerics.ActiveBackend.TransportTensor3(wedgeTransported, scratch1, scratch2,
			state.Wedge, sampledTransition, state.Bins)
	}
	var histogramTransported, xiTransported, componentTransported, bridgeTransported []float64
	if state.Layer >= LayerHistogram {
		histogramTransported = transportHistogram(state, sampledTransition)
	}
	if state.Layer >= LayerCandidate {
		xiTransported = transportXi(state, sampledTransition)
	}
	if state.Layer >= LayerTopology {
		componentTransported = transportComponents(state, sampledTransition)
		bridgeTransported = make([]float64, len(state.Bridges))
		numerics.ActiveBackend.Sandwich(bridgeTransported, matrixScratch, state.Bridges,
			sampledTransition, state.Bins)
	}

	oldRho := append([]float64(nil), state.Rho...)
	state.Rho, state.Edge = rhoNext, edgeNext
	state.Score, state.Wedge = scoreTransported, wedgeTransported
	state.Histogram, state.Xi = histogramTransported, xiTransported
	state.Components, state.Bridges = componentTransported, bridgeTransported
	if state.Layer >= LayerCandidate {
		state.projectXi()
	} else {
		state.rebuildCandidate()
		for index := range state.Score {
			if candidateTransported[index] > numerics.ProbabilityEpsilon {
				ratio := state.Candidate[index] / candidateTransported[index]
				state.Score[index] *= ratio
				if state.Layer >= LayerWedge {
					i, j := index/state.Bins, index%state.Bins
					for center := 0; center < state.Bins; center++ {
						state.Wedge[state.wedgeIndex(i, center, j)] *= ratio
					}
				}
			} else {
				state.Score[index] = 0
				if state.Layer >= LayerWedge {
					i, j := index/state.Bins, index%state.Bins
					for center := 0; center < state.Bins; center++ {
						state.Wedge[state.wedgeIndex(i, center, j)] = 0
					}
				}
			}
		}
	}

	// Relax transported auxiliary coordinates toward targets consistent with
	// the newly sampled rho/E. This is the only non-projective part of their
	// propagation and is controlled entirely by explicit closure parameters.
	fresh, err := independentTarget(state, request, profile, state.Edge)
	if err != nil {
		return StepDiagnostics{}, err
	}
	if state.Layer >= LayerHistogram {
		updated := make([]float64, len(state.Histogram))
		numerics.MixInto(updated, state.Histogram, fresh.Histogram, request.Closure.HistogramRelaxation)
		state.Histogram = updated
	}
	if state.Layer >= LayerCandidate {
		updated := make([]float64, len(state.Xi))
		numerics.MixInto(updated, state.Xi, fresh.Xi, request.Closure.CandidateRelaxation)
		state.Xi = updated
		state.projectXi()
	}
	if state.Layer >= LayerTopology {
		updatedComponents := make([]float64, len(state.Components))
		updatedBridges := make([]float64, len(state.Bridges))
		numerics.MixInto(updatedComponents, state.Components, fresh.Components, request.Closure.TopologyRelaxation)
		numerics.MixInto(updatedBridges, state.Bridges, fresh.Bridges, request.Closure.TopologyRelaxation)
		state.Components, state.Bridges = updatedComponents, updatedBridges
	}
	if err := state.Validate(); err != nil {
		return StepDiagnostics{}, fmt.Errorf("post-step validation: %w", err)
	}
	maxChange, expectedMoves := 0.0, 0.0
	for i := range state.Rho {
		maxChange = math.Max(maxChange, math.Abs(state.Rho[i]-oldRho[i]))
		expectedMoves += float64(state.Population) * oldRho[i] * (1 - transition[state.matrixIndex(i, i)])
	}
	return StepDiagnostics{RewiringEvents: rewiringEvents, ExpectedMoves: expectedMoves, MaxRhoChange: maxChange}, nil
}
