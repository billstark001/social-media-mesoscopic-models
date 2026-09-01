package meso

import (
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
)

type StepDiagnostics struct {
	RewiringEvents        int
	ExpectedMoves         float64
	MaxRhoChange          float64
	FastSlowApplied       bool
	FastSubsteps          int
	FastRewiringEvents    int
	FastMaxHit            bool
	FastResidualIntensity float64
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

func inferredBridgeFractions(state *State) []float64 {
	adjacency := state.undirectedAdjacencyProbabilities()
	means := state.commonNeighborMeans(adjacency)
	result := make([]float64, state.Bins*state.Bins)
	for source := 0; source < state.Bins; source++ {
		for target := 0; target < state.Bins; target++ {
			index := state.matrixIndex(source, target)
			if state.Edge[index] > numerics.ProbabilityEpsilon {
				result[index] = 1 / (1 + means[index])
			}
		}
	}
	return result
}

func topologyBridgeFractions(state *State) []float64 {
	result := make([]float64, state.Bins*state.Bins)
	for index, edge := range state.Edge {
		if edge > numerics.ProbabilityEpsilon {
			result[index] = numerics.Clamp(state.Bridges[index]/edge, 0, 1)
		}
	}
	return result
}

func inferredRewiringEligibility(
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
		result[source] = numerics.Clamp(result[source], 0, 1)
	}
	return result
}

func histogramRewiringEligibility(
	state *State,
	_ config.RunRequest,
	_ ClosureProfile,
	_, _ []float64,
) []float64 {
	result := make([]float64, state.Bins)
	for source := range result {
		result[source] = state.histogramEligibility(source)
	}
	return result
}

func topologyRewiringEligibility(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
	neighbors, recommendations []float64,
) []float64 {
	result := histogramRewiringEligibility(state, request, profile, neighbors, recommendations)
	for source := range result {
		if state.Rho[source] <= numerics.ProbabilityEpsilon {
			continue
		}
		smallMass := 0.0
		for sizeBin := 0; sizeBin < state.ComponentSizeBins-1; sizeBin++ {
			smallMass += state.Components[state.componentIndex(source, sizeBin)]
		}
		isolatedFraction := numerics.Clamp(smallMass/state.Rho[source], 0, 1)
		result[source] *= 1 - isolatedFraction*(1-request.Recommender.RandomRatio)
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
	eligibility := state.plan.rewiringEligibility(state, request, profile, neighbors, recommendations)
	bridgeFractions := state.plan.bridgeFractions(state)
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
				bridgeFraction := bridgeFractions[index]
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
	if err := target.plan.initializeAuxiliary(target, request, profile); err != nil {
		return nil, err
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
	globalPersistence := adjustPersistence(
		math.Pow(1-globalChange, 2), profile.MotifPersistence,
	)
	previousWedge := state.plan.rewireWedge(
		state, target, request.Closure, profile, centerChange, globalPersistence,
	)
	state.plan.rewireScore(state, target, request.Closure, profile, previousWedge, globalPersistence)
	state.plan.rewireHistogram(state, target, request.Closure, profile, previousWedge, globalPersistence)
	state.plan.rewireTopology(state, target, request.Closure, profile, previousWedge, globalPersistence)
}

func rewireNaiveScore(
	state, target *State,
	_ config.ClosureConfig,
	_ ClosureProfile,
	_ []float64,
	_ float64,
) {
	copy(state.Candidate, target.Candidate)
	copy(state.Score, target.Score)
}

func rewireBaseScore(
	state, target *State,
	closure config.ClosureConfig,
	_ ClosureProfile,
	_ []float64,
	globalPersistence float64,
) {
	targetWeight := closure.MotifRelaxation * (1 - globalPersistence)
	updated := make([]float64, len(state.Score))
	numerics.MixInto(updated, state.Score, target.Score, targetWeight)
	state.Score = updated
}

func rewireWedgeCoordinate(
	state, target *State,
	closure config.ClosureConfig,
	profile ClosureProfile,
	centerChange []float64,
	_ float64,
) []float64 {
	previousWedge := state.Wedge
	updated := make([]float64, len(state.Wedge))
	for center, change := range centerChange {
		persistence := adjustPersistence(math.Pow(1-change, 2), profile.MotifPersistence)
		targetWeight := closure.MotifRelaxation * (1 - persistence)
		for i := 0; i < state.Bins; i++ {
			for j := 0; j < state.Bins; j++ {
				index := state.wedgeIndex(i, center, j)
				updated[index] = (1-targetWeight)*state.Wedge[index] + targetWeight*target.Wedge[index]
			}
		}
	}
	state.Wedge = updated
	return previousWedge
}

func rewireWedgeScore(
	state, target *State,
	_ config.ClosureConfig,
	_ ClosureProfile,
	previousWedge []float64,
	_ float64,
) {
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
}

func rewireCandidateScore(
	state, target *State,
	closure config.ClosureConfig,
	_ ClosureProfile,
	_ []float64,
	globalPersistence float64,
) {
	targetWeight := closure.CandidateRelaxation * (1 - globalPersistence)
	updated := make([]float64, len(state.Xi))
	numerics.MixInto(updated, state.Xi, target.Xi, targetWeight)
	state.Xi = updated
	state.projectXi()
}

func rewireHistogramCoordinate(
	state, target *State,
	closure config.ClosureConfig,
	_ ClosureProfile,
	_ []float64,
	globalPersistence float64,
) {
	updated := make([]float64, len(state.Histogram))
	numerics.MixInto(
		updated,
		state.Histogram,
		target.Histogram,
		closure.HistogramRelaxation*(1-globalPersistence),
	)
	state.Histogram = updated
}

func rewireTopologyCoordinate(
	state, target *State,
	closure config.ClosureConfig,
	_ ClosureProfile,
	_ []float64,
	globalPersistence float64,
) {
	updatedComponents := make([]float64, len(state.Components))
	updatedBridges := make([]float64, len(state.Bridges))
	weight := closure.TopologyRelaxation * (1 - globalPersistence)
	numerics.MixInto(updatedComponents, state.Components, target.Components, weight)
	numerics.MixInto(updatedBridges, state.Bridges, target.Bridges, weight)
	state.Components, state.Bridges = updatedComponents, updatedBridges
}

func copyFastCandidate(state, target *State) {
	copy(state.Candidate, target.Candidate)
}

func advanceOpinion(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
	transition, edgeBefore []float64,
	rng *rand.Rand,
) (StepDiagnostics, error) {
	rhoNext, sampledTransition, nextCounts := sampleNodeTransition(state, transition, rng)
	matrixScratch := make([]float64, state.Bins*state.Bins)
	edgeExpected := make([]float64, state.Bins*state.Bins)
	numerics.ActiveBackend.Sandwich(edgeExpected, matrixScratch, edgeBefore, sampledTransition, state.Bins)
	edgeNext := sampleEdgeBlocks(state, rhoNext, nextCounts, edgeExpected, rng)

	score, candidate := state.plan.transportBase(state, sampledTransition, matrixScratch)
	wedge := state.plan.transportWedge(state, sampledTransition)
	histogram := state.plan.transportHistogram(state, sampledTransition)
	xi := state.plan.transportCandidate(state, sampledTransition)
	components, bridges := state.plan.transportTopology(state, sampledTransition, matrixScratch)

	oldRho := append([]float64(nil), state.Rho...)
	state.Rho, state.Edge = rhoNext, edgeNext
	state.plan.installOpinion(
		state, profile, score, candidate, wedge, histogram, xi, components, bridges,
	)
	if err := state.plan.relaxOpinion(state, request, profile); err != nil {
		return StepDiagnostics{}, err
	}
	if err := state.Validate(); err != nil {
		return StepDiagnostics{}, fmt.Errorf("post-step validation: %w", err)
	}
	maxChange, expectedMoves := 0.0, 0.0
	for i := range state.Rho {
		maxChange = math.Max(maxChange, math.Abs(state.Rho[i]-oldRho[i]))
		expectedMoves += float64(state.Population) * oldRho[i] * (1 - transition[state.matrixIndex(i, i)])
	}
	return StepDiagnostics{ExpectedMoves: expectedMoves, MaxRhoChange: maxChange}, nil
}

func transportBaseOpinion(
	state *State,
	transition, matrixScratch []float64,
) ([]float64, []float64) {
	score := make([]float64, len(state.Score))
	numerics.ActiveBackend.Sandwich(
		score, matrixScratch, state.Score, transition, state.Bins,
	)
	candidate := make([]float64, len(state.Candidate))
	numerics.ActiveBackend.Sandwich(
		candidate, matrixScratch, state.Candidate, transition, state.Bins,
	)
	return score, candidate
}

func transportWedgeOpinion(state *State, transition []float64) []float64 {
	wedge := make([]float64, len(state.Wedge))
	scratch1 := make([]float64, len(state.Wedge))
	scratch2 := make([]float64, len(state.Wedge))
	numerics.ActiveBackend.TransportTensor3(
		wedge, scratch1, scratch2, state.Wedge, transition, state.Bins,
	)
	return wedge
}

func transportHistogramOpinion(state *State, transition []float64) []float64 {
	return transportHistogram(state, transition)
}

func transportCandidateOpinion(state *State, transition []float64) []float64 {
	return transportXi(state, transition)
}

func transportTopologyOpinion(
	state *State,
	transition, matrixScratch []float64,
) ([]float64, []float64) {
	components := transportComponents(state, transition)
	bridges := make([]float64, len(state.Bridges))
	numerics.ActiveBackend.Sandwich(
		bridges,
		matrixScratch,
		state.Bridges,
		transition,
		state.Bins,
	)
	return components, bridges
}

func installNaiveOpinion(
	state *State,
	profile ClosureProfile,
	_, _, _, _, _, _, _ []float64,
) {
	state.rebuildCandidate()
	state.rebuildScoreState(profile.ScoreAvailability)
}

func installBaseOpinion(
	state *State,
	_ ClosureProfile,
	score, candidate, wedge, histogram, _, _, _ []float64,
) {
	state.Score, state.Wedge = score, wedge
	state.Histogram = histogram
	state.rebuildCandidate()
	state.plan.rescaleCandidate(state, candidate)
}

func installCandidateOpinion(
	state *State,
	_ ClosureProfile,
	_, _, wedge, histogram, xi, components, bridges []float64,
) {
	state.Wedge, state.Histogram = wedge, histogram
	state.Xi = xi
	state.Components, state.Bridges = components, bridges
	state.projectXi()
}

func rescaleBaseCandidate(state *State, transported []float64) {
	rescaleCandidate(state, transported, false)
}

func rescaleWedgeCandidate(state *State, transported []float64) {
	rescaleCandidate(state, transported, true)
}

func rescaleCandidate(state *State, transported []float64, rescaleWedge bool) {
	for index := range state.Score {
		if transported[index] > numerics.ProbabilityEpsilon {
			ratio := state.Candidate[index] / transported[index]
			state.Score[index] *= ratio
			if rescaleWedge {
				i, j := index/state.Bins, index%state.Bins
				for center := 0; center < state.Bins; center++ {
					state.Wedge[state.wedgeIndex(i, center, j)] *= ratio
				}
			}
			continue
		}
		state.Score[index] = 0
		if rescaleWedge {
			i, j := index/state.Bins, index%state.Bins
			for center := 0; center < state.Bins; center++ {
				state.Wedge[state.wedgeIndex(i, center, j)] = 0
			}
		}
	}
}

func relaxRetainedOpinion(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
) error {
	// Relax transported auxiliary coordinates toward targets consistent with
	// the newly sampled rho/E. This is the only non-projective part of their
	// propagation and is controlled entirely by explicit closure parameters.
	fresh, err := independentTarget(state, request, profile, state.Edge)
	if err != nil {
		return err
	}
	state.plan.relaxHistogram(state, fresh, request)
	state.plan.relaxCandidate(state, fresh, request)
	state.plan.relaxTopology(state, fresh, request)
	return nil
}

func relaxHistogramOpinion(state, fresh *State, request config.RunRequest) {
	updated := make([]float64, len(state.Histogram))
	numerics.MixInto(
		updated, state.Histogram, fresh.Histogram, request.Closure.HistogramRelaxation,
	)
	state.Histogram = updated
}

func relaxCandidateOpinion(state, fresh *State, request config.RunRequest) {
	updated := make([]float64, len(state.Xi))
	numerics.MixInto(updated, state.Xi, fresh.Xi, request.Closure.CandidateRelaxation)
	state.Xi = updated
	state.projectXi()
}

func relaxTopologyOpinion(state, fresh *State, request config.RunRequest) {
	updatedComponents := make([]float64, len(state.Components))
	updatedBridges := make([]float64, len(state.Bridges))
	numerics.MixInto(
		updatedComponents, state.Components, fresh.Components, request.Closure.TopologyRelaxation,
	)
	numerics.MixInto(
		updatedBridges, state.Bridges, fresh.Bridges, request.Closure.TopologyRelaxation,
	)
	state.Components, state.Bridges = updatedComponents, updatedBridges
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
	columns := (state.Degree + 1) * (state.Degree + 1) * state.AvailabilityBins
	numerics.ActiveBackend.ApplyDenseTransitionBatch(result, state.Histogram, transition, state.Bins, columns)
	return result
}

func transportXi(state *State, transition []float64) []float64 {
	result := make([]float64, len(state.Xi))
	numerics.ActiveBackend.TransportMatrixChannels(
		result, make([]float64, len(result)), state.Xi, transition,
		state.Bins, 2*(state.ScoreMax+1),
	)
	return result
}

func transportComponents(state *State, transition []float64) []float64 {
	result := make([]float64, len(state.Components))
	numerics.ActiveBackend.ApplyDenseTransitionBatch(
		result, state.Components, transition, state.Bins, state.ComponentSizeBins,
	)
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

	diagnostics, err := advanceOpinion(state, request, profile, transition, edgeRewired, rng)
	diagnostics.RewiringEvents = rewiringEvents
	return diagnostics, err
}
