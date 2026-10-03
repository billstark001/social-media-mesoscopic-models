package lifted

import (
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
)

// DeterministicStep is the original D0/D1 plug-in mean update with the zero
// closure profile. Its two random channels are disabled in HybridStep.
func DeterministicStep(state *State, request config.RunRequest) (float64, error) {
	diagnostics, err := HybridStep(state, request, ClosureProfile{}, nil, false, false)
	return diagnostics.MaxRhoChange, err
}

// HybridStep independently selects sampled or plug-in mean rewiring and
// opinion/edge transport. (false,false) is D0/D1; (true,true) is S0/S1.
// The mixed cases retain the same recommendation and HK exposure kernels.
func HybridStep(state *State, request config.RunRequest, profile ClosureProfile,
	rng *rand.Rand, sampleRewiring, sampleTransport bool) (StepDiagnostics, error) {
	strength := 0.0
	if sampleTransport {
		strength = 1
	}
	return HybridStepWithNoise(state, request, profile, rng, sampleRewiring,
		TransportNoise{Nodes: strength, Edges: strength})
}

// TransportNoise scales the node-transition and edge-block sampling residuals
// independently. Zero is the plug-in mean; one recovers the original sample.
// Fractional strengths are used for variance calibration, not as literal
// integer-valued microscopic node or edge counts.
type TransportNoise struct {
	Nodes float64 `json:"nodes"`
	Edges float64 `json:"edges"`
}

// HybridStepWithNoise makes rewiring, node movement and edge-block resampling
// independently controllable. The (1,1) transport endpoint calls the original
// stochastic implementation; (0,0) calls the original deterministic update.
func HybridStepWithNoise(state *State, request config.RunRequest, profile ClosureProfile,
	rng *rand.Rand, sampleRewiring bool, noise TransportNoise) (StepDiagnostics, error) {
	if math.IsNaN(noise.Nodes) || math.IsNaN(noise.Edges) ||
		noise.Nodes < 0 || noise.Nodes > 1 || noise.Edges < 0 || noise.Edges > 1 {
		return StepDiagnostics{}, fmt.Errorf("transport noise strengths must lie in [0,1]")
	}
	if (noise.Nodes != 1 || noise.Edges != 1) && state.Layer != LayerNaive && state.Layer != LayerBase {
		return StepDiagnostics{}, fmt.Errorf("split transport supports naive and base layers only")
	}
	if (sampleRewiring || noise.Nodes > 0 || noise.Edges > 0) && rng == nil {
		return StepDiagnostics{}, fmt.Errorf("sampled channels require an RNG")
	}
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
	var edge []float64
	var events int
	if sampleRewiring {
		edge, _, events = rewire(state, request, profile, neighbors, recommendations, rng)
	} else {
		edge = meanRewire(state, request, profile, neighbors, recommendations)
	}
	centerChange, globalChange := edgeChangeByCenter(state, state.Edge, edge)
	target, err := independentTarget(state, request, profile, edge)
	if err != nil {
		return StepDiagnostics{}, err
	}
	updateRewiredCoordinates(state, target, request, profile, centerChange, globalChange)
	var diagnostics StepDiagnostics
	if noise.Nodes == 1 && noise.Edges == 1 {
		diagnostics, err = advanceOpinion(state, request, profile, transition, edge, rng)
	} else if noise.Nodes == 0 && noise.Edges == 0 {
		diagnostics, err = meanTransport(state, profile, transition, edge)
	} else {
		diagnostics, err = splitTransport(state, request, profile, transition, edge, rng, noise)
	}
	diagnostics.RewiringEvents = events
	return diagnostics, err
}

func meanRewire(state *State, request config.RunRequest, profile ClosureProfile,
	neighbors, recommendations []float64) []float64 {
	edge := append([]float64(nil), state.Edge...)
	eligibility := state.plan.rewiringEligibility(state, request, profile, neighbors, recommendations)
	for i := 0; i < state.Bins; i++ {
		lossMass, gainMass := 0.0, 0.0
		for j := 0; j < state.Bins; j++ {
			idx := i*state.Bins + j
			if math.Abs(state.Axis[i]-state.Axis[j]) > request.Dynamics.Tolerance {
				lossMass += edge[idx]
			} else {
				gainMass += recommendations[idx]
			}
		}
		amount := math.Min(state.Rho[i]*numerics.Clamp(request.Dynamics.RewiringRate*eligibility[i], 0, 1), lossMass)
		if amount <= 0 || gainMass <= numerics.ProbabilityEpsilon {
			continue
		}
		for j := 0; j < state.Bins; j++ {
			idx := i*state.Bins + j
			if math.Abs(state.Axis[i]-state.Axis[j]) > request.Dynamics.Tolerance {
				edge[idx] *= math.Max(1-amount/lossMass, 0)
			} else {
				edge[idx] += amount * recommendations[idx] / gainMass
			}
		}
	}
	return edge
}

func meanTransport(state *State, profile ClosureProfile,
	transition, edge []float64) (StepDiagnostics, error) {
	rho := make([]float64, state.Bins)
	for i, mass := range state.Rho {
		for j := range rho {
			rho[j] += mass * transition[i*state.Bins+j]
		}
	}
	scratch := make([]float64, state.Bins*state.Bins)
	edgeNext := make([]float64, len(edge))
	numerics.ActiveBackend.Sandwich(edgeNext, scratch, edge, transition, state.Bins)
	for i, mass := range rho {
		row := edgeNext[i*state.Bins : (i+1)*state.Bins]
		numerics.NormalizeInPlace(row, rho)
		for j := range row {
			row[j] *= float64(state.Degree) * mass
		}
	}
	score, candidate := state.plan.transportBase(state, transition, scratch)
	change := 0.0
	for i, mass := range rho {
		change = math.Max(change, math.Abs(mass-state.Rho[i]))
	}
	state.Rho, state.Edge = rho, edgeNext
	state.plan.installOpinion(state, profile, score, candidate, nil, nil, nil, nil, nil)
	if err := state.Validate(); err != nil {
		return StepDiagnostics{}, err
	}
	return StepDiagnostics{MaxRhoChange: change}, nil
}

func splitTransport(state *State, request config.RunRequest, profile ClosureProfile,
	transition, edgeBefore []float64, rng *rand.Rand, noise TransportNoise) (StepDiagnostics, error) {
	used := append([]float64(nil), transition...)
	var sampledRho []float64
	if noise.Nodes > 0 {
		var sampled []float64
		sampledRho, sampled, _ = sampleNodeTransition(state, transition, rng)
		for k := range used {
			used[k] += noise.Nodes * (sampled[k] - used[k])
		}
	}
	rho := make([]float64, state.Bins)
	for i, mass := range state.Rho {
		for j := range rho {
			rho[j] += mass * used[i*state.Bins+j]
		}
	}
	if noise.Nodes == 1 {
		copy(rho, sampledRho)
	}
	scratch := make([]float64, state.Bins*state.Bins)
	edgeMean := make([]float64, len(edgeBefore))
	numerics.ActiveBackend.Sandwich(edgeMean, scratch, edgeBefore, used, state.Bins)
	for i, mass := range rho {
		row := edgeMean[i*state.Bins : (i+1)*state.Bins]
		numerics.NormalizeInPlace(row, rho)
		for j := range row {
			row[j] *= float64(state.Degree) * mass
		}
	}
	edge := edgeMean
	if noise.Edges > 0 {
		sampled := sampleFractionalEdgeBlocks(state, rho, edgeMean, rng)
		edge = make([]float64, len(edgeMean))
		for k := range edge {
			edge[k] = edgeMean[k] + noise.Edges*(sampled[k]-edgeMean[k])
		}
	}
	score, candidate := state.plan.transportBase(state, used, scratch)
	maxChange, expectedMoves := 0.0, 0.0
	for i, old := range state.Rho {
		maxChange = math.Max(maxChange, math.Abs(rho[i]-old))
		expectedMoves += float64(state.Population) * old * (1 - transition[state.matrixIndex(i, i)])
	}
	state.Rho, state.Edge = rho, edge
	state.plan.installOpinion(state, profile, score, candidate, nil, nil, nil, nil, nil)
	if err := state.Validate(); err != nil {
		return StepDiagnostics{}, fmt.Errorf("post-split-step validation: %w", err)
	}
	return StepDiagnostics{ExpectedMoves: expectedMoves, MaxRhoChange: maxChange}, nil
}

// A deterministic node update can have fractional mass. Draw an integer
// multinomial only for the destination proportions, then restore each exact
// source-row mass. This is the edge-only counterfactual, not S1's integer law.
func sampleFractionalEdgeBlocks(state *State, rho, expected []float64, rng *rand.Rand) []float64 {
	result := make([]float64, len(expected))
	counts := make([]int, state.Bins)
	for i, mass := range rho {
		if mass <= 0 {
			continue
		}
		totalMass := float64(state.Degree) * mass
		count := int(math.Round(totalMass * float64(state.Population)))
		if count < 1 {
			count = 1
		}
		probabilities := append([]float64(nil), expected[i*state.Bins:(i+1)*state.Bins]...)
		numerics.NormalizeInPlace(probabilities, rho)
		numerics.SampleMultinomial(count, probabilities, rng, counts)
		for j, value := range counts {
			result[state.matrixIndex(i, j)] = totalMass * float64(value) / float64(count)
		}
	}
	return result
}
