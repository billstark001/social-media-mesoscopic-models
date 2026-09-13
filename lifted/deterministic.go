package lifted

import (
	"fmt"
	"math"
	"smp-meso/config"
	"smp-meso/numerics"
)

// DeterministicStep advances a naive or base state using continuous masses.
// It retains the finite-exposure transition law, but replaces node, edge and
// rewiring draws with plug-in means. It is not the full conditional mean of
// Step: caps, shared reclassification and nonlinear score updates do not
// commute with expectations. The closure profile is the zero/base profile.
func DeterministicStep(state *State, request config.RunRequest) (float64, error) {
	if state.Layer != LayerNaive && state.Layer != LayerBase {
		return 0, fmt.Errorf("deterministic evolution supports naive and base layers only")
	}
	profile := ClosureProfile{}
	recommendations, err := RecommendationKernel(state, request)
	if err != nil {
		return 0, err
	}
	neighbors := state.neighborKernel()
	transition, err := TransitionKernel(state, request, neighbors, recommendations)
	if err != nil {
		return 0, err
	}
	eligibility := state.plan.rewiringEligibility(state, request, profile, neighbors, recommendations)
	edge := append([]float64(nil), state.Edge...)
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
	centerChange, globalChange := edgeChangeByCenter(state, state.Edge, edge)
	target, err := independentTarget(state, request, profile, edge)
	if err != nil {
		return 0, err
	}
	updateRewiredCoordinates(state, target, request, profile, centerChange, globalChange)
	rho := make([]float64, state.Bins)
	for i, m := range state.Rho {
		for j := range rho {
			rho[j] += m * transition[i*state.Bins+j]
		}
	}
	scratch := make([]float64, state.Bins*state.Bins)
	edgeNext := make([]float64, len(edge))
	numerics.ActiveBackend.Sandwich(edgeNext, scratch, edge, transition, state.Bins)
	// Same row-normalization as expected edge-block resampling, without integers.
	for i, m := range rho {
		row := edgeNext[i*state.Bins : (i+1)*state.Bins]
		numerics.NormalizeInPlace(row, rho)
		for j := range row {
			row[j] *= float64(state.Degree) * m
		}
	}
	score, candidate := state.plan.transportBase(state, transition, scratch)
	change := 0.0
	for i, m := range rho {
		change = math.Max(change, math.Abs(m-state.Rho[i]))
	}
	state.Rho, state.Edge = rho, edgeNext
	state.plan.installOpinion(state, profile, score, candidate, nil, nil, nil, nil, nil)
	if err := state.Validate(); err != nil {
		return 0, err
	}
	return change, nil
}
