package meso

import (
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/numerics"
	"strings"
)

func fastSlowRatio(request config.RunRequest) float64 {
	if request.Dynamics.Influence > 0 {
		return request.Dynamics.RewiringRate / request.Dynamics.Influence
	}
	if request.Dynamics.RewiringRate > 0 {
		return math.Inf(1)
	}
	return 0
}

// FastSlowActive reports whether the requested singular-limit projection is
// active at this parameter point. Below the explicit ratio threshold the
// numerical path is the ordinary unsplit generator.
func FastSlowActive(request config.RunRequest) bool {
	return strings.EqualFold(strings.TrimSpace(request.FastSlow.Mode), "conditional_absorption") &&
		fastSlowRatio(request) >= request.FastSlow.RatioThreshold
}

func fastRewiringInputs(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
) (recommendations, neighbors, eligibility []float64, residual float64, err error) {
	recommendations, err = RecommendationKernel(state, request)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	applyComponentAmbiguity(state, request, recommendations, profile.ComponentMix)
	neighbors = state.neighborKernel()
	eligibility = rewiringEligibility(state, request, profile, neighbors, recommendations)
	for source, mass := range state.Rho {
		residual += float64(state.Population) * request.Dynamics.RewiringRate * mass * eligibility[source]
	}
	return recommendations, neighbors, eligibility, residual, nil
}

// FastSlowStep samples the conditional absorbing class of the rewiring
// subchain at fixed rho and then takes one opinion step. The fast subchain is
// generally non-ergodic, so this preserves the sampled absorbing class instead
// of replacing it with a unique stationary average.
func FastSlowStep(
	state *State,
	request config.RunRequest,
	profile ClosureProfile,
	rng *rand.Rand,
) (StepDiagnostics, error) {
	if !FastSlowActive(request) {
		return Step(state, request, profile, rng)
	}

	zeroBatches := 0
	fastEvents := 0
	residual := math.Inf(1)
	substeps := 0
	for ; substeps < request.FastSlow.MaxSubsteps; substeps++ {
		recommendations, neighbors, _, currentResidual, err :=
			fastRewiringInputs(state, request, profile)
		if err != nil {
			return StepDiagnostics{}, err
		}
		residual = currentResidual
		if residual <= request.FastSlow.ResidualTolerance {
			break
		}
		edgeRewired, _, events := rewire(
			state, request, profile, neighbors, recommendations, rng,
		)
		centerChange, globalChange := edgeChangeByCenter(state, state.Edge, edgeRewired)
		target, err := independentTarget(state, request, profile, edgeRewired)
		if err != nil {
			return StepDiagnostics{}, err
		}
		updateRewiredCoordinates(state, target, request, profile, centerChange, globalChange)
		if state.Layer < LayerCandidate {
			copy(state.Candidate, target.Candidate)
		}
		state.Edge = edgeRewired
		if state.Layer == LayerNaive {
			copy(state.Score, target.Score)
		}
		if err := state.Validate(); err != nil {
			return StepDiagnostics{}, err
		}
		fastEvents += events
		if events == 0 {
			zeroBatches++
		} else {
			zeroBatches = 0
		}
		if zeroBatches >= request.FastSlow.ZeroEventBatches &&
			residual <= request.FastSlow.ZeroEventResidual {
			substeps++
			break
		}
	}

	recommendations, neighbors, _, finalResidual, err :=
		fastRewiringInputs(state, request, profile)
	if err != nil {
		return StepDiagnostics{}, err
	}
	residual = finalResidual
	transition, err := TransitionKernel(state, request, neighbors, recommendations)
	if err != nil {
		return StepDiagnostics{}, err
	}
	diagnostics, err := advanceOpinion(state, request, profile, transition, state.Edge, rng)
	if err != nil {
		return StepDiagnostics{}, err
	}
	diagnostics.RewiringEvents = fastEvents
	diagnostics.FastSlowApplied = true
	diagnostics.FastSubsteps = substeps
	diagnostics.FastRewiringEvents = fastEvents
	diagnostics.FastMaxHit = substeps >= request.FastSlow.MaxSubsteps &&
		residual > request.FastSlow.ResidualTolerance
	diagnostics.FastResidualIntensity = numerics.Clamp(residual, 0, math.MaxFloat64)
	return diagnostics, nil
}
