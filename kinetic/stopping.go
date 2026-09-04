package kinetic

import (
	"math"
	"smp-meso/kinetic/statistics"
)

type stoppingStatus struct {
	stop           bool
	reason         string
	stateL1Rate    float64
	nodeEnergyRate float64
	edgeEnergyRate float64
	stableSteps    int
}

type stoppingPlan struct {
	config             StoppingConfig
	outDegree          float64
	interaction        *statistics.InteractionPlan
	previousStep       int
	previousRho        []float64
	previousEdge       []float64
	previousNode       float64
	previousEdgeEnergy float64
	stableSteps        int
	status             stoppingStatus
}

func newStoppingPlan(request RunRequest, axis []float64, current *state) *stoppingPlan {
	if normalize(request.Stopping.Mode) == "fixed_steps" {
		return &stoppingPlan{config: request.Stopping, outDegree: float64(request.OutDegree)}
	}
	interaction := statistics.NewInteractionPlan(axis, request.Dynamics.Tolerance, request.OutDegree)
	node, edgeEnergy := interaction.Energies(current.Rho, current.Edge)
	return &stoppingPlan{
		config: request.Stopping, outDegree: float64(request.OutDegree), interaction: interaction,
		previousRho:  append([]float64(nil), current.Rho...),
		previousEdge: append([]float64(nil), current.Edge...),
		previousNode: node, previousEdgeEnergy: edgeEnergy,
	}
}

func l1Rate(current, previous []float64, elapsed int, scale float64) float64 {
	total := 0.0
	for index, value := range current {
		total += math.Abs(value - previous[index])
	}
	return total / (float64(elapsed) * scale)
}

func energyStable(rate, previous, current, absolute, relative float64) bool {
	scale := math.Max(math.Abs(previous), math.Abs(current))
	return rate <= absolute+relative*scale
}

func (plan *stoppingPlan) check(step int, current *state) stoppingStatus {
	mode := normalize(plan.config.Mode)
	if mode == "fixed_steps" || step%plan.config.CheckEvery != 0 {
		return plan.status
	}
	elapsed := step - plan.previousStep
	stateRate := math.Max(
		l1Rate(current.Rho, plan.previousRho, elapsed, 1),
		l1Rate(current.Edge, plan.previousEdge, elapsed, plan.outDegree),
	)
	node, edgeEnergy := plan.interaction.Energies(current.Rho, current.Edge)
	nodeRate := math.Abs(node-plan.previousNode) / float64(elapsed)
	edgeRate := math.Abs(edgeEnergy-plan.previousEdgeEnergy) / float64(elapsed)
	stateStable := stateRate <= plan.config.StateL1Tolerance
	energiesStable := energyStable(
		nodeRate, plan.previousNode, node,
		plan.config.EnergyAbsoluteTolerance, plan.config.EnergyRelativeTolerance,
	) && energyStable(
		edgeRate, plan.previousEdgeEnergy, edgeEnergy,
		plan.config.EnergyAbsoluteTolerance, plan.config.EnergyRelativeTolerance,
	)
	stable := false
	switch mode {
	case "state":
		stable = stateStable
	case "energy":
		stable = energiesStable
	case "state_and_energy":
		stable = stateStable && energiesStable
	case "state_or_energy":
		stable = stateStable || energiesStable
	}
	if step < plan.config.MinimumSteps {
		stable = false
	}
	if stable {
		plan.stableSteps += elapsed
	} else {
		plan.stableSteps = 0
	}
	copy(plan.previousRho, current.Rho)
	copy(plan.previousEdge, current.Edge)
	plan.previousNode = node
	plan.previousEdgeEnergy = edgeEnergy
	plan.previousStep = step
	plan.status = stoppingStatus{
		stop:   step >= plan.config.MinimumSteps && plan.stableSteps >= plan.config.PatienceSteps,
		reason: "converged_" + mode, stateL1Rate: stateRate, nodeEnergyRate: nodeRate,
		edgeEnergyRate: edgeRate, stableSteps: plan.stableSteps,
	}
	return plan.status
}
