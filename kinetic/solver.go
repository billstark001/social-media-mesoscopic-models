package kinetic

import (
	"fmt"
	"math"
	"smp-meso/kinetic/statistics"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"time"
)

type Series struct {
	Time         *protocol.EncodedArray `json:"time,omitempty"`
	Polarization *protocol.EncodedArray `json:"polarization,omitempty"`
	Subjective   *protocol.EncodedArray `json:"subjective,omitempty"`
	Homophily    *protocol.EncodedArray `json:"homophily,omitempty"`
	HomophilyRaw *protocol.EncodedArray `json:"homophily_raw,omitempty"`
	NodeEnergy   *protocol.EncodedArray `json:"node_energy,omitempty"`
	EdgeEnergy   *protocol.EncodedArray `json:"edge_energy,omitempty"`
}

type PassageResult struct {
	Reached bool     `json:"reached"`
	Time    *float64 `json:"time"`
}

type Summary struct {
	Pathway                  *float64       `json:"pathway,omitempty"`
	PolarizationFirstPassage *PassageResult `json:"polarization_first_passage,omitempty"`
	HomophilyFirstPassage    *PassageResult `json:"homophily_first_passage,omitempty"`
}

type Diagnostics struct {
	Backend                string  `json:"backend"`
	StateDimension         int     `json:"state_dimension"`
	RecordedPoints         int     `json:"recorded_points"`
	ElapsedSeconds         float64 `json:"elapsed_seconds"`
	OpinionMethod          string  `json:"opinion_method"`
	Recommender            string  `json:"recommender"`
	MaxNodeMassResidual    float64 `json:"max_node_mass_residual"`
	MaxFixedDegreeResidual float64 `json:"max_fixed_degree_residual"`
	ExecutedSteps          int     `json:"executed_steps"`
	StopReason             string  `json:"stop_reason"`
	Converged              bool    `json:"converged"`
	FinalStateL1Rate       float64 `json:"final_state_l1_rate"`
	FinalNodeEnergyRate    float64 `json:"final_node_energy_rate"`
	FinalEdgeEnergyRate    float64 `json:"final_edge_energy_rate"`
	StableSteps            int     `json:"stable_steps"`
}

type Result struct {
	RequestID   string          `json:"request_id"`
	Series      Series          `json:"series"`
	Summary     Summary         `json:"summary"`
	Snapshots   *SnapshotSeries `json:"snapshots,omitempty"`
	Diagnostics Diagnostics     `json:"diagnostics"`
}

func encoded(values []float64) (*protocol.EncodedArray, error) {
	if values == nil {
		return nil, nil
	}
	value, err := protocol.EncodeFloat64(values, len(values))
	if err != nil {
		return nil, err
	}
	return &value, nil
}

func passage(value statistics.FirstPassage) *PassageResult {
	result := &PassageResult{Reached: value.Reached}
	if value.Reached {
		timestamp := value.Time
		result.Time = &timestamp
	}
	return result
}

func statisticsPlan(request RunRequest, grid *gridGeometry) *statistics.Plan {
	observables := request.Observables
	return statistics.NewPlan(grid.Axis, grid.Concordance, statistics.Config{
		Polarization: observables.Polarization, Subjective: observables.Subjective,
		Homophily: observables.Homophily, HomophilyRaw: observables.HomophilyRaw,
		NodeEnergy: observables.NodeEnergy, EdgeEnergy: observables.EdgeEnergy,
		Pathway:                   observables.Pathway,
		PolarizationFirstPassage:  observables.PolarizationFirstPassage,
		HomophilyFirstPassage:     observables.HomophilyFirstPassage,
		PolarizationThreshold:     observables.PolarizationThreshold,
		HomophilyThreshold:        observables.HomophilyThreshold,
		MinimumBandwidth:          observables.MinimumBandwidth,
		ObjectiveEffectiveSamples: observables.ObjectiveEffectiveSamples,
		DistanceGridSize:          request.Resolution.DistanceGridSize,
		Population:                request.Population, OutDegree: request.OutDegree,
		Tolerance:   request.Dynamics.Tolerance,
		DomainWidth: request.Initial.OpinionMax - request.Initial.OpinionMin,
	})
}

func resultFromOutcome(
	request RunRequest,
	outcome statistics.Outcome,
	dimension, recorded int,
	elapsed time.Duration,
	maxNodeMassResidual, maxFixedDegreeResidual float64,
	executedSteps int, stopping stoppingStatus,
) (Result, error) {
	result := Result{RequestID: request.RequestID, Diagnostics: Diagnostics{
		Backend: numerics.ActiveBackend.Name(), StateDimension: dimension, RecordedPoints: recorded,
		ElapsedSeconds: elapsed.Seconds(), OpinionMethod: normalize(request.Dynamics.OpinionMethod),
		Recommender: normalize(request.Recommender.Type), MaxNodeMassResidual: maxNodeMassResidual,
		MaxFixedDegreeResidual: maxFixedDegreeResidual, ExecutedSteps: executedSteps,
		StopReason: stopping.reason, Converged: stopping.stop,
		FinalStateL1Rate: stopping.stateL1Rate, FinalNodeEnergyRate: stopping.nodeEnergyRate,
		FinalEdgeEnergyRate: stopping.edgeEnergyRate, StableSteps: stopping.stableSteps,
	}}
	var err error
	if result.Series.Time, err = encoded(outcome.Time); err != nil {
		return Result{}, err
	}
	if result.Series.Polarization, err = encoded(outcome.Polarization); err != nil {
		return Result{}, err
	}
	if result.Series.Subjective, err = encoded(outcome.Subjective); err != nil {
		return Result{}, err
	}
	if result.Series.Homophily, err = encoded(outcome.Homophily); err != nil {
		return Result{}, err
	}
	if result.Series.HomophilyRaw, err = encoded(outcome.HomophilyRaw); err != nil {
		return Result{}, err
	}
	if result.Series.NodeEnergy, err = encoded(outcome.NodeEnergy); err != nil {
		return Result{}, err
	}
	if result.Series.EdgeEnergy, err = encoded(outcome.EdgeEnergy); err != nil {
		return Result{}, err
	}
	if outcome.HasPathway {
		value := outcome.Pathway
		result.Summary.Pathway = &value
	}
	if request.Observables.PolarizationFirstPassage {
		result.Summary.PolarizationFirstPassage = passage(outcome.PolarizationPass)
	}
	if request.Observables.HomophilyFirstPassage {
		result.Summary.HomophilyFirstPassage = passage(outcome.HomophilyPass)
	}
	return result, nil
}

func conservationResiduals(current *state) (float64, float64) {
	total := 0.0
	for _, value := range current.Rho {
		total += value
	}
	nodeResidual := math.Abs(total - 1)
	degreeResidual := 0.0
	size := current.request.OpinionBins
	degree := float64(current.request.OutDegree)
	for source := 0; source < size; source++ {
		row := 0.0
		for target := 0; target < size; target++ {
			row += current.Edge[source*size+target]
		}
		degreeResidual = math.Max(degreeResidual, math.Abs(row-degree*current.Rho[source]))
	}
	return nodeResidual, degreeResidual
}

func Run(request RunRequest) (Result, error) {
	return RunWithProgress(request, 0, nil)
}

func RunWithProgress(request RunRequest, progressStepInterval int, progress protocol.ProgressFunc) (Result, error) {
	started := time.Now()
	if err := request.Validate(); err != nil {
		return Result{}, err
	}
	grid, err := newGrid(request)
	if err != nil {
		return Result{}, err
	}
	current, err := newState(request, grid)
	if err != nil {
		return Result{}, err
	}
	recommend := planRecommender(request)
	advanceOpinion := planOpinionEvolution(request, grid)
	measure := statisticsPlan(request, grid)
	snapshots, err := newSnapshotCollector(request, grid.Axis)
	if err != nil {
		return Result{}, err
	}
	emit := func(event protocol.ProgressEvent) {
		if progress == nil {
			return
		}
		event.RequestID = request.RequestID
		event.Solver = "kinetic"
		event.TotalSteps = request.Steps
		event.ElapsedSeconds = time.Since(started).Seconds()
		progress(event)
	}
	emit(protocol.ProgressEvent{Event: "request_started"})
	measure.ObservePathway(current.Rho, current.Edge)
	measure.Record(0, current.Rho, current.Edge)
	maxNodeMassResidual, maxFixedDegreeResidual := conservationResiduals(current)
	recorded := 1
	executedSteps := 0
	stopStatus := stoppingStatus{reason: "max_steps"}
	stopper := newStoppingPlan(request, grid.Axis, current)
	workspace := newStepWorkspace(request.OpinionBins)
	for step := 1; step <= request.Steps; step++ {
		structuralScore := current.plan.structuralScore(current)
		values := computeFields(current, recommend, structuralScore)
		if snapshots != nil {
			if err := snapshots.record(step-1, current, &values); err != nil {
				return Result{}, fmt.Errorf("snapshot at step %d: %w", step-1, err)
			}
		}
		for index := range workspace.edgeRewired {
			workspace.edgeRewired[index] = current.Edge[index] + request.Dt*values.RewiringFlux[index]
			if workspace.edgeRewired[index] < 0 && workspace.edgeRewired[index] > -1e-12 {
				workspace.edgeRewired[index] = 0
			}
		}
		current.plan.rewire(current, current.Edge, workspace.edgeRewired)
		if err := advanceOpinion(current, values, workspace.edgeRewired, workspace); err != nil {
			return Result{}, fmt.Errorf("step %d opinion evolution: %w", step, err)
		}
		if err := current.validate(); err != nil {
			return Result{}, fmt.Errorf("step %d: %w", step, err)
		}
		measure.ObservePathway(current.Rho, current.Edge)
		nodeResidual, degreeResidual := conservationResiduals(current)
		maxNodeMassResidual = math.Max(maxNodeMassResidual, nodeResidual)
		maxFixedDegreeResidual = math.Max(maxFixedDegreeResidual, degreeResidual)
		if step%request.RecordEvery == 0 || step == request.Steps {
			measure.Record(float64(step)*request.Dt, current.Rho, current.Edge)
			recorded++
		}
		if progressStepInterval > 0 && step%progressStepInterval == 0 {
			emit(protocol.ProgressEvent{Event: "step_heartbeat", Step: step})
		}
		executedSteps = step
		status := stopper.check(step, current)
		if status.stop {
			stopStatus = status
			if step%request.RecordEvery != 0 && step != request.Steps {
				measure.Record(float64(step)*request.Dt, current.Rho, current.Edge)
				recorded++
			}
			break
		}
	}
	if !stopStatus.stop && normalize(request.Stopping.Mode) != "fixed_steps" {
		stopStatus = stopper.status
		stopStatus.reason = "max_steps"
		stopStatus.stop = false
	}
	if snapshots != nil && snapshots.wants(executedSteps) {
		var finalFields *fields
		if snapshots.requiresField {
			structuralScore := current.plan.structuralScore(current)
			values := computeFields(current, recommend, structuralScore)
			finalFields = &values
		}
		if err := snapshots.record(executedSteps, current, finalFields); err != nil {
			return Result{}, fmt.Errorf("snapshot at final step: %w", err)
		}
	}
	outcome := measure.Outcome()
	dimension := len(current.Rho) + len(current.Edge) + len(current.Wedge)
	result, err := resultFromOutcome(
		request,
		outcome,
		dimension,
		recorded,
		time.Since(started),
		maxNodeMassResidual,
		maxFixedDegreeResidual,
		executedSteps,
		stopStatus,
	)
	if err != nil {
		return Result{}, err
	}
	result.Snapshots, err = snapshots.outcome(current)
	if err != nil {
		return Result{}, err
	}
	emit(protocol.ProgressEvent{Event: "request_completed", Step: executedSteps, StateDimension: dimension})
	return result, nil
}
