package solver

import (
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"time"
)

type PointEstimate struct {
	Paths                  int       `json:"paths"`
	Counts                 []int     `json:"counts"`
	Probabilities          []float64 `json:"probabilities"`
	MeanSteps              float64   `json:"mean_steps"`
	MeanRewiringEvents     float64   `json:"mean_rewiring_events"`
	ConvergedPaths         int       `json:"converged_paths"`
	FastSlowAppliedPaths   int       `json:"fast_slow_applied_paths"`
	MeanFastSubsteps       float64   `json:"mean_fast_substeps"`
	MeanFastRewiringEvents float64   `json:"mean_fast_rewiring_events"`
	FastMaxHits            int       `json:"fast_max_hits"`
	MeanFinalFastResidual  float64   `json:"mean_final_fast_residual"`
}

type Diagnostics struct {
	Backend        string  `json:"backend"`
	StateDimension int     `json:"state_dimension"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
	IntervalKind   string  `json:"interval_kind"`
	Decomposition  string  `json:"decomposition"`
}

type Result struct {
	RequestID   string           `json:"request_id"`
	Layer       string           `json:"layer"`
	Categories  []string         `json:"categories"`
	Point       PointEstimate    `json:"point"`
	Interval    IntervalEstimate `json:"interval"`
	Diagnostics Diagnostics      `json:"diagnostics"`
}

func Run(request config.RunRequest) (Result, error) {
	return RunWithProgress(request, 0, nil)
}

// RunWithProgress preserves the numerical path of Run while exposing optional
// telemetry. A zero progressStepInterval disables within-path heartbeats but
// still emits request, scenario, and path-completion events.
func RunWithProgress(
	request config.RunRequest,
	progressStepInterval int,
	progress protocol.ProgressFunc,
) (Result, error) {
	started := time.Now()
	if err := request.Validate(); err != nil {
		return Result{}, err
	}
	layer, err := config.ParseLayer(request.Layer)
	if err != nil {
		return Result{}, err
	}
	emit := func(event protocol.ProgressEvent) {
		if progress == nil {
			return
		}
		event.RequestID = request.RequestID
		event.Solver = "lifted"
		event.Layer = layer.String()
		event.ElapsedSeconds = time.Since(started).Seconds()
		progress(event)
	}
	emit(protocol.ProgressEvent{Event: "request_started"})
	pointProgress := func(event protocol.ProgressEvent) {
		event.Stage = "point"
		emit(event)
	}
	if progress == nil {
		pointProgress = nil
	}
	point, err := runEnsembleWithProgress(
		request, layer, lifted.ClosureProfile{}, request.Paths,
		0xbb67ae8584caa73b, progressStepInterval, pointProgress,
	)
	if err != nil {
		return Result{}, err
	}
	interval, err := runInterval(request, layer, point, progressStepInterval, emit)
	if err != nil {
		return Result{}, err
	}
	result := Result{
		RequestID: request.RequestID, Layer: layer.String(),
		Categories: append([]string(nil), Categories...),
		Point: PointEstimate{
			Paths: point.Paths, Counts: point.Counts, Probabilities: point.Probabilities,
			MeanSteps: point.MeanSteps, MeanRewiringEvents: point.MeanRewiringEvents,
			ConvergedPaths:         point.ConvergedPaths,
			FastSlowAppliedPaths:   point.FastSlowAppliedPaths,
			MeanFastSubsteps:       point.MeanFastSubsteps,
			MeanFastRewiringEvents: point.MeanFastRewiringEvents,
			FastMaxHits:            point.FastMaxHits,
			MeanFinalFastResidual:  point.MeanFinalFastResidual,
		},
		Interval: interval,
		Diagnostics: Diagnostics{
			Backend: numerics.ActiveBackend.Name(), StateDimension: point.StateDimension,
			ElapsedSeconds: time.Since(started).Seconds(),
			IntervalKind:   "closure-envelope; not a certified continuum robust optimum",
			Decomposition:  request.FastSlow.Mode,
		},
	}
	emit(protocol.ProgressEvent{
		Event: "request_completed", StateDimension: point.StateDimension,
		CompletedPaths: request.Paths + interval.ScenarioCount*request.IntervalPaths,
		TotalPaths:     request.Paths + interval.ScenarioCount*request.IntervalPaths,
	})
	return result, nil
}
