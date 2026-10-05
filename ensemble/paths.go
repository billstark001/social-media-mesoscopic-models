package ensemble

import (
	"fmt"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/protocol"
	"smp-meso/trajectory"
	"sync"
	"sync/atomic"
)

type PathOutcome struct {
	Category           int
	Steps              int
	RewiringEvents     int
	StateDimension     int
	FastSlowApplied    bool
	FastSubsteps       int
	FastRewiringEvents int
	FastMaxHits        int
	FinalFastResidual  float64
}

type Ensemble struct {
	Counts                 []int
	Probabilities          []float64
	Paths                  int
	MeanSteps              float64
	MeanRewiringEvents     float64
	ConvergedPaths         int
	StateDimension         int
	FastSlowAppliedPaths   int
	MeanFastSubsteps       float64
	MeanFastRewiringEvents float64
	FastMaxHits            int
	MeanFinalFastResidual  float64
}

func splitMix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func runPath(
	request config.RunRequest,
	layer config.Layer,
	profile lifted.ClosureProfile,
	seed uint64,
	progressStepInterval int,
	onStep func(step int),
) (PathOutcome, error) {
	seed1 := splitMix64(seed)
	seed2 := splitMix64(seed1)
	rng := rand.New(rand.NewPCG(seed1, seed2))
	state, err := lifted.InitialState(request, layer, profile, rng)
	if err != nil {
		return PathOutcome{}, err
	}
	cursor, err := trajectory.New[lifted.State, lifted.StepDiagnostics](state, rng, 0)
	if err != nil {
		return PathOutcome{}, err
	}
	outcome := PathOutcome{StateDimension: state.Dimension()}
	advance := func(state *lifted.State, rng *rand.Rand) (lifted.StepDiagnostics, error) {
		return lifted.FastSlowStep(state, request, profile, rng)
	}
	observe := func(point trajectory.Point[lifted.State, lifted.StepDiagnostics]) (bool, error) {
		if point.Step > 0 {
			diagnostics := point.Diagnostics
			outcome.RewiringEvents += diagnostics.RewiringEvents
			outcome.FastSubsteps += diagnostics.FastSubsteps
			outcome.FastRewiringEvents += diagnostics.FastRewiringEvents
			if diagnostics.FastMaxHit {
				outcome.FastMaxHits++
			}
			outcome.FastSlowApplied = outcome.FastSlowApplied || diagnostics.FastSlowApplied
			outcome.FinalFastResidual = diagnostics.FastResidualIntensity
			if progressStepInterval > 0 && point.Step%progressStepInterval == 0 && onStep != nil {
				onStep(point.Step)
			}
		}
		category, done, err := terminalCategory(point.State, request)
		if err != nil {
			if point.Step == 0 {
				return false, err
			}
			return false, fmt.Errorf("classify terminal state at step %d: %w", point.Step, err)
		}
		if done {
			outcome.Category, outcome.Steps = category, point.Step
		}
		return done, nil
	}
	if err := cursor.Continue(request.MaxSteps, advance, observe); err != nil {
		return PathOutcome{}, err
	}
	if !cursor.Stopped() {
		outcome.Category, outcome.Steps = len(Categories)-1, cursor.Step()
	}
	return outcome, nil
}

func runEnsembleWithProgress(
	request config.RunRequest,
	layer config.Layer,
	profile lifted.ClosureProfile,
	paths int,
	seedNamespace uint64,
	progressStepInterval int,
	progress protocol.ProgressFunc,
) (Ensemble, error) {
	workers := min(request.Workers, paths)
	type job struct {
		index int
		seed  uint64
	}
	jobs := make(chan job)
	outcomes := make([]PathOutcome, paths)
	errChannel := make(chan error, workers)
	var group sync.WaitGroup
	var completed atomic.Int64
	var progressMutex sync.Mutex
	emit := func(event protocol.ProgressEvent) {
		if progress == nil {
			return
		}
		progressMutex.Lock()
		progress(event)
		progressMutex.Unlock()
	}
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			for item := range jobs {
				outcome, err := runPath(
					request, layer, profile, item.seed, progressStepInterval,
					func(step int) {
						emit(protocol.ProgressEvent{
							Event: "path_heartbeat", PathIndex: item.index + 1,
							TotalPaths: paths, Step: step,
						})
					},
				)
				if err != nil {
					select {
					case errChannel <- fmt.Errorf("path %d: %w", item.index, err):
					default:
					}
					continue
				}
				outcomes[item.index] = outcome
				done := int(completed.Add(1))
				category := ""
				if outcome.Category >= 0 && outcome.Category < len(Categories) {
					category = Categories[outcome.Category]
				}
				emit(protocol.ProgressEvent{
					Event: "path_completed", PathIndex: item.index + 1,
					CompletedPaths: done, TotalPaths: paths, Step: outcome.Steps,
					Category: category, StateDimension: outcome.StateDimension,
				})
			}
		}()
	}
	for index := 0; index < paths; index++ {
		jobs <- job{index: index, seed: splitMix64(request.Seed ^ seedNamespace ^ uint64(index+1))}
	}
	close(jobs)
	group.Wait()
	close(errChannel)
	if err := <-errChannel; err != nil {
		return Ensemble{}, err
	}

	result := Ensemble{
		Counts: make([]int, len(Categories)), Probabilities: make([]float64, len(Categories)),
		Paths: paths,
	}
	for _, outcome := range outcomes {
		result.Counts[outcome.Category]++
		result.MeanSteps += float64(outcome.Steps)
		result.MeanRewiringEvents += float64(outcome.RewiringEvents)
		result.MeanFastSubsteps += float64(outcome.FastSubsteps)
		result.MeanFastRewiringEvents += float64(outcome.FastRewiringEvents)
		result.FastMaxHits += outcome.FastMaxHits
		result.MeanFinalFastResidual += outcome.FinalFastResidual
		if outcome.FastSlowApplied {
			result.FastSlowAppliedPaths++
		}
		result.StateDimension = outcome.StateDimension
		if outcome.Category != len(Categories)-1 {
			result.ConvergedPaths++
		}
	}
	result.MeanSteps /= float64(paths)
	result.MeanRewiringEvents /= float64(paths)
	result.MeanFastSubsteps /= float64(paths)
	result.MeanFastRewiringEvents /= float64(paths)
	result.MeanFinalFastResidual /= float64(paths)
	for category, count := range result.Counts {
		result.Probabilities[category] = float64(count) / float64(paths)
	}
	return result, nil
}
