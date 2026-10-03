package paired

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/config"
	"smp-meso/lifted"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"smp-meso/terminal"
	"sync"
	"sync/atomic"
	"time"
)

var categories = []string{"k1", "k2", "k3", "k4plus", "censored"}

type Hit struct {
	Step     int    `json:"step"`
	Category string `json:"category"`
}

type PathResult struct {
	Variant
	Seed
	InitialHash    string                           `json:"initial_hash"`
	Category       string                           `json:"category"`
	Status         string                           `json:"status"`
	Steps          int                              `json:"steps"`
	ElapsedSeconds float64                          `json:"elapsed_seconds"`
	PointHit       *Hit                             `json:"point_hit"`
	FinalPoint     terminal.Result                  `json:"final_point"`
	FinalPrimary   terminal.Result                  `json:"final_primary"`
	Snapshots      map[string]protocol.EncodedArray `json:"snapshots"`
	MaxMassError   float64                          `json:"max_mass_error"`
	MaxRowError    float64                          `json:"max_row_error"`
}

type Estimate struct {
	Counts        []int     `json:"counts"`
	Probabilities []float64 `json:"probabilities"`
}

type Summary struct {
	Variant
	Paths     int      `json:"paths"`
	Primary   Estimate `json:"primary"`
	PointHit  Estimate `json:"point_hit"`
	MeanSteps float64  `json:"mean_steps"`
}

type Diagnostics struct {
	Backend        string  `json:"backend"`
	SeedScheme     string  `json:"seed_scheme"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
}

type Result struct {
	RequestID   string                `json:"request_id"`
	Categories  []string              `json:"categories"`
	Axis        protocol.EncodedArray `json:"axis"`
	Paths       []PathResult          `json:"paths"`
	Summaries   []Summary             `json:"summaries"`
	Diagnostics Diagnostics           `json:"diagnostics"`
}

func Run(request RunRequest) (Result, error) {
	return RunWithProgress(request, 0, nil)
}

// RunWithProgress returns seed-major, variant-minor output independent of worker
// scheduling. Progress is serialized and never consumes either random stream.
func RunWithProgress(request RunRequest, progressStepInterval int, progress protocol.ProgressFunc) (Result, error) {
	started := time.Now()
	if err := request.Validate(); err != nil {
		return Result{}, err
	}
	if progressStepInterval < 0 {
		return Result{}, fmt.Errorf("progress step interval must be nonnegative")
	}
	probabilities, _, err := request.Initial.Probabilities.DecodeFloat64()
	if err != nil {
		return Result{}, err
	}
	steps, err := request.snapshotSteps()
	if err != nil {
		return Result{}, err
	}
	variants := append([]Variant(nil), request.Variants...)
	models := make([]config.RunRequest, len(variants))
	for i := range variants {
		layer, err := config.ParseLayer(variants[i].Layer)
		if err != nil {
			return Result{}, err
		}
		variants[i].Layer = layer.String()
		models[i] = request.liftedRequest(layer, probabilities)
	}
	total := len(request.Seeds) * len(variants) // checked before allocating
	paths := make([]PathResult, total)
	errors := make([]error, total)
	var next atomic.Int64
	var failed atomic.Bool
	var progressMutex sync.Mutex
	completed := 0
	emit := func(event protocol.ProgressEvent) {
		if progress == nil {
			return
		}
		progressMutex.Lock()
		defer progressMutex.Unlock()
		if event.Event == "path_completed" {
			completed++
		}
		event.CompletedPaths = completed
		event.RequestID, event.Solver = request.RequestID, "transfer"
		event.TotalPaths, event.ElapsedSeconds = total, time.Since(started).Seconds()
		progress(event)
	}
	emit(protocol.ProgressEvent{Event: "request_started"})
	var group sync.WaitGroup
	for range min(request.Workers, total) {
		group.Add(1)
		go func() {
			defer group.Done()
			for !failed.Load() {
				index := int(next.Add(1)) - 1
				if index >= total {
					return
				}
				variantIndex := index % len(variants)
				variant, seed := variants[variantIndex], request.Seeds[index/len(variants)]
				heartbeat := func(step int) {
					emit(protocol.ProgressEvent{Event: "path_heartbeat", PathIndex: index + 1,
						Layer: variant.Layer, Stage: string(variant.Evolution), Step: step})
				}
				if progress == nil {
					heartbeat = nil
				}
				paths[index], errors[index] = runPath(request.Snapshots, steps, models[variantIndex], variant,
					seed, progressStepInterval, heartbeat)
				if errors[index] != nil {
					failed.Store(true)
					return
				}
				emit(protocol.ProgressEvent{Event: "path_completed", PathIndex: index + 1,
					Layer: variant.Layer, Stage: string(variant.Evolution), Step: paths[index].Steps,
					Category: paths[index].Category})
			}
		}()
	}
	group.Wait()
	for index, err := range errors {
		if err != nil {
			return Result{}, fmt.Errorf("path %d: %w", index, err)
		}
	}
	axis := make([]float64, request.OpinionBins)
	dx := (request.Initial.OpinionMax - request.Initial.OpinionMin) / float64(len(axis))
	for i := range axis {
		axis[i] = request.Initial.OpinionMin + (float64(i)+.5)*dx
	}
	encodedAxis, err := protocol.EncodeFloat64(axis, len(axis))
	if err != nil {
		return Result{}, err
	}
	result := Result{RequestID: request.RequestID, Categories: append([]string(nil), categories...),
		Axis: encodedAxis, Paths: paths, Summaries: summarize(paths, variants),
		Diagnostics: Diagnostics{Backend: numerics.ActiveBackend.Name(),
			SeedScheme:     "pcg-v1: init=(s,s^0xcafe); process=(p,p^0xbeef)",
			ElapsedSeconds: time.Since(started).Seconds()}}
	emit(protocol.ProgressEvent{Event: "request_completed"})
	return result, nil
}

func runPath(snapshots SnapshotsConfig, steps []int, request config.RunRequest, variant Variant,
	seed Seed, progressStepInterval int, heartbeat func(int)) (PathResult, error) {
	started := time.Now()
	layer, err := config.ParseLayer(variant.Layer)
	if err != nil {
		return PathResult{}, err
	}
	state, err := lifted.InitialState(request, layer, lifted.ClosureProfile{},
		rand.New(rand.NewPCG(seed.InitSeed, seed.InitSeed^0xcafe)))
	if err != nil {
		return PathResult{}, err
	}
	initial, err := json.Marshal([][]float64{state.Rho, state.Edge, state.Candidate, state.Score})
	if err != nil {
		return PathResult{}, err
	}
	hash := sha256.Sum256(initial)
	result := PathResult{Variant: variant, Seed: seed, InitialHash: hex.EncodeToString(hash[:]), Category: "censored"}
	rng := rand.New(rand.NewPCG(seed.ProcessSeed, seed.ProcessSeed^0xbeef))
	primary := terminal.Options{Epsilon: request.Dynamics.Tolerance, OccupiedMass: .5 / float64(request.Population),
		MajorMass: request.MajorClusterMass, PositionResolution: request.TerminalPositionResolution,
		MassResolution: request.TerminalMassResolution}
	point := primary
	point.PositionResolution, point.MassResolution = 0, 0
	collector := newSnapshotCollector(snapshots, steps)
	// Resolve evolution once. Both routes use the same zero closure profile and
	// unsplit law; no stochastic fast-slow or interval controls are silently ignored.
	advance := func() (float64, error) { return lifted.DeterministicStep(state, request) }
	if variant.Evolution == Stochastic {
		advance = func() (float64, error) {
			diagnostics, err := lifted.Step(state, request, lifted.ClosureProfile{}, rng)
			return diagnostics.MaxRhoChange, err
		}
	}
	for step := 0; ; step++ {
		change := 0.0
		if step > 0 {
			change, err = advance()
			if err != nil {
				return PathResult{}, fmt.Errorf("step %d: %w", step, err)
			}
		}
		result.Steps = step
		result.MaxMassError = math.Max(result.MaxMassError, math.Abs(numerics.Sum(state.Rho)-1))
		for i, mass := range state.Rho {
			residual := numerics.Sum(state.Edge[i*state.Bins:(i+1)*state.Bins]) - float64(state.Degree)*mass
			result.MaxRowError = math.Max(result.MaxRowError, math.Abs(residual))
		}
		collector.record(step, state, change)
		result.FinalPoint, err = terminal.Classify(state.Axis, state.Rho, point)
		if err != nil {
			return PathResult{}, err
		}
		if result.PointHit == nil && result.FinalPoint.Status == terminal.StatusAbsorbed {
			result.PointHit = &Hit{Step: step, Category: result.FinalPoint.Category}
		}
		result.FinalPrimary, err = terminal.Classify(state.Axis, state.Rho, primary)
		if err != nil {
			return PathResult{}, err
		}
		if heartbeat != nil && progressStepInterval > 0 && step > 0 && step%progressStepInterval == 0 {
			heartbeat(step)
		}
		if result.FinalPrimary.Status == terminal.StatusAbsorbed {
			result.Category = result.FinalPrimary.Category
			break
		}
		if step == request.MaxSteps {
			break
		}
	}
	result.Status = result.FinalPrimary.Status
	result.Snapshots, err = collector.encode(state)
	if err != nil {
		return PathResult{}, err
	}
	result.ElapsedSeconds = time.Since(started).Seconds()
	return result, nil
}

func summarize(paths []PathResult, variants []Variant) []Summary {
	result := make([]Summary, len(variants))
	for i, variant := range variants {
		summary := Summary{Variant: variant, Primary: Estimate{Counts: make([]int, len(categories))},
			PointHit: Estimate{Counts: make([]int, len(categories))}}
		for j := i; j < len(paths); j += len(variants) {
			path := paths[j]
			summary.Paths++
			summary.MeanSteps += float64(path.Steps)
			point := "censored"
			if path.PointHit != nil {
				point = path.PointHit.Category
			}
			for k, category := range categories {
				if path.Category == category {
					summary.Primary.Counts[k]++
				}
				if point == category {
					summary.PointHit.Counts[k]++
				}
			}
		}
		summary.MeanSteps /= float64(summary.Paths)
		for _, estimate := range []*Estimate{&summary.Primary, &summary.PointHit} {
			estimate.Probabilities = make([]float64, len(categories))
			for k, count := range estimate.Counts {
				estimate.Probabilities[k] = float64(count) / float64(summary.Paths)
			}
		}
		result[i] = summary
	}
	return result
}
