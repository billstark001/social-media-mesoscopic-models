package kinetic

import (
	"fmt"
	"math"
	"smp-meso/numerics"
)

type stepWorkspace struct {
	rhoNext     []float64
	edgeNext    []float64
	edgeScratch []float64
	edgeRewired []float64
}

func newStepWorkspace(size int) *stepWorkspace {
	return &stepWorkspace{
		rhoNext: make([]float64, size), edgeNext: make([]float64, size*size),
		edgeScratch: make([]float64, size*size), edgeRewired: make([]float64, size*size),
	}
}

type opinionEvolution func(*state, fields, []float64, *stepWorkspace) error

func applyPDE(current *state, edgeSource []float64, workspace *stepWorkspace, system *numerics.TridiagonalSystem) error {
	if err := numerics.ActiveBackend.ApplyTridiagonal(workspace.rhoNext, current.Rho, system); err != nil {
		return err
	}
	if err := numerics.ActiveBackend.TransportTridiagonalMatrix(
		workspace.edgeNext, workspace.edgeScratch, edgeSource, system,
	); err != nil {
		return err
	}
	if err := current.plan.transportPDE(current, system); err != nil {
		return err
	}
	current.Rho, workspace.rhoNext = workspace.rhoNext, current.Rho
	current.Edge, workspace.edgeNext = workspace.edgeNext, current.Edge
	return nil
}

func constantDiffusionSystem(request RunRequest, grid *gridGeometry) (*numerics.TridiagonalSystem, error) {
	diffusion := make([]float64, request.OpinionBins)
	for index := range diffusion {
		diffusion[index] = request.NoiseDiffusion
	}
	return fokkerPlanckSystem(make([]float64, request.OpinionBins), diffusion, grid.Dx, request.Dt)
}

func planOpinionEvolution(request RunRequest, grid *gridGeometry) opinionEvolution {
	if normalize(request.Dynamics.OpinionMethod) == "measure" {
		buildTransition := planMeasureTransition(request)
		var background *numerics.TridiagonalSystem
		var backgroundErr error
		applyBackground := func(*state, *stepWorkspace) error { return nil }
		if request.NoiseDiffusion > 0 {
			background, backgroundErr = constantDiffusionSystem(request, grid)
			applyBackground = func(current *state, workspace *stepWorkspace) error {
				if backgroundErr != nil {
					return backgroundErr
				}
				return applyPDE(current, current.Edge, workspace, background)
			}
		}
		return func(current *state, values fields, edgeRewired []float64, workspace *stepWorkspace) error {
			transition, err := buildTransition(current, values)
			if err != nil {
				return err
			}
			numerics.ActiveBackend.ApplyTransition(workspace.rhoNext, current.Rho, transition)
			numerics.ActiveBackend.SandwichTransition(workspace.edgeNext, workspace.edgeScratch, edgeRewired, transition)
			current.plan.transportMeasure(current, transition)
			current.Rho, workspace.rhoNext = workspace.rhoNext, current.Rho
			current.Edge, workspace.edgeNext = workspace.edgeNext, current.Edge
			return applyBackground(current, workspace)
		}
	}
	buildMoments := planMomentBuilder(request)
	return func(current *state, values fields, edgeRewired []float64, workspace *stepWorkspace) error {
		moments := buildMoments(current, values)
		velocity := make([]float64, request.OpinionBins)
		diffusion := make([]float64, request.OpinionBins)
		for index := range velocity {
			velocity[index] = request.Dynamics.Influence * moments.Mean[index]
			diffusion[index] = request.NoiseDiffusion +
				0.5*request.Dt*request.Dynamics.Influence*request.Dynamics.Influence*moments.Second[index]
			if !isFiniteNonnegative(diffusion[index]) || math.IsNaN(velocity[index]) || math.IsInf(velocity[index], 0) {
				return fmt.Errorf("invalid Fokker-Planck coefficient at cell %d", index)
			}
		}
		system, err := fokkerPlanckSystem(velocity, diffusion, grid.Dx, request.Dt)
		if err != nil {
			return err
		}
		return applyPDE(current, edgeRewired, workspace, system)
	}
}

func isFiniteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

// fokkerPlanckSystem is the backward-Euler no-flux finite-volume system for
// d_t rho = -d_x(v rho) + d_xx(D rho). Upwind drift and cell-sided diffusion
// make the generator conservative and Metzler.
func fokkerPlanckSystem(velocity, diffusion []float64, dx, dt float64) (*numerics.TridiagonalSystem, error) {
	size := len(velocity)
	lower := make([]float64, size-1)
	upper := make([]float64, size-1)
	diagonal := make([]float64, size)
	for index := range diagonal {
		diagonal[index] = 1
	}
	for face := 0; face < size-1; face++ {
		faceVelocity := 0.5 * (velocity[face] + velocity[face+1])
		leftToRight := math.Max(faceVelocity, 0)/dx + diffusion[face]/(dx*dx)
		rightToLeft := math.Max(-faceVelocity, 0)/dx + diffusion[face+1]/(dx*dx)
		lower[face] = -dt * leftToRight
		upper[face] = -dt * rightToLeft
		diagonal[face] += dt * leftToRight
		diagonal[face+1] += dt * rightToLeft
	}
	return numerics.NewTridiagonalSystem(lower, diagonal, upper)
}
