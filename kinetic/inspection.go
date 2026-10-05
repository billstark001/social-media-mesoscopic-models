// State inspection and trajectory diagnostics share the production step operator.
package kinetic

import (
	"fmt"
	"math"
	"math/rand/v2"
	"smp-meso/kinetic/statistics"
	"smp-meso/numerics"
	"smp-meso/protocol"
	"smp-meso/trajectory"
	"time"
)

type InspectionFrame struct {
	Step            int        `json:"step"`
	Time            float64    `json:"time"`
	NodeEnergy      float64    `json:"node_energy"`
	EdgeEnergy      float64    `json:"edge_energy"`
	ReferenceEnergy float64    `json:"reference_energy"`
	Variance        float64    `json:"variance"`
	Discordant      float64    `json:"discordant"`
	Rho             []float64  `json:"rho,omitempty"`
	Edge            []float64  `json:"edge,omitempty"`
	Velocity        []float64  `json:"velocity,omitempty"`
	NodePotential   []float64  `json:"node_potential,omitempty"`
	EdgePotential   []*float64 `json:"edge_potential,omitempty"`
	BudgetRewire    float64    `json:"budget_rewire"`
	BudgetMotion    float64    `json:"budget_motion"`
}

type TraceResult struct {
	Axis         []float64         `json:"axis"`
	Frames       []InspectionFrame `json:"frames"`
	Diagnostics  Diagnostics       `json:"diagnostics"`
	Pathway      *float64          `json:"pathway"`
	Polarization []float64         `json:"polarization,omitempty"`
	Homophily    []float64         `json:"homophily,omitempty"`
}

func newProjectedState(r RunRequest, rho, edge []float64) (*state, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	grid, err := newGrid(r)
	if err != nil {
		return nil, err
	}
	s, err := newState(r, grid)
	if err != nil {
		return nil, err
	}
	if (len(rho) == 0) != (len(edge) == 0) {
		return nil, fmt.Errorf("both projected arrays are required")
	}
	if len(rho) > 0 {
		if len(rho) != len(s.Rho) || len(edge) != len(s.Edge) {
			return nil, fmt.Errorf("projected dimensions")
		}
		copy(s.Rho, rho)
		copy(s.Edge, edge)
		s.plan.initialize(s)
	}
	return s, s.validate()
}

func DescribeState(axis, rho, edge []float64, epsilon float64, degree int) InspectionFrame {
	p := statistics.NewInteractionPlan(axis, epsilon, degree)
	u, e := p.Energies(rho, edge)
	f := InspectionFrame{NodeEnergy: u, EdgeEnergy: e}
	mean := 0.0
	for i, x := range axis {
		mean += x * rho[i]
	}
	incoming := make([]float64, len(axis))
	for i, x := range axis {
		f.Variance += rho[i] * (x - mean) * (x - mean)
		for j, y := range axis {
			incoming[j] += edge[i*len(axis)+j] / float64(degree)
			if math.Abs(x-y) > epsilon {
				f.Discordant += edge[i*len(axis)+j] / float64(degree)
			}
		}
	}
	for i, x := range axis {
		for j, y := range axis {
			f.ReferenceEnergy += .5 * math.Min((x-y)*(x-y), epsilon*epsilon) * rho[i] * incoming[j]
		}
	}
	return f
}

// TraceOptions separates the physical increment interval from the numerical step.
// InitialState uses the same compressed node/edge projection schema as lifted.
// SnapshotSteps select state inspection; scalar records follow request.RecordEvery.
// EnergyBudgets record exact changes across the implemented split, not a Lyapunov claim.
type TraceOptions struct {
	InitialState  *protocol.NodeEdgeProjection
	PhysicalStep  float64
	SnapshotSteps []int
	EnergyBudgets bool
}

// NewTraceSession retains the production state and measurement/stopping policies
// across Continue calls, like the lifted trajectory cursor. It starts from a
// fixed-degree projected state. For F-P, PhysicalStep fixes the second-moment
// coefficient while request.Dt is the numerical step. Measure requires equality.
func NewTraceSession(r RunRequest, options TraceOptions) (*TraceSession, error) {
	var rho, edge []float64
	if options.InitialState != nil {
		var err error
		rho, edge, err = options.InitialState.Decode(r.OpinionBins)
		if err != nil {
			return nil, err
		}
	}
	physicalDt, snapshotSteps, budgets := options.PhysicalStep, options.SnapshotSteps, options.EnergyBudgets
	if physicalDt*math.Max(r.Dynamics.Influence, r.Dynamics.RewiringRate) > 1+1e-12 {
		return nil, fmt.Errorf("physical increment exceeds unit interval")
	}
	if !isFiniteNonnegative(physicalDt) || physicalDt == 0 {
		return nil, fmt.Errorf("physical step must be finite and positive")
	}
	for i, k := range snapshotSteps {
		if k < 0 || k > r.Steps || (i > 0 && k <= snapshotSteps[i-1]) {
			return nil, fmt.Errorf("snapshot steps must increase within [0,steps]")
		}
	}
	checked := r
	encodedSteps := make([]float64, len(snapshotSteps))
	for i, k := range snapshotSteps {
		encodedSteps[i] = float64(k)
	}
	var err error
	checked.Snapshots.RecordSteps, err = protocol.EncodeFloat64(encodedSteps, len(encodedSteps))
	if err != nil {
		return nil, err
	}
	checked.Snapshots.Rho = true
	checked.Snapshots.Edge = true
	checked.Snapshots.Velocity = true
	checked.Snapshots.NodePotential = true
	checked.Snapshots.EdgePotential = true
	if err := checked.Validate(); err != nil {
		return nil, err
	}
	s, err := newProjectedState(r, rho, edge)
	if err != nil {
		return nil, err
	}
	if physicalDt <= 0 {
		return nil, fmt.Errorf("physical_dt must be positive")
	}
	if r.Dynamics.OpinionMethod == "measure" && math.Abs(physicalDt-r.Dt) > 1e-12 {
		return nil, fmt.Errorf("measure physical_dt must equal dt")
	}
	out := &TraceResult{Axis: append([]float64(nil), s.grid.Axis...)}
	wanted := map[int]bool{}
	for _, k := range snapshotSteps {
		wanted[k] = true
	}
	recommend := planRecommender(r)
	advance := planOpinionEvolution(r, s.grid, physicalDt)
	w := newStepWorkspace(r.OpinionBins)
	observables := statisticsPlan(r, s.grid)
	observables.ObservePathway(s.Rho, s.Edge)
	stopper := newStoppingPlan(r, s.grid.Axis, s)
	status := stoppingStatus{reason: "max_steps"}
	cumulativeRw, cumulativeMotion := 0.0, 0.0
	record := func(k int) {
		observables.Record(float64(k)*r.Dt, s.Rho, s.Edge)
		f := DescribeState(s.grid.Axis, s.Rho, s.Edge, r.Dynamics.Tolerance, r.OutDegree)
		f.Step = k
		f.Time = float64(k) * r.Dt
		f.BudgetRewire = cumulativeRw
		f.BudgetMotion = cumulativeMotion
		if wanted[k] || k == r.Steps {
			f.Rho = append([]float64(nil), s.Rho...)
			f.Edge = append([]float64(nil), s.Edge...)
			values := computeFields(s, recommend, s.plan.structuralScore(s))
			m := planMomentBuilder(r)(s, values)
			f.Velocity = make([]float64, len(s.Rho))
			for i := range f.Velocity {
				f.Velocity[i] = r.Dynamics.Influence * m.Mean[i]
			}
			np, ep := make([]float64, len(s.Rho)), make([]float64, len(s.Rho))
			statistics.NewInteractionPlan(s.grid.Axis, r.Dynamics.Tolerance, r.OutDegree).Potentials(s.Rho, s.Edge, np, ep)
			f.NodePotential = np
			f.EdgePotential = make([]*float64, len(ep))
			for i, x := range ep {
				if !math.IsNaN(x) {
					a := x
					f.EdgePotential[i] = &a
				}
			}
		}
		out.Frames = append(out.Frames, f)
	}
	nodeRes, rowRes := conservationResiduals(s)
	out.Diagnostics.MaxNodeMassResidual = nodeRes
	out.Diagnostics.MaxFixedDegreeResidual = rowRes
	energy := statistics.NewInteractionPlan(s.grid.Axis, r.Dynamics.Tolerance, r.OutDegree)
	cursor, err := trajectory.New[state, struct{}](s, nil, 0)
	if err != nil {
		return nil, err
	}
	advanceOne := func(s *state, _ *rand.Rand) (struct{}, error) {
		before, afterRw := 0.0, 0.0
		if budgets {
			_, before = energy.Energies(s.Rho, s.Edge)
		}
		onRewire := func(edge []float64) {
			if budgets {
				_, afterRw = energy.Energies(s.Rho, edge)
				cumulativeRw += afterRw - before
			}
		}
		values := computeFields(s, recommend, s.plan.structuralScore(s))
		if err := applyKineticStep(s, r, values, advance, w, onRewire); err != nil {
			return struct{}{}, err
		}
		if budgets {
			_, after := energy.Energies(s.Rho, s.Edge)
			cumulativeMotion += after - afterRw
		}
		observables.ObservePathway(s.Rho, s.Edge)
		nr, er := conservationResiduals(s)
		out.Diagnostics.MaxNodeMassResidual = math.Max(out.Diagnostics.MaxNodeMassResidual, nr)
		out.Diagnostics.MaxFixedDegreeResidual = math.Max(out.Diagnostics.MaxFixedDegreeResidual, er)
		return struct{}{}, nil
	}
	observe := func(point trajectory.Point[state, struct{}]) (bool, error) {
		k := point.Step
		if k == 0 {
			record(0)
			return false, nil
		}
		status = stopper.check(k, s)
		out.Diagnostics.ExecutedSteps = k
		if k%r.RecordEvery == 0 || wanted[k] || k == r.Steps || status.stop {
			if status.stop {
				wanted[k] = true
			}
			record(k)
		}

		out.Diagnostics.Converged = status.stop
		out.Diagnostics.StopReason = status.reason
		out.Diagnostics.FinalStateL1Rate = status.stateL1Rate
		out.Diagnostics.FinalNodeEnergyRate = status.nodeEnergyRate
		out.Diagnostics.FinalEdgeEnergyRate = status.edgeEnergyRate
		out.Diagnostics.StableSteps = status.stableSteps
		return status.stop, nil
	}
	out.Diagnostics.OpinionMethod = r.Dynamics.OpinionMethod
	out.Diagnostics.Backend = numerics.ActiveBackend.Name()
	out.Diagnostics.StateDimension = len(s.Rho) + len(s.Edge) + len(s.Wedge)
	out.Diagnostics.Recommender = normalize(r.Recommender.Type)
	return &TraceSession{cursor: cursor, result: out, advance: advanceOne, observe: observe, maximumSteps: r.Steps, observables: observables, pathway: r.Observables.Pathway}, nil
}

type FrozenFields struct {
	Axis                 []float64 `json:"axis"`
	NeighborKernel       []float64 `json:"neighbor_kernel"`
	RecommendationKernel []float64 `json:"recommendation_kernel"`
	Concordance          []float64 `json:"concordance"`
	Mean                 []float64 `json:"mean"`
	HKSecond             []float64 `json:"hk_second"`
	DeffuantSecond       []float64 `json:"deffuant_second"`
	RewiringFlux         []float64 `json:"rewiring_flux"`
	Transition           []float64 `json:"transition"`
	Eligibility          []float64 `json:"eligibility"`
}

// InspectFrozenState exposes the actual exposure closure, moments and deposited
// nonlocal transition at a frozen state without advancing or consuming an RNG.
func InspectFrozenState(r RunRequest, projection *protocol.NodeEdgeProjection) (FrozenFields, error) {
	var rho, edge []float64
	if projection != nil {
		var err error
		rho, edge, err = projection.Decode(r.OpinionBins)
		if err != nil {
			return FrozenFields{}, err
		}
	}
	s, err := newProjectedState(r, rho, edge)
	if err != nil {
		return FrozenFields{}, err
	}
	v := computeFields(s, planRecommender(r), s.plan.structuralScore(s))
	hk := hkIncrementMoments(s, v)
	d := deffuantIncrementMoments(s, v)
	quad, err := numerics.NewNormalQuadrature(r.Resolution.OpinionQuadratureRule, r.Resolution.OpinionQuadraturePoints)
	if err != nil {
		return FrozenFields{}, err
	}
	trans := hkReferenceTransition(s, v, quad)
	if r.Dynamics.Type == "deffuant" {
		trans = deffuantReferenceTransition(s, v, quad)
	}
	h := make([]float64, r.OpinionBins)
	for i := range h {
		h[i] = (1 - math.Pow(v.Neighbors.ConcordantMass[i], float64(r.OutDegree))) * (1 - math.Pow(1-v.Recommendations.ConcordantMass[i], float64(r.RecommendationCount)))
	}
	return FrozenFields{s.grid.Axis, v.Neighbors.Kernel, v.Recommendations.Kernel, s.grid.Concordance, hk.Mean, hk.Second, d.Second, v.RewiringFlux, trans, h}, nil
}

// TraceSession uses the same cursor implementation as lifted trajectories. Its
// in-memory continuation preserves stopping patience and the online path integral.
// A node/edge projection alone restarts state, not these accumulated observables.
type TraceSession struct {
	cursor       *trajectory.Cursor[state, struct{}]
	result       *TraceResult
	advance      trajectory.Stepper[state, struct{}]
	observe      trajectory.Observer[state, struct{}]
	maximumSteps int
	observables  *statistics.Plan
	pathway      bool
}

func (s *TraceSession) Continue(lastStep int) error {
	if s == nil || s.cursor == nil {
		return fmt.Errorf("nil kinetic trace session")
	}
	if lastStep > s.maximumSteps {
		return fmt.Errorf("horizon exceeds configured maximum")
	}
	started := time.Now()
	err := s.cursor.Continue(lastStep, s.advance, s.observe)
	s.result.Diagnostics.ElapsedSeconds += time.Since(started).Seconds()
	return err
}
func (s *TraceSession) Step() int { return s.cursor.Step() }
func (s *TraceSession) Result() TraceResult {
	outcome := s.observables.Outcome()
	if s.pathway {
		v := outcome.Pathway
		s.result.Pathway = &v
	}
	s.result.Polarization = outcome.Polarization
	s.result.Homophily = outcome.Homophily
	s.result.Diagnostics.RecordedPoints = len(s.result.Frames)
	if !s.result.Diagnostics.Converged {
		s.result.Diagnostics.StopReason = "max_steps"
		if s.cursor.Step() < s.maximumSteps {
			s.result.Diagnostics.StopReason = "partial_horizon"
		}
	}
	return *s.result
}

// TraceFromState is the one-shot adapter for the resumable kinetic trace session.
func TraceFromState(r RunRequest, options TraceOptions) (TraceResult, error) {
	session, err := NewTraceSession(r, options)
	if err != nil {
		return TraceResult{}, err
	}
	if err := session.Continue(r.Steps); err != nil {
		return TraceResult{}, err
	}
	return session.Result(), nil
}
