package kinetic

import (
	"math"
	"smp-meso/protocol"
	"testing"
)

func TestTraceUsesProductionStepAndClosesEnergyBudget(t *testing.T) {
	for _, d := range []string{"hk", "deffuant"} {
		for _, method := range []string{"measure", "fokker_planck"} {
			r := testRequest(d, method, "opinion_random")
			r.Observables.NodeEnergy = true
			r.Observables.EdgeEnergy = true
			r.Snapshots.FinalRho = true
			r.Snapshots.FinalEdge = true
			reference, err := Run(r)
			if err != nil {
				t.Fatal(err)
			}
			trace, err := TraceFromState(r, TraceOptions{PhysicalStep: r.Dt, SnapshotSteps: []int{0, r.Steps}, EnergyBudgets: true})
			if err != nil {
				t.Fatal(err)
			}
			rho, _, _ := reference.Snapshots.FinalRho.DecodeFloat64()
			edge, _, _ := reference.Snapshots.FinalEdge.DecodeFloat64()
			last := trace.Frames[len(trace.Frames)-1]
			for i, x := range rho {
				if math.Abs(last.Rho[i]-x) > 1e-13 {
					t.Fatalf("rho parity %s/%s", d, method)
				}
			}
			for i, x := range edge {
				if math.Abs(last.Edge[i]-x) > 1e-13 {
					t.Fatalf("edge parity %s/%s", d, method)
				}
			}
			if math.Abs(last.BudgetRewire+last.BudgetMotion-last.EdgeEnergy+trace.Frames[0].EdgeEnergy) > 1e-13 {
				t.Fatal("split budget does not close")
			}
			if trace.Pathway == nil || math.Abs(*trace.Pathway-*reference.Summary.Pathway) > 1e-13 {
				t.Fatal("pathway parity")
			}
		}
	}
}
func TestProjectedStateIsCopiedAndConstraintsAreChecked(t *testing.T) {
	r := testRequest("hk", "measure", "random")
	s, err := newProjectedState(r, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rho := append([]float64(nil), s.Rho...)
	edge := append([]float64(nil), s.Edge...)
	before := edge[0]
	projection, _ := protocol.EncodeNodeEdgeProjection(rho, edge, r.OpinionBins)
	if _, err = TraceFromState(r, TraceOptions{InitialState: &projection, PhysicalStep: r.Dt, SnapshotSteps: []int{0, r.Steps}}); err != nil {
		t.Fatal(err)
	}
	if edge[0] != before {
		t.Fatal("input state mutated")
	}
	edge[0] += .1
	projection, _ = protocol.EncodeNodeEdgeProjection(rho, edge, r.OpinionBins)
	if _, err = InspectFrozenState(r, &projection); err == nil {
		t.Fatal("invalid fixed-degree row accepted")
	}
	bad := projection
	bad.Rho, _ = protocol.EncodeFloat64([]float64{1}, 1)
	if _, err = InspectFrozenState(r, &bad); err == nil {
		t.Fatal("wrong projection shape accepted")
	}
	if _, err = TraceFromState(r, TraceOptions{PhysicalStep: 1}); err == nil {
		t.Fatal("measure intervals differ")
	}
}
func TestPhysicalStepCanBeHeldFixedWhileRefiningFPE(t *testing.T) {
	r := testRequest("deffuant", "fokker_planck", "random")
	r.Dynamics.Influence = .8
	r.Dynamics.Tolerance = 1
	r.NoiseDiffusion = 0
	r.Dt = .5
	r.Steps = 1
	fixed, err := TraceFromState(r, TraceOptions{PhysicalStep: 1, SnapshotSteps: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	tied, err := TraceFromState(r, TraceOptions{PhysicalStep: .5, SnapshotSteps: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	a, b := fixed.Frames[len(fixed.Frames)-1], tied.Frames[len(tied.Frames)-1]
	difference := 0.0
	for i, x := range a.Rho {
		difference += math.Abs(x - b.Rho[i])
	}
	if difference < 1e-8 {
		t.Fatal("physical diffusion interval had no effect")
	}
	if fixed.Diagnostics.MaxFixedDegreeResidual > 1e-10 {
		t.Fatal("fixed coefficient evolution lost row constraint")
	}
}

func TestTraceContinuationRetainsMeasurementsAndDoesNotReplayStart(t *testing.T) {
	r := testRequest("deffuant", "measure", "random")
	options := TraceOptions{PhysicalStep: r.Dt, SnapshotSteps: []int{0, 2, r.Steps}, EnergyBudgets: true}
	whole, err := TraceFromState(r, options)
	if err != nil {
		t.Fatal(err)
	}
	session, err := NewTraceSession(r, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []int{0, 2, 2, r.Steps} {
		if err := session.Continue(step); err != nil {
			t.Fatal(err)
		}
	}
	chunked := session.Result()
	if len(chunked.Frames) != len(whole.Frames) {
		t.Fatal("duplicate observation on continuation")
	}
	if *chunked.Pathway != *whole.Pathway {
		t.Fatal("pathway restarted on continuation")
	}
	a, b := chunked.Frames[len(chunked.Frames)-1], whole.Frames[len(whole.Frames)-1]
	for i, x := range a.Edge {
		if x != b.Edge[i] {
			t.Fatal("state differs after continuation")
		}
	}
	if a.BudgetRewire != b.BudgetRewire || a.BudgetMotion != b.BudgetMotion {
		t.Fatal("budget restarted")
	}
}
