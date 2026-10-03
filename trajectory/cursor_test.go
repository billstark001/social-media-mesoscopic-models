package trajectory

import (
	"math/rand/v2"
	"reflect"
	"smp-meso/lifted"
	"testing"
)

func TestContinuePreservesStateRandomStreamAndObservationOrder(t *testing.T) {
	advance := func(state *lifted.State, rng *rand.Rand) (lifted.StepDiagnostics, error) {
		draw := rng.Uint64()
		state.Rho[0] += float64(draw % 100)
		return lifted.StepDiagnostics{RewiringEvents: int(draw % 7)}, nil
	}
	type sample struct{ step, mass, events int }
	run := func(split bool) ([]sample, float64, uint64) {
		rng := rand.New(rand.NewPCG(17, 23))
		state := &lifted.State{Rho: []float64{0}}
		cursor, err := New(state, rng, 0)
		if err != nil {
			t.Fatal(err)
		}
		observed := []sample{}
		observe := func(point Point) (bool, error) {
			observed = append(observed, sample{point.Step, int(point.State.Rho[0]), point.Diagnostics.RewiringEvents})
			return false, nil
		}
		if split {
			if err := cursor.Continue(3, advance, observe); err != nil {
				t.Fatal(err)
			}
			if cursor.Step() != 3 || cursor.Stopped() {
				t.Fatalf("unexpected partial cursor: step=%d stopped=%v", cursor.Step(), cursor.Stopped())
			}
		}
		if err := cursor.Continue(10, advance, observe); err != nil {
			t.Fatal(err)
		}
		return observed, state.Rho[0], rng.Uint64()
	}
	wholeEvents, wholeState, wholeNextDraw := run(false)
	partEvents, partState, partNextDraw := run(true)
	if !reflect.DeepEqual(wholeEvents, partEvents) || wholeState != partState || wholeNextDraw != partNextDraw {
		t.Fatalf("continuation differs: whole=%v/%g/%d split=%v/%g/%d",
			wholeEvents, wholeState, wholeNextDraw, partEvents, partState, partNextDraw)
	}
	if len(partEvents) != 11 || partEvents[0].step != 0 || partEvents[len(partEvents)-1].step != 10 {
		t.Fatalf("unexpected observations: %v", partEvents)
	}
}

func TestInitialStopDoesNotAdvanceOrRepeatObservation(t *testing.T) {
	state := &lifted.State{Rho: []float64{1}}
	cursor, err := New(state, nil, 8)
	if err != nil {
		t.Fatal(err)
	}
	observations := 0
	stop := func(point Point) (bool, error) {
		observations++
		if point.Step != 8 {
			t.Fatalf("observed step %d", point.Step)
		}
		return true, nil
	}
	if err := cursor.Continue(20, nil, stop); err != nil {
		t.Fatal(err)
	}
	if err := cursor.Continue(25, nil, stop); err != nil {
		t.Fatal(err)
	}
	if cursor.Step() != 8 || !cursor.Stopped() || observations != 1 {
		t.Fatalf("initial stop changed trajectory: step=%d stopped=%v observations=%d",
			cursor.Step(), cursor.Stopped(), observations)
	}
}
