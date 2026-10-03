// Package trajectory advances a lifted state along one path. Experiment
// packages supply the numerical step and decide what to observe or when to stop.
package trajectory

import (
	"fmt"
	"math/rand/v2"
	"smp-meso/lifted"
)

// Stepper performs one full model step using the cursor's process RNG.
type Stepper func(*lifted.State, *rand.Rand) (lifted.StepDiagnostics, error)

// Point is the state after Step steps. Diagnostics is zero for the starting
// state, which is observed before any further evolution.
type Point struct {
	State       *lifted.State
	Step        int
	Diagnostics lifted.StepDiagnostics
}

// Observer receives each reached state and may stop the path. A cursor retains
// whether its current state was already observed across Continue calls.
type Observer func(Point) (stop bool, err error)

// Cursor owns the current step and retains the same state and process RNG
// across calls. Continue can extend a path's horizon without replaying earlier
// steps or drawing a new random stream.
type Cursor struct {
	state    *lifted.State
	rng      *rand.Rand
	step     int
	observed bool
	stopped  bool
	failure  error
}

// New starts at an existing state, including a projected or cloned state.
// The caller must supply the RNG at its corresponding position if exact
// stochastic continuation is required; a process seed alone cannot resume it.
func New(state *lifted.State, rng *rand.Rand, step int) (*Cursor, error) {
	if state == nil {
		return nil, fmt.Errorf("trajectory state is nil")
	}
	if step < 0 {
		return nil, fmt.Errorf("trajectory step must be nonnegative")
	}
	return &Cursor{state: state, rng: rng, step: step}, nil
}

func (c *Cursor) State() *lifted.State { return c.state }
func (c *Cursor) Step() int            { return c.step }
func (c *Cursor) Stopped() bool        { return c.stopped }

// Continue observes the starting state once, then advances through lastStep
// inclusive unless the observer stops the path. Calling it again with a larger
// horizon continues the same state and random stream without a duplicate
// observation. The caller's stepper and observer must retain their own policy
// state (for example, terminal hits and snapshot cursors) across calls.
func (c *Cursor) Continue(lastStep int, advance Stepper, observe Observer) error {
	if c == nil || c.state == nil {
		return fmt.Errorf("trajectory cursor has no state")
	}
	if lastStep < c.step {
		return fmt.Errorf("trajectory horizon %d precedes current step %d", lastStep, c.step)
	}
	if c.failure != nil {
		return c.failure
	}
	if c.stopped {
		return nil
	}
	if !c.observed {
		if err := c.observe(observe, lifted.StepDiagnostics{}); err != nil {
			c.failure = err
			return err
		}
	}
	if c.stopped {
		return nil
	}
	if lastStep > c.step && advance == nil {
		return fmt.Errorf("trajectory stepper is nil")
	}
	for !c.stopped && c.step < lastStep {
		next := c.step + 1
		diagnostics, err := advance(c.state, c.rng)
		if err != nil {
			c.failure = fmt.Errorf("step %d: %w", next, err)
			return c.failure
		}
		c.step = next
		if err := c.observe(observe, diagnostics); err != nil {
			c.failure = err
			return err
		}
	}
	return nil
}

func (c *Cursor) observe(observe Observer, diagnostics lifted.StepDiagnostics) error {
	if observe == nil {
		c.observed = true
		return nil
	}
	stop, err := observe(Point{State: c.state, Step: c.step, Diagnostics: diagnostics})
	if err != nil {
		return err
	}
	c.observed = true
	c.stopped = stop
	return nil
}
