// Package trajectory provides a common in-memory continuation cursor. Model-specific
// state, one-step diagnostics and stopping/measurement policies remain explicit.
package trajectory

import (
	"fmt"
	"math/rand/v2"
)

// Stepper performs one full model step using the cursor's process RNG.
type Stepper[S any, D any] func(*S, *rand.Rand) (D, error)

// Point is the state after Step steps. Diagnostics is zero for the starting
// state, which is observed before any further evolution.
type Point[S any, D any] struct {
	State       *S
	Step        int
	Diagnostics D
}

// Observer receives each reached state and may stop the path. A cursor retains
// whether its current state was already observed across Continue calls.
type Observer[S any, D any] func(Point[S, D]) (stop bool, err error)

// Cursor owns the current step and retains the same state and process RNG
// across calls. Continue can extend a path's horizon without replaying earlier
// steps or drawing a new random stream.
type Cursor[S any, D any] struct {
	state    *S
	rng      *rand.Rand
	step     int
	observed bool
	stopped  bool
	failure  error
}

// New starts at an existing state, including a projected or cloned state.
// The caller must supply the RNG at its corresponding position if exact
// stochastic continuation is required; a process seed alone cannot resume it.
func New[S any, D any](state *S, rng *rand.Rand, step int) (*Cursor[S, D], error) {
	if state == nil {
		return nil, fmt.Errorf("trajectory state is nil")
	}
	if step < 0 {
		return nil, fmt.Errorf("trajectory step must be nonnegative")
	}
	return &Cursor[S, D]{state: state, rng: rng, step: step}, nil
}

func (c *Cursor[S, D]) State() *S     { return c.state }
func (c *Cursor[S, D]) Step() int     { return c.step }
func (c *Cursor[S, D]) Stopped() bool { return c.stopped }

// Continue observes the starting state once, then advances through lastStep
// inclusive unless the observer stops the path. Calling it again with a larger
// horizon continues the same state and random stream without a duplicate
// observation. The caller's stepper and observer must retain their own policy
// state (for example, terminal hits and snapshot cursors) across calls.
func (c *Cursor[S, D]) Continue(lastStep int, advance Stepper[S, D], observe Observer[S, D]) error {
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
		var initial D
		if err := c.observe(observe, initial); err != nil {
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

func (c *Cursor[S, D]) observe(observe Observer[S, D], diagnostics D) error {
	if observe == nil {
		c.observed = true
		return nil
	}
	stop, err := observe(Point[S, D]{State: c.state, Step: c.step, Diagnostics: diagnostics})
	if err != nil {
		return err
	}
	c.observed = true
	c.stopped = stop
	return nil
}
