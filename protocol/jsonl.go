// Package protocol implements the shared process transport used by every
// mesoscopic command in this repository. Model-specific decoding and numerical
// execution stay behind ExecuteFunc.
package protocol

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
)

// ProgressEvent is operational telemetry, not part of a numerical result.
type ProgressEvent struct {
	Event          string  `json:"event"`
	RequestID      string  `json:"request_id,omitempty"`
	Solver         string  `json:"solver,omitempty"`
	Layer          string  `json:"layer,omitempty"`
	BatchIndex     int     `json:"batch_index,omitempty"`
	Stage          string  `json:"stage,omitempty"`
	ScenarioIndex  int     `json:"scenario_index,omitempty"`
	ScenarioCount  int     `json:"scenario_count,omitempty"`
	PathIndex      int     `json:"path_index,omitempty"`
	CompletedPaths int     `json:"completed_paths,omitempty"`
	TotalPaths     int     `json:"total_paths,omitempty"`
	Step           int     `json:"step,omitempty"`
	TotalSteps     int     `json:"total_steps,omitempty"`
	Category       string  `json:"category,omitempty"`
	StateDimension int     `json:"state_dimension,omitempty"`
	ElapsedSeconds float64 `json:"elapsed_seconds,omitempty"`
	Message        string  `json:"message,omitempty"`
}

type ProgressFunc func(ProgressEvent)

// Response is deliberately model-neutral. Numerical arrays inside Result use
// EncodedArray rather than JSON arrays.
type Response struct {
	RequestID string `json:"request_id,omitempty"`
	Result    any    `json:"result,omitempty"`
	Error     string `json:"error,omitempty"`
}

type ExecuteFunc func([]byte, int, ProgressFunc) Response

func RunJSONL(input io.Reader, output io.Writer, execute ExecuteFunc) error {
	return RunJSONLWithProgress(input, output, execute, 0, nil)
}

// RunJSONLWithProgress processes many parameter points in one process. A
// failed request is represented by an error response and does not abort later
// lines.
func RunJSONLWithProgress(
	input io.Reader,
	output io.Writer,
	execute ExecuteFunc,
	progressStepInterval int,
	progress ProgressFunc,
) error {
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	encoder := json.NewEncoder(output)
	line := 0
	for scanner.Scan() {
		line++
		data := append([]byte(nil), scanner.Bytes()...)
		if len(data) == 0 {
			continue
		}
		lineProgress := func(event ProgressEvent) {
			event.BatchIndex = line
			progress(event)
		}
		if progress == nil {
			lineProgress = nil
		}
		if err := encoder.Encode(execute(data, progressStepInterval, lineProgress)); err != nil {
			return fmt.Errorf("encode response for line %d: %w", line, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read JSONL: %w", err)
	}
	return nil
}
