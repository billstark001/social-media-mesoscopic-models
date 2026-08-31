package protocol

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"smp-meso/config"
	"smp-meso/solver"
)

type Response struct {
	RequestID string         `json:"request_id,omitempty"`
	Result    *solver.Result `json:"result,omitempty"`
	Error     string         `json:"error,omitempty"`
}

func requestIDFromRaw(data []byte) string {
	var partial struct {
		RequestID string `json:"request_id"`
	}
	_ = json.Unmarshal(data, &partial)
	return partial.RequestID
}

func Execute(data []byte) Response {
	return ExecuteWithProgress(data, 0, nil)
}

func ExecuteWithProgress(data []byte, progressStepInterval int, progress solver.ProgressFunc) Response {
	request, err := config.DecodeRequest(data)
	if err != nil {
		requestID := requestIDFromRaw(data)
		if progress != nil {
			progress(solver.ProgressEvent{
				Event: "request_rejected", RequestID: requestID, Message: err.Error(),
			})
		}
		return Response{RequestID: requestID, Error: err.Error()}
	}
	result, err := solver.RunWithProgress(request, progressStepInterval, progress)
	if err != nil {
		if progress != nil {
			progress(solver.ProgressEvent{
				Event: "request_failed", RequestID: request.RequestID,
				Layer: request.Layer, Message: err.Error(),
			})
		}
		return Response{RequestID: request.RequestID, Error: err.Error()}
	}
	return Response{RequestID: request.RequestID, Result: &result}
}

// RunJSONL processes many parameter points in one process. A failed request is
// returned as an error response without aborting later lines.
func RunJSONL(input io.Reader, output io.Writer) error {
	return RunJSONLWithProgress(input, output, 0, nil)
}

func RunJSONLWithProgress(
	input io.Reader,
	output io.Writer,
	progressStepInterval int,
	progress solver.ProgressFunc,
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
		lineProgress := func(event solver.ProgressEvent) {
			event.BatchIndex = line
			progress(event)
		}
		if progress == nil {
			lineProgress = nil
		}
		if err := encoder.Encode(ExecuteWithProgress(data, progressStepInterval, lineProgress)); err != nil {
			return fmt.Errorf("encode response for line %d: %w", line, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read JSONL: %w", err)
	}
	return nil
}
