package main

import (
	"encoding/json"
	"os"
	"smp-meso/command"
	"smp-meso/config"
	"smp-meso/protocol"
	"smp-meso/solver"
)

func requestIDFromRaw(data []byte) string {
	var partial struct {
		RequestID string `json:"request_id"`
	}
	_ = json.Unmarshal(data, &partial)
	return partial.RequestID
}

func execute(data []byte, progressStepInterval int, progress protocol.ProgressFunc) protocol.Response {
	request, err := config.DecodeRequest(data)
	if err != nil {
		requestID := requestIDFromRaw(data)
		if progress != nil {
			progress(protocol.ProgressEvent{
				Event: "request_rejected", RequestID: requestID,
				Solver: "lifted", Message: err.Error(),
			})
		}
		return protocol.Response{RequestID: requestID, Error: err.Error()}
	}
	result, err := solver.RunWithProgress(request, progressStepInterval, progress)
	if err != nil {
		if progress != nil {
			progress(protocol.ProgressEvent{
				Event: "request_failed", RequestID: request.RequestID,
				Solver: "lifted", Layer: request.Layer, Message: err.Error(),
			})
		}
		return protocol.Response{RequestID: request.RequestID, Error: err.Error()}
	}
	return protocol.Response{RequestID: request.RequestID, Result: &result}
}

func main() {
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
