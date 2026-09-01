package main

import (
	"encoding/json"
	"os"
	"smp-meso/command"
	"smp-meso/kinetic"
	"smp-meso/protocol"
)

func requestIDFromRaw(data []byte) string {
	var partial struct {
		RequestID string `json:"request_id"`
	}
	_ = json.Unmarshal(data, &partial)
	return partial.RequestID
}

func execute(data []byte, progressStepInterval int, progress protocol.ProgressFunc) protocol.Response {
	request, err := kinetic.DecodeRequest(data)
	if err != nil {
		requestID := requestIDFromRaw(data)
		if progress != nil {
			progress(protocol.ProgressEvent{Event: "request_rejected", RequestID: requestID, Solver: "kinetic", Message: err.Error()})
		}
		return protocol.Response{RequestID: requestID, Error: err.Error()}
	}
	result, err := kinetic.RunWithProgress(request, progressStepInterval, progress)
	if err != nil {
		if progress != nil {
			progress(protocol.ProgressEvent{Event: "request_failed", RequestID: request.RequestID, Solver: "kinetic", Message: err.Error()})
		}
		return protocol.Response{RequestID: request.RequestID, Error: err.Error()}
	}
	return protocol.Response{RequestID: request.RequestID, Result: &result}
}

func main() {
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
