// Package command is the shared run/batch CLI adapter for repository binaries.
package command

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"smp-meso/protocol"
	"strings"
	"sync"
)

type options struct {
	progressMode         string
	progressStepInterval int
	arguments            []string
}

// NewExecutor adapts a typed request decoder and solver to the shared JSONL
// protocol, including consistent rejection/failure telemetry.
func NewExecutor[Request, Result any](
	solver string,
	decode func([]byte) (Request, error),
	context func(Request) (requestID, layer string),
	run func(Request, int, protocol.ProgressFunc) (Result, error),
) protocol.ExecuteFunc {
	return func(data []byte, progressStepInterval int, progress protocol.ProgressFunc) protocol.Response {
		request, err := decode(data)
		if err != nil {
			var partial struct {
				RequestID string `json:"request_id"`
			}
			_ = json.Unmarshal(data, &partial)
			if progress != nil {
				progress(protocol.ProgressEvent{
					Event: "request_rejected", RequestID: partial.RequestID,
					Solver: solver, Message: err.Error(),
				})
			}
			return protocol.Response{RequestID: partial.RequestID, Error: err.Error()}
		}
		requestID, layer := context(request)
		result, err := run(request, progressStepInterval, progress)
		if err != nil {
			if progress != nil {
				progress(protocol.ProgressEvent{
					Event: "request_failed", RequestID: requestID,
					Solver: solver, Layer: layer, Message: err.Error(),
				})
			}
			return protocol.Response{RequestID: requestID, Error: err.Error()}
		}
		return protocol.Response{RequestID: requestID, Result: &result}
	}
}

func usage(program string) {
	fmt.Fprintf(os.Stderr, "Usage:\n  %s run [--progress off|human|jsonl] [--progress-step-interval N] <JSON|-|@file>\n  %s batch [--progress off|human|jsonl] [--progress-step-interval N] < requests.jsonl > responses.jsonl\n", program, program)
}

func parseOptions(name string, arguments []string) (options, error) {
	set := flag.NewFlagSet(name, flag.ContinueOnError)
	set.SetOutput(os.Stderr)
	result := options{}
	set.StringVar(&result.progressMode, "progress", "off", "progress output: off, human, or jsonl")
	set.IntVar(&result.progressStepInterval, "progress-step-interval", 0,
		"emit a heartbeat every N numerical steps; 0 disables within-run heartbeats")
	if err := set.Parse(arguments); err != nil {
		return options{}, err
	}
	result.arguments = set.Args()
	if result.progressMode != "off" && result.progressMode != "human" && result.progressMode != "jsonl" {
		return options{}, fmt.Errorf("invalid --progress value %q", result.progressMode)
	}
	if result.progressStepInterval < 0 {
		return options{}, errors.New("--progress-step-interval must be nonnegative")
	}
	if result.progressMode == "off" && result.progressStepInterval != 0 {
		return options{}, errors.New("--progress-step-interval requires --progress human or jsonl")
	}
	return result, nil
}

func humanProgress(event protocol.ProgressEvent) {
	prefix := event.RequestID
	if event.BatchIndex > 0 {
		prefix = fmt.Sprintf("[%d] %s", event.BatchIndex, prefix)
	}
	switch event.Event {
	case "request_started":
		fmt.Fprintf(os.Stderr, "%s (%s) started\n", prefix, event.Solver)
	case "scenario_started", "scenario_completed":
		fmt.Fprintf(os.Stderr, "%s %s %d/%d (%.1fs)\n", prefix, event.Event,
			event.ScenarioIndex, event.ScenarioCount, event.ElapsedSeconds)
	case "path_heartbeat":
		fmt.Fprintf(os.Stderr, "%s %s path %d/%d step %d (%.1fs)\n", prefix,
			event.Stage, event.PathIndex, event.TotalPaths, event.Step, event.ElapsedSeconds)
	case "path_completed":
		fmt.Fprintf(os.Stderr, "%s %s paths %d/%d; last=%s step=%d (%.1fs)\n", prefix,
			event.Stage, event.CompletedPaths, event.TotalPaths, event.Category,
			event.Step, event.ElapsedSeconds)
	case "step_heartbeat":
		fmt.Fprintf(os.Stderr, "%s step %d/%d (%.1fs)\n", prefix, event.Step,
			event.TotalSteps, event.ElapsedSeconds)
	case "request_completed":
		fmt.Fprintf(os.Stderr, "%s completed (%.1fs)\n", prefix, event.ElapsedSeconds)
	case "request_rejected", "request_failed":
		fmt.Fprintf(os.Stderr, "%s %s: %s\n", prefix, event.Event, event.Message)
	default:
		fmt.Fprintf(os.Stderr, "%s %s (%.1fs)\n", prefix, event.Event, event.ElapsedSeconds)
	}
}

func progressReporter(mode string) protocol.ProgressFunc {
	if mode == "off" {
		return nil
	}
	var mutex sync.Mutex
	encoder := json.NewEncoder(os.Stderr)
	return func(event protocol.ProgressEvent) {
		mutex.Lock()
		defer mutex.Unlock()
		if mode == "jsonl" {
			_ = encoder.Encode(event)
			return
		}
		humanProgress(event)
	}
}

func readRequest(argument string) ([]byte, error) {
	switch {
	case argument == "-":
		return io.ReadAll(os.Stdin)
	case strings.HasPrefix(argument, "@"):
		return os.ReadFile(strings.TrimPrefix(argument, "@"))
	default:
		return []byte(argument), nil
	}
}

// Run executes a repository command and returns the intended process exit code.
func Run(program string, arguments []string, execute protocol.ExecuteFunc) int {
	if len(arguments) < 1 {
		usage(program)
		return 2
	}
	switch arguments[0] {
	case "run":
		parsed, err := parseOptions("run", arguments[1:])
		if err != nil || len(parsed.arguments) != 1 {
			usage(program)
			return 2
		}
		data, err := readRequest(parsed.arguments[0])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		response := execute(data, parsed.progressStepInterval, progressReporter(parsed.progressMode))
		if err := json.NewEncoder(os.Stdout).Encode(response); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		if response.Error != "" {
			return 1
		}
		return 0
	case "batch":
		parsed, err := parseOptions("batch", arguments[1:])
		if err != nil || len(parsed.arguments) != 0 {
			usage(program)
			return 2
		}
		if err := protocol.RunJSONLWithProgress(os.Stdin, os.Stdout, execute,
			parsed.progressStepInterval, progressReporter(parsed.progressMode)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		return 0
	default:
		usage(program)
		return 2
	}
}
