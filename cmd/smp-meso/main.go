package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"smp-meso/protocol"
	"smp-meso/solver"
	"strings"
	"sync"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage:\n  %s run [--progress off|human|jsonl] [--progress-step-interval N] <JSON|-|@file>\n  %s batch [--progress off|human|jsonl] [--progress-step-interval N] < requests.jsonl > responses.jsonl\n", os.Args[0], os.Args[0])
}

type commandOptions struct {
	progressMode         string
	progressStepInterval int
	arguments            []string
}

func parseOptions(command string, arguments []string) (commandOptions, error) {
	set := flag.NewFlagSet(command, flag.ContinueOnError)
	set.SetOutput(os.Stderr)
	options := commandOptions{}
	set.StringVar(&options.progressMode, "progress", "off", "progress output: off, human, or jsonl")
	set.IntVar(&options.progressStepInterval, "progress-step-interval", 0,
		"emit a heartbeat every N path steps; 0 disables within-path heartbeats")
	if err := set.Parse(arguments); err != nil {
		return commandOptions{}, err
	}
	options.arguments = set.Args()
	if options.progressMode != "off" && options.progressMode != "human" && options.progressMode != "jsonl" {
		return commandOptions{}, fmt.Errorf("invalid --progress value %q", options.progressMode)
	}
	if options.progressStepInterval < 0 {
		return commandOptions{}, errors.New("--progress-step-interval must be nonnegative")
	}
	if options.progressMode == "off" && options.progressStepInterval != 0 {
		return commandOptions{}, errors.New("--progress-step-interval requires --progress human or jsonl")
	}
	return options, nil
}

func progressReporter(mode string) solver.ProgressFunc {
	if mode == "off" {
		return nil
	}
	var mutex sync.Mutex
	encoder := json.NewEncoder(os.Stderr)
	return func(event solver.ProgressEvent) {
		mutex.Lock()
		defer mutex.Unlock()
		if mode == "jsonl" {
			_ = encoder.Encode(event)
			return
		}
		prefix := event.RequestID
		if event.BatchIndex > 0 {
			prefix = fmt.Sprintf("[%d] %s", event.BatchIndex, prefix)
		}
		switch event.Event {
		case "request_started":
			fmt.Fprintf(os.Stderr, "%s (%s) started\n", prefix, event.Layer)
		case "scenario_started":
			fmt.Fprintf(os.Stderr, "%s interval scenario %d/%d started\n",
				prefix, event.ScenarioIndex, event.ScenarioCount)
		case "path_heartbeat":
			fmt.Fprintf(os.Stderr, "%s %s", prefix, event.Stage)
			if event.ScenarioIndex > 0 {
				fmt.Fprintf(os.Stderr, " scenario %d/%d", event.ScenarioIndex, event.ScenarioCount)
			}
			fmt.Fprintf(os.Stderr, " path %d/%d step %d (%.1fs)\n",
				event.PathIndex, event.TotalPaths, event.Step, event.ElapsedSeconds)
		case "path_completed":
			fmt.Fprintf(os.Stderr, "%s %s", prefix, event.Stage)
			if event.ScenarioIndex > 0 {
				fmt.Fprintf(os.Stderr, " scenario %d/%d", event.ScenarioIndex, event.ScenarioCount)
			}
			fmt.Fprintf(os.Stderr, " paths %d/%d; last=%s at step %d (%.1fs)\n",
				event.CompletedPaths, event.TotalPaths, event.Category, event.Step, event.ElapsedSeconds)
		case "scenario_completed":
			fmt.Fprintf(os.Stderr, "%s interval scenario %d/%d completed (%.1fs)\n",
				prefix, event.ScenarioIndex, event.ScenarioCount, event.ElapsedSeconds)
		case "request_completed":
			fmt.Fprintf(os.Stderr, "%s completed: %d paths, dimension %d (%.1fs)\n",
				prefix, event.CompletedPaths, event.StateDimension, event.ElapsedSeconds)
		case "request_rejected", "request_failed":
			fmt.Fprintf(os.Stderr, "%s %s: %s\n", prefix, event.Event, event.Message)
		default:
			fmt.Fprintf(os.Stderr, "%s %s (%.1fs)\n", prefix, event.Event, event.ElapsedSeconds)
		}
	}
}

func readRequestArgument(argument string) ([]byte, error) {
	switch {
	case argument == "-":
		return io.ReadAll(os.Stdin)
	case strings.HasPrefix(argument, "@"):
		return os.ReadFile(strings.TrimPrefix(argument, "@"))
	default:
		return []byte(argument), nil
	}
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "run":
		options, err := parseOptions("run", os.Args[2:])
		if err != nil || len(options.arguments) != 1 {
			usage()
			os.Exit(2)
		}
		data, err := readRequestArgument(options.arguments[0])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		response := protocol.ExecuteWithProgress(
			data, options.progressStepInterval, progressReporter(options.progressMode),
		)
		if err := json.NewEncoder(os.Stdout).Encode(response); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if response.Error != "" {
			os.Exit(1)
		}
	case "batch":
		options, err := parseOptions("batch", os.Args[2:])
		if err != nil || len(options.arguments) != 0 {
			usage()
			os.Exit(2)
		}
		if err := protocol.RunJSONLWithProgress(
			os.Stdin, os.Stdout, options.progressStepInterval,
			progressReporter(options.progressMode),
		); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		usage()
		os.Exit(2)
	}
}
