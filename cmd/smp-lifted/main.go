package main

import (
	"os"
	"smp-meso/command"
	"smp-meso/config"
	"smp-meso/ensemble"
)

func main() {
	execute := command.NewExecutor(
		"lifted", config.DecodeRequest,
		func(request config.RunRequest) (string, string) { return request.RequestID, request.Layer },
		ensemble.RunWithProgress,
	)
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
