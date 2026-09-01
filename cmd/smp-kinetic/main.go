package main

import (
	"os"
	"smp-meso/command"
	"smp-meso/kinetic"
)

func main() {
	execute := command.NewExecutor(
		"kinetic", kinetic.DecodeRequest,
		func(request kinetic.RunRequest) (string, string) { return request.RequestID, "" },
		kinetic.RunWithProgress,
	)
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
