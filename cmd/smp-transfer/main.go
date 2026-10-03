package main

import (
	"os"
	"smp-meso/command"
	"smp-meso/paired"
)

func main() {
	execute := command.NewExecutor(
		"transfer", paired.DecodeRequest,
		func(request paired.RunRequest) (string, string) { return request.RequestID, "" },
		paired.RunWithProgress,
	)
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
