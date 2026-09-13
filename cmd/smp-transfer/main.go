package main

import (
	"os"
	"smp-meso/command"
	"smp-meso/transfer"
)

func main() {
	execute := command.NewExecutor(
		"transfer", transfer.DecodeRequest,
		func(request transfer.RunRequest) (string, string) { return request.RequestID, "" },
		transfer.RunWithProgress,
	)
	os.Exit(command.Run(os.Args[0], os.Args[1:], execute))
}
