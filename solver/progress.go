package solver

// ProgressEvent is emitted only when a caller explicitly requests progress.
// It is operational telemetry, not part of the numerical Result.
type ProgressEvent struct {
	Event          string  `json:"event"`
	RequestID      string  `json:"request_id,omitempty"`
	Layer          string  `json:"layer,omitempty"`
	BatchIndex     int     `json:"batch_index,omitempty"`
	Stage          string  `json:"stage,omitempty"`
	ScenarioIndex  int     `json:"scenario_index,omitempty"`
	ScenarioCount  int     `json:"scenario_count,omitempty"`
	PathIndex      int     `json:"path_index,omitempty"`
	CompletedPaths int     `json:"completed_paths,omitempty"`
	TotalPaths     int     `json:"total_paths,omitempty"`
	Step           int     `json:"step,omitempty"`
	Category       string  `json:"category,omitempty"`
	StateDimension int     `json:"state_dimension,omitempty"`
	ElapsedSeconds float64 `json:"elapsed_seconds,omitempty"`
	Message        string  `json:"message,omitempty"`
}

type ProgressFunc func(ProgressEvent)
