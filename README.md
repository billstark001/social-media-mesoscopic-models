# social-media-mesoscopic-models

Finite-`N` stochastic mesoscopic terminal-probability solver for the social
media opinion model. The implementation is deliberately split into ordinary
Go packages (there is no `internal` tree):

- `config`: strict, explicit request schema and validation;
- `numerics`: probability utilities and optional dense BLAS contractions;
- `meso`: the six nested retained states, unsplit law, and conditional fast-absorption law;
- `solver`: path ensembles, absorbing terminal categories, and uncertainty
  envelopes;
- `protocol`: recoverable JSONL batch transport;
- `cmd/smp-meso`: the small command-line adapter.

The Python package `smp_meso_bindings` builds and orchestrates the Go binary.
It sends all parameter points in one JSONL stream, so a scan does not create a
new process for every point.

## Retained-state ladder

All masses use a per-agent normalization. The six nested layers are:

| request value | retained coordinates beyond `rho,E` | ambiguity removed from the preceding layer |
|---|---|---|
| `naive` / `rho_edge` | none; `C,S_zeta` are reconstructed workspaces, not transported state | none |
| `base` / `l0` | candidate mass `C` and the requested powered score mass `S_zeta` | none |
| `wedge` / `l1` | candidate-weighted centre-opinion wedge `W` | first-score/motif allocation |
| `histogram` / `l2` | agent histogram `H_i(k,d,c)` | discordant-followee/concordant-feed eligibility coupling |
| `candidate` / `l3` | score/availability measure `Xi_ij(a,s)` | candidate availability--score coupling |
| `topology` / `l4` | component-size marks and bridge-edge mass | coarse fast absorbing-network class |

This is an information-sufficiency ladder, not a claim that the final layer is
an exact lumping of the microscopic Markov chain. Each retained coordinate is
transported explicitly, while its rewiring-created higher hierarchy is relaxed
toward an independently reconstructed target using the supplied closure rates.
For the `wedge` and `histogram` layers, rewiring updates `S_zeta` from the
resolved first moment in `W`: the projection is exact at `zeta=1`, while higher
powers preserve the pre-rewiring conditional shape factor
`S_zeta / S_1^zeta`. Candidate and topology states instead project `C`,
`S_zeta`, and the first moment of `W` from the richer `Xi` coordinate.

Layer-dependent work is resolved once per simulated path by a cumulative
layer plan. Each successive plan inherits its predecessor and replaces only
the newly refined numerical stages, so plan definitions do not repeat and
the time-critical kernels contain no per-coordinate layer checks.

The request also selects `fast_slow.mode`. `unsplit` applies the ordinary
synchronous generator. `conditional_absorption` is activated only when
`rewiring_rate / influence` reaches the explicit `ratio_threshold`; at fixed
`rho` it repeatedly samples rewiring until the conditional intensity vanishes
or an explicit safety rule fires, and then takes one opinion step. This is a
sampled non-ergodic absorbing-class projection, not a stationary average over
a presumed unique fast invariant distribution. Below threshold it calls the
unsplit step exactly, including the same random stream.

The reported point estimate is the empirical probability vector in the order
`[k1,k2,k3,k4plus,censored]`. The interval result contains two distinct
widths:

- `closure_width`: envelope over sampled unresolved closure profiles;
- `width`: the closure envelope enlarged by Wilson Monte Carlo bounds.

The closure envelope is a finite sensitivity audit over the requested profiles,
not a certified optimum over a continuous ambiguity set. Wilson bounds use the
requested confidence level separately for each scenario and terminal category;
they are marginal bounds, without a simultaneous or multiple-comparison
correction. Results include every sampled closure profile and its probability
vector so downstream audits can reconstruct the envelope.

## Build

The dependency-free build uses the native Go contraction loops:

```sh
go build -o bin/smp-meso ./cmd/smp-meso
go test ./...
```

On macOS, Accelerate is optional:

```sh
go build -tags accelerate -o bin/smp-meso ./cmd/smp-meso
go test -tags accelerate ./...
```

OpenBLAS is also optional. With Homebrew, the Python build helper discovers
its prefix; a direct Go build can instead provide the usual CGO include and
linker flags:

```sh
OPENBLAS_PREFIX="$(brew --prefix openblas)"
CGO_CFLAGS="-I${OPENBLAS_PREFIX}/include" \
CGO_LDFLAGS="-L${OPENBLAS_PREFIX}/lib -lopenblas" \
go build -tags openblas -o bin/smp-meso ./cmd/smp-meso
```

The `openblas` and `accelerate` tags are mutually exclusive. Backend selection
changes only dense contractions, not the model or random streams.

## Request and CLI

No model or numerical parameter is defaulted. Even parameters irrelevant to a
selected recommender must be present; this makes saved requests complete and
auditable. A request has this shape:

```json
{
  "request_id": "example",
  "layer": "candidate",
  "population": 500,
  "opinion_bins": 15,
  "out_degree": 15,
  "recommendation_count": 10,
  "max_steps": 20000,
  "paths": 40,
  "interval_paths": 40,
  "ambiguity_samples": 11,
  "confidence_level": 0.95,
  "workers": 8,
  "seed": 20260818,
  "major_cluster_mass": 0.002,
  "dynamics": {
    "type": "hk",
    "tolerance": 0.45,
    "influence": 0.046415888336127795,
    "rewiring_rate": 0.046415888336127795
  },
  "recommender": {
    "type": "structure_random",
    "steepness": 4.0,
    "random_ratio": 0.0,
    "opinion_tolerance": 0.4,
    "noise_std": 0.1,
    "noise_quadrature_points": 9
  },
  "initial": {
    "type": "uniform",
    "opinion_min": -1.0,
    "opinion_max": 1.0,
    "probabilities": []
  },
  "resolution": {
    "score_max": 45,
    "availability_bins": 11,
    "component_size_bins": 12,
    "opinion_quadrature_points": 7
  },
  "closure": {
    "motif_relaxation": 0.25,
    "histogram_relaxation": 0.25,
    "candidate_relaxation": 0.25,
    "topology_relaxation": 0.25
  },
  "fast_slow": {
    "mode": "unsplit",
    "ratio_threshold": 10.0,
    "max_substeps": 400,
    "zero_event_batches": 8,
    "residual_tolerance": 1e-12,
    "zero_event_residual": 0.25
  },
  "ambiguity": {
    "eligibility_correlation_radius": 0.75,
    "score_availability_radius": 0.75,
    "motif_persistence_radius": 0.75,
    "bridge_bias_radius": 0.75,
    "component_mix_radius": 0.75
  }
}
```

Supported `dynamics.type` values are `hk` and `deffuant`. Supported
`recommender.type` values are `random`, `opinion_random`, and
`structure_random`; both weighted recommenders use `steepness`.
`noise_std` and `noise_quadrature_points` affect `opinion_random`. They are
explicit but inert for `random` and `structure_random`, matching the microscopic
weighted StructureRandom implementation, which samples from unperturbed raw
common-neighbour counts. OpinionRandom noise is integrated by deterministic
normal quadrature; retaining the realised microscopic noisy weight matrix would
require an additional state coordinate.

Supported `fast_slow.mode` values are `unsplit` and
`conditional_absorption`. All stopping controls are required even for
`unsplit`, keeping saved requests complete. Point diagnostics report applied
paths, fast substeps, fast rewiring events, safety-cap hits, and final residual
conditional intensity.

Run one JSON object or a batch of newline-delimited objects. The quiet form
keeps stdout as result-only JSON/JSONL:

```sh
bin/smp-meso run @request.json
bin/smp-meso batch < requests.jsonl > responses.jsonl
```

A malformed item in batch mode yields an error response for that line and does
not abort subsequent items.

Progress is opt-in and always goes to stderr. `human` is suitable for a
terminal; `jsonl` is stable machine-readable telemetry. A positive heartbeat
interval additionally reports paths that have not yet terminated:

```sh
bin/smp-meso run --progress human --progress-step-interval 1000 @request.json
bin/smp-meso batch --progress jsonl --progress-step-interval 1000 \
  < requests.jsonl > responses.jsonl 2> progress.jsonl
```

Events identify the request and batch line, point/interval stage, interval
scenario, completed paths, last terminal category, elapsed time, and periodic
within-path step. Enabling telemetry does not change random streams.

## Python orchestration

Install from the repository, including editable installation:

```sh
python -m pip install -e .
smp-meso-build --backend purego
# or: smp-meso-build --backend accelerate
# or: smp-meso-build --backend openblas
```

Then reuse one Go process for a list of requests:

```python
from smp_meso_bindings import (
    build_binary,
    print_progress,
    run_batch,
    run_batch_parallel,
)

binary = build_binary(backend="accelerate")
responses = run_batch(
    binary,
    requests,
    progress=print_progress,
    progress_step_interval=1000,
)

# Four long-lived Go batch processes with dynamic request scheduling;
# response order remains input order.
# Set each request's explicit workers value with the total CPU budget in mind.
responses = run_batch_parallel(
    binary,
    requests,
    4,
    progress=print_progress,
    progress_step_interval=1000,
)
```

`run_batch(..., check=False)` returns both successful and error responses;
the default raises `BatchExecutionError` after collecting the complete batch.
`run_batch` remains the clean single-process execution path. The parallel
orchestrator uses Python threads only to supervise independent Go processes;
the numerical work remains in Go. A free process pulls the next parameter point
from a shared queue, avoiding the long tail of a static partition when hitting
times differ strongly. Budget approximately
`processes × request.workers` runnable paths to avoid oversubscription.
