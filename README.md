# social-media-mesoscopic-models

Go runtimes for two mesoscopic reductions of the social-media opinion model:

- `smp-lifted`: finite-`N` stochastic lifted terminal probabilities;
- `smp-kinetic`: deterministic measure or strong-form Fokker--Planck
  evolution with online density observables and no full-trajectory output.

The implementation is deliberately split into ordinary Go packages (there is
no `internal` tree):

- `config`: strict, explicit request schema and validation;
- `numerics`: probability utilities, canonical sparse/dense transport,
  batched contractions, and reusable tridiagonal factorizations;
- `lifted`: the six nested retained states, unsplit law, and conditional fast-absorption law;
- `solver`: lifted path ensembles, absorbing terminal categories, and uncertainty
  envelopes;
- `kinetic`: nonlocal measure and finite-volume Fokker--Planck dynamics;
- `kinetic/statistics`: requested-only online density observables;
- `protocol`: recoverable JSONL batch transport;
- `command`: the shared run/batch command adapter;
- `cmd/smp-lifted` and `cmd/smp-kinetic`: thin executable adapters.

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
go build -o bin/smp-lifted ./cmd/smp-lifted
go build -o bin/smp-kinetic ./cmd/smp-kinetic
go test ./...
```

On macOS, Accelerate is optional:

```sh
go build -tags accelerate -o bin/smp-lifted ./cmd/smp-lifted
go build -tags accelerate -o bin/smp-kinetic ./cmd/smp-kinetic
go test -tags accelerate ./...
```

OpenBLAS is also optional. With Homebrew, the Python build helper discovers
its prefix; a direct Go build can instead provide the usual CGO include and
linker flags:

```sh
OPENBLAS_PREFIX="$(brew --prefix openblas)"
CGO_CFLAGS="-I${OPENBLAS_PREFIX}/include" \
CGO_LDFLAGS="-L${OPENBLAS_PREFIX}/lib -lopenblas" \
go build -tags openblas -o bin/smp-lifted ./cmd/smp-lifted
go build -tags openblas -o bin/smp-kinetic ./cmd/smp-kinetic
```

The `openblas` and `accelerate` tags are mutually exclusive. They select native
BLAS for dense contractions and native LAPACK for reusable tridiagonal
factorizations and batched solves. Sparse transition kernels remain sparse when
their density is at most 25%; denser operators use BLAS. Both representations
are built from the same pruned, row-normalized transition, so backend selection
does not change the model or random streams, although floating-point reduction
order can change last-bit results.

## Resource limits

Both solvers validate dimension products before allocating and reject requests
whose conservatively estimated concurrent working set exceeds 512 MiB. Each
`base64+zlib+f64le` field is likewise limited to 512 MiB after decompression,
and its decoded byte count must exactly match its declared shape. These are
recoverable request errors in JSONL batch mode rather than allocation panics or
process-wide failures.

## Lifted request

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
    "opinion_quadrature_points": 7,
    "opinion_quadrature_rule": "unit_variance_quantile"
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
Supported opinion quadrature rules are `unit_variance_quantile` and
`gauss_hermite`. Both use centered nodes with unit weighted variance before
linear deposition; a one-point rule is deterministic. The first rule uses
equal-weight midpoint normal quantiles, while the second uses transformed
Gauss--Hermite nodes and nonuniform weights. Lifted and kinetic requests use
the same rules and implementation.
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
bin/smp-lifted run @request.json
bin/smp-lifted batch < requests.jsonl > responses.jsonl
```

A malformed item in batch mode yields an error response for that line and does
not abort subsequent items.

Progress is opt-in and always goes to stderr. `human` is suitable for a
terminal; `jsonl` is stable machine-readable telemetry. A positive heartbeat
interval additionally reports paths that have not yet terminated:

```sh
bin/smp-lifted run --progress human --progress-step-interval 1000 @request.json
bin/smp-kinetic batch --progress jsonl --progress-step-interval 1000 \
  < requests.jsonl > responses.jsonl 2> progress.jsonl
```

Events identify the request and batch line, point/interval stage, interval
scenario, completed paths, last terminal category, elapsed time, and periodic
within-path step. Enabling telemetry does not change random streams.

## Kinetic request

Kinetic requests select `dynamics.opinion_method` as either `measure` or
`fokker_planck`. `measure` applies the complete nonlocal finite-exposure
push-forward. `fokker_planck` derives drift and diffusion from the same
finite-exposure first two moments and advances a conservative no-flux finite
volume system. Initial categorical densities and every returned scalar series
use `base64+zlib+f64le`; JSON numerical arrays are not part of the kinetic API.
Only requested observables are evaluated, and full `rho`, `edge`, and wedge
trajectories never cross the process boundary.

Selected state inspection is a separate opt-in interface. The required
`snapshots` block contains binary-encoded `record_steps` plus independent
`rho`, `edge`, `velocity`, and `rewiring_flux` switches. Returned snapshot
fields use the same compressed binary encoding and include only those steps.
When every switch is false, `record_steps` has shape `0`; the solver creates no
snapshot collector and performs no trajectory copies. This keeps scalar scans
on the original allocation path while supporting reproducible field figures.

### Frozen drift-landscape semantics

The snapshot `velocity` is the actual finite-exposure first-moment drift. For
HK it includes the no-update event and has the coefficient-bearing convention

```text
v(x,t) = influence * E[1_{C>0} S/C].
```

It is deliberately not the historical ratio-of-expectations closure
`influence * E[S] / E[C]`, which conditions away the probability of seeing no
concordant item. Downstream landscape analysis removes the update coefficient,
defines `F = velocity / influence`, reconstructs `V` from `F = -dV/dx`, and
defines a drift barrier between an unstable zero `x_s` and an adjacent stable
zero `x_w` by

```text
Delta V_drift = V(x_s) - V(x_w) = integral_[x_s,x_w] F(x) dx.
```

This definition is frozen for comparisons between `measure` and
`fokker_planck`. It is a coefficient-free effective drift diagnostic, not a
general escape or quasipotential: the measure operator is nonlocal, and the
Fokker--Planck operator also contains spatially varying endogenous diffusion.
The solver therefore returns the requested drift field but does not report a
scalar barrier. Analysis code must identify the relevant stable/unstable pair;
`max(V)-min(V)` is equivalent only in special symmetric two-well cases. Cells
with negligible `rho` describe an explicitly counterfactual drift extension
and should be reported as low-support when a barrier passes through them.

Kinetic `structure_random_l0` powers the pair-level overlap proxy. The distinct
`structure_random_l1` path transports four directional wedge channels, converts
their score mass to a finite-population common-neighbor mean, and uses the
capped Poisson moment `E[min(C, score_max)^steepness]`. Thus `steepness=1`
recovers the transported first score moment, while larger steepness values use
an explicit distributional closure instead of taking a power of that mean.

## Python orchestration

Install from the repository, including editable installation:

```sh
python -m pip install -e .
smp-mesoscopic-build --command lifted --backend purego
smp-mesoscopic-build --command kinetic --backend accelerate
```

Then reuse one Go process for a list of requests:

```python
from smp_meso_bindings import (
    build_binary,
    print_progress,
    run_kinetic_batch,
    run_lifted_batch_parallel,
)

kinetic_binary = build_binary(command_name="kinetic", backend="accelerate")
responses = run_kinetic_batch(
    kinetic_binary,
    requests,
    progress=print_progress,
    progress_step_interval=1000,
)

# Four long-lived Go batch processes with dynamic request scheduling;
# response order remains input order.
# Set each request's explicit workers value with the total CPU budget in mind.
lifted_binary = build_binary(command_name="lifted", backend="accelerate")
responses = run_lifted_batch_parallel(
    lifted_binary,
    requests,
    4,
    progress=print_progress,
    progress_step_interval=1000,
)
```

The corresponding runner's `check=False` returns both successful and error responses;
the default raises `BatchExecutionError` after collecting the complete batch.
Each `*_batch` function is the clean single-process execution path. The parallel
orchestrator uses Python threads only to supervise independent Go processes;
the numerical work remains in Go. A free process pulls the next parameter point
from a shared queue, avoiding the long tail of a static partition when hitting
times differ strongly. Budget approximately
`processes × request.workers` runnable paths to avoid oversubscription.
