# Adaptive Feedback-LLM for Sidelink Scheduling

Public research implementation of Adaptive Feedback-LLM, accompanying the paper. The method combines
LLM proposals with deterministic evaluation and structured feedback to select flows,
routes and per-link radio resources. This repository includes the evaluator,
three-candidate feedback controller, prompt assets, GA reference, full/sparse masked
PPO implementations, 120 synthetic test scenarios, and the published result tables.

[Method and provenance](docs/METHODS.md) ·
[Security](SECURITY.md) · [Development](CONTRIBUTING.md)

## Install

Use Python **3.12**. Commands below work from the repository root on Windows,
Linux and macOS; activate your virtual environment using your shell's convention.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Optional dependencies: `python -m pip install -e ".[rl]"` for PyTorch baselines.
The wheel contains the
Python package and prompt assets; clone/download this source repository for
benchmark data, examples and development tools.

## Offline quick start

```sh
sidelink evaluate examples/scenario.json examples/schedule.json
sidelink demo examples/scenario.json --output outputs/demo
sidelink llm examples/scenario.json --mock-response examples/schedule.json --output outputs/mock
sidelink llm examples/scenario.json --mock-response examples/schedule.json --one-round --output outputs/mock-first-round
sidelink check-data data/benchmark/manifest.json
python tools/summarize_results.py
```

The mock repeats the supplied schedule; it demonstrates the controller and does
not reproduce LLM quality. Each output directory must be new. Package import,
offline testing, evaluation and demos do not read credentials or send LLM requests.

## Explicit live LLM use

Set `LLM_API_KEY`, `LLM_BASE_URL` (an HTTPS OpenAI-compatible API base, typically
ending in `/v1`) and `LLM_MODEL=deepseek-v4-flash` in the current process using
your secret manager or a hidden-input prompt. No endpoint or credential is
provided and `.env` files are **not automatically loaded**.

```sh
sidelink llm examples/scenario.json --live --output outputs/live-run
```

This command can incur provider charges: at most 60 HTTP requests (20 rounds,
3 candidates per round), or 3 with `--one-round`, each with a maximum output
budget of 32,768 tokens and a 240-second timeout. Reasoning is enabled with low
effort. There is one HTTP attempt per request. Any transport/content failure
halts subsequent dispatch. An admission journal is written before each request;
unknown outcomes are never automatically retried or resumed. Prices, availability
and the model behind an alias are determined by your provider. New calls are
new experiments, not the archived paper observations.

Request headers, API keys and HTTP error bodies are not saved. Successful feedback
runs contain scenario prompts and model responses in ignored output directories;
review these before sharing. See [SECURITY.md](SECURITY.md) for safe setup.

## Baselines

```sh
# Explicitly expensive: 5,760,000 population evaluations for this one input.
sidelink ga examples/scenario.json --full-budget --output outputs/ga

# Requires the rl extra and separately supplied disjoint datasets.
sidelink rl-train train/manifest.json validation/manifest.json --variant full --seed 42 --steps 200000 --output outputs/full-rl
sidelink rl-evaluate examples/scenario.json outputs/full-rl/selected.pt --output outputs/rl-evaluation
```

Use `--variant sparse` for Sparse RL. For the single-scenario full-RL design, pass
the same one-scene training manifest as both positional arguments and add
`--source-only`; checkpoint selection then uses only that source instance.
The default design requires disjoint training and validation sets. The benchmark
manifest has `split: test` and is rejected by both training modes.
[DATA.md](docs/DATA.md) defines the manifest format.

Pretrained checkpoints, the historical 6,000/600 training/validation pools, raw
provider responses and billing logs are not distributed. Consequently this
release supports running and extending the methods and inspecting published
statistics, but does not claim a one-command regeneration of the archived paper.

## Published paper results

| Method | Mean quality relative to GA | Local work (rounded) |
|---|---:|---:|
| GA | 100% | 95,000 |
| Adaptive Feedback-LLM | 99.78 ± 0.04% | 1.00 |
| One-round LLM | 92.98 ± 0.02% | 0.68 |
| Full RL | 94.43 ± 0.60% | 27 |
| Sparse RL | 88.54 ± 0.38% | 1.04 |

Quality is averaged over paired scenario ratios, then across 3 LLM repeats or
5 RL seeds; ± denotes sample SD. All LLM repeats share the same 120 test scenarios.
GA is a finite-budget empirical reference. Local work is a structural measure,
not elapsed time, provider compute or total computational cost. The fixed
normalization denominator is 1,876,511.5097109186.

## Repository guide

For a first read, follow `examples/` and the offline commands above. To understand
the algorithm, start with `workflows.py`, then `solvers/llm/runner.py`, prompt
assembly and `evaluator/core.py`. For implementation changes, first read the
domain models and the tests for the affected component.

```text
release/
  src/adaptive_feedback_llm/    # installable algorithm package and CLI
  configs/paper.yaml           # sole paper configuration profile
  data/benchmark/              # public synthetic test inputs and manifest
  data/published/              # existing paper numerical observations
  examples/                    # small scenario and known schedule
  tests/                       # unit, integration, API, security and RL tests
  tools/                       # result display, secret scans and manifests
  docs/                        # methods, data, publishing and provenance
  .githooks/                   # local pre-commit check
  .github/workflows/           # hosted CI configuration
```

### Package entry points and configuration

The following paths are relative to `src/adaptive_feedback_llm/`.

| File | Responsibility |
|---|---|
| `__init__.py` | Package identity and version; importing it does not start a run. |
| `__main__.py` | Enables `python -m adaptive_feedback_llm`, equivalent to `sidelink`. |
| `cli.py` | Parses commands, selects a method, loads inputs and publishes outputs. |
| `io.py` | Strict scenario/schedule loading, dataset hashes, path containment and exclusive JSON writes. |
| `config.py` | Configuration models and YAML loading, including duplicate-key and unknown-field rejection. |
| `profiles.py` | Active paper LLM configuration, prompt version and prompt identity hash. |
| `workflows.py` | Combines providers, request admission records and full/first-round execution. |
| `exceptions.py` | Shared error types for configuration, inputs, solvers and providers. Retained error classes do not imply that every historical runner is distributed. |

[`configs/paper.yaml`](configs/paper.yaml) is the sole distributed configuration
profile. It records shared physics/numerics, the paper LLM protocol, the ordered
GA portfolio, RL training settings and study counts. `load_config()` validates it
strictly; tests check it against runtime defaults. The CLI uses those frozen
Python defaults, so editing YAML alone does not retune a run. Evaluation uses
the supplied scenario's own `physics` and `numerics` fields.

| Section | Paper settings |
|---|---|
| `llm` | Flash, Thinking(low), 3 candidates, at most 20 rounds, patience 3, 32,768 tokens, 240 s, one attempt. |
| `genetic_algorithm` | (256,150), (512,450), (768,150); 15 restarts per component; 5,760,000 population evaluations per scenario. |
| `rl` | Hidden 128, PPO 4 epochs, batch 32, learning rate 0.0003, rollout 512, validation every 5,000 steps; study budget 200,000 steps per seed and 5 seeds. |
| `study` | 120 fixed cases, 40 per difficulty, 3 LLM repeats; local-work denominator 1876511.5097109186. |

Only the paper profile is supported. Small fixtures and explicitly reduced
offline test budgets are software tests, not alternative release profiles.

The LLM profile selects the paper prompt and the `deepseek-v4-flash` interface alias,
with thinking enabled and low reasoning effort. It sets 3 candidates per round,
a 20-round cap, 3 complete post-feasibility stagnation rounds, 32,768 output
tokens, 240 seconds per request and one transport attempt.

### Shared scientific core

| Module | Responsibility |
|---|---|
| `models/domain.py` | Links, flows, candidate paths, SRU allocations, schedules and evaluator scenarios. |
| `models/results.py` | Feasibility, objective, per-flow diagnostics and per-link usage. |
| `models/llm.py` | Allowlisted LLM views, evaluated candidate history and structured feedback. |
| `models/serialization.py` | Strict JSON boundaries for scenarios, schedules and results. |
| `physics/core.py` | SINR conversion, rates, loss/success probabilities, path reliability, delay and utility. |
| `paths/catalog.py` | Deterministic simple-path catalog construction in a directed graph. |
| `guidance/static.py` | Nonbinding single-flow path references and hotspot/resource information. |
| `evaluator/core.py` | Shared resource/QoS feasibility checks, objective evaluation and incumbent retention. |

SRU is the discrete radio resource allocation unit. All methods ultimately use
the same evaluator. A standalone path reference cannot certify feasibility when
multiple flows compete for capacity. Subpackage `__init__.py` files mainly expose
importable interfaces; implementation resides in the named modules.

### Prompt assets and feedback

| File or group | Purpose |
|---|---|
| `prompts/__init__.py` | Asset selection, prompt rendering and bundle hashing. |
| `instructions.txt` | The sole task instruction asset. |
| `request_template.txt` | The sole prompt assembly template. |
| `schedule_schema.json` | Expected schedule JSON schema. |
| `minimal_example.json` | An output-format example, not a benchmark result. |
| `compact_context.py` | Compact, recoverable public-history representation used by paper. |
| `opportunities.py` | Deterministic reference facts derived from the public view. |
| `evidence_matrix.py` | Evidence organization based on evaluated candidates. |
| `global_advisor.py` | Search-direction/feedback organization for the paper prompt. |

Only the paper prompt and its active helpers are included. Public format versions
identify JSON/transport contracts. The workflow verifies the public bundle hash
before execution; changing protected assets requires a new documented identity.

### Solvers

| Module under `solvers/` | Responsibility |
|---|---|
| `common.py`, `results.py` | Shared schedule construction/resource reservation and solver result models. |
| `heuristics/core.py` | Priority-Aware, Max-SINR and Load-Aware heuristics; `demo` runs Priority-Aware. |
| `genetic_algorithm/core.py` | GA initialization, selection, crossover, mutation, restarts and search audits. |
| `ga_portfolio.py` | The paper three-component GA reference and cross-component selection. |
| `llm/runner.py` | Rounds, three candidates, parsing/evaluation, history, incumbents and stopping. |
| `llm/projection.py` | Allowlisted model-visible scenario data and QoS-gap feedback. |
| `llm/provider.py` | Provider interface, deterministic mock, explicit HTTP transport and credential handling. |
| `llm/telemetry.py` | Limited request/timing/token metadata, without credentials or HTTP error bodies. |

The GA portfolio uses population/generation pairs `(256,150)`, `(512,450)` and
`(768,150)`, with 15 restarts per component, totaling 5,760,000 population
evaluations per input. The components are named `balanced`, `deep-search` and
`wide-search`, in stable tie-breaking order.

### Reinforcement learning

| File under `rl/` | Responsibility |
|---|---|
| `decoder.py` | Finite path/allocation options and their conversion into schedules. |
| `full_environment.py` | Full action environment, remaining capacity, masks, observations and rewards. |
| `full_policy.py` | Full message-passing actor-critic. |
| `sparse_environment.py` | Reduced service-action construction and sparse environment state. |
| `sparse_policy.py` | Sparse message-passing actor-critic. |
| `ppo.py` | Complete-episode rollouts, stored masks, returns, clipped losses and PPO updates. |
| `training.py` | Variant selection, dataset checks, step budgets, checkpoint selection and inference. |

An episode processes the scenario's flows. Inference uses deterministic argmax
over legal actions and independently checks the terminal schedule. By default,
training selects checkpoints using disjoint validation data; strict improvements
replace the incumbent, and exact ties keep the earlier checkpoint. The explicit
`--source-only` exception uses the same single training source for selection with
the full policy. The included `test` manifest is rejected in both modes.

### Data and examples

| Path | Contents and intended use |
|---|---|
| `data/benchmark/manifest.json` | Membership, difficulty, relative paths, public file hashes and source fingerprints. |
| `data/benchmark/scenarios/` | 120 synthetic test inputs: 40 each easy/medium/hard, with 24/28/32 nodes and 14/16/18 flows. |
| `data/published/statistics.json` | Method/tier aggregates, repetition counts, sample SD, iteration summaries and fixed work denominator. |
| `data/published/llm_metrics.json` | 360 existing per-scene repeat observations, including objectives and iteration histories, without private checkpoint paths. |
| `data/published/iteration120.csv` | Columns `round`, `repeat1`–`repeat3`, `mean`, `sample_sd`, `scenarios_per_repeat`. |
| `examples/scenario.json` | Handcrafted 6-node, 2-flow input with 5 candidate paths per flow. |
| `examples/schedule.json` | A known schedule for the small input, usable for evaluation or a mock response. |

Scenario IDs use `easy-001` through `hard-040`, with 40 numbered cases per
difficulty. The manifest marks all 120 supplied cases as test data.

A scenario describes the problem through `node_ids`, `links`, `flows`,
`path_catalogs`, `physics` and `numerics`. A schedule describes decisions through
`served`, `path_id` and per-link `sru_allocations`. The 360 LLM observations reuse
120 networks; they are not 360 independent test networks.

### Execution flow and outputs

```text
scenario JSON → io/models → workflows → round controller
  → projected view + path references + prompt assets
  → request admission journal → provider (3 candidates per round)
  → JSON parsing → shared evaluator/physics
  → incumbent + evaluated history → next-round feedback → final result
```

The three candidates in a round use that round's rendered prompt. Their facts
enter the following round's history. The first-round baseline stops after three
candidates. GA searches schedules and RL constructs them via masked actions;
both use the same underlying feasibility/objective definitions.

| Command | Output |
|---|---|
| `evaluate` | Feasibility/objective JSON on stdout. |
| `check-data` | Validated count and split on stdout. |
| `demo`, `ga`, `rl-evaluate` | `result.json` in the requested new directory. |
| `llm`, including `--one-round` | `run.json`, `requests/` and `result.json`. |
| `rl-train` | `selected.pt` and `summary.json`. |

LLM `run.json` identifies the mode, prompt hash and request cap. A request's
`*.intent.json` is written before dispatch; `*.terminal.json` records its local
outcome. Missing terminal state does not establish that the provider did no work.
The full result also includes interactions, evaluated history, rounds and the
selected schedule. These records can contain prompts and model responses.

`transport_halted` means further HTTP dispatch was stopped. The controller can
still record logical failures after that point, so logical candidate counts are
not necessarily actual HTTP request counts. Check `evaluation` for feasibility;
exit code 0 alone does not mean a feasible schedule was found.

RL's public checkpoint contains model weights/metadata, and its summary records
steps, selection scores and input hashes. Inference uses `weights_only=True`
and accepts the public checkpoint format, not old private campaign checkpoints.
The CLI does not automatically regenerate training pools or run a 120-scene batch.

### Tests, tools and documentation

| Path | Purpose |
|---|---|
| `tests/conftest.py`, `tests/fixtures/` | Shared small fixtures and a default network-socket block. |
| `tests/unit/` | Model, physics, evaluator, path, heuristic and prompt-component regressions. |
| `tests/integration/` | GA and LLM component interactions; provider calls are simulated. |
| `tests/test_public_api.py` | CLI, prompt identity, datasets, published aggregation, portfolio budget and path checks. |
| `tests/test_security.py` | Provider secrecy, network admission, request paths, index scans and historical leaks. |
| `tests/test_rl.py` | Episodes, policies, small PPO updates, checkpoint round trips, step budgets and source-only selection. |
| `tools/summarize_results.py` | Displays existing aggregate results without rerunning methods. |
| `tools/check_secrets.py` | Scans public working files, optionally Git index blobs and reachable history. |
| `tools/update_manifest.py` | Checks the public file policy and rewrites the release hash manifest. |
| `tools/verify_release.py` | Checks the files listed in the manifest, without experimental replay. |
| `docs/METHODS.md` | Manuscript identity, algorithm/statistical definitions and refactor boundaries. |
| `docs/DATA.md` | Scenario and dataset interfaces and split rules. |
| `docs/PUBLISHING.md` | Source review, builds and publication steps. |
| `docs/VALIDATION.md` | The recorded local validation environment, results and limits. |
| `docs/source_map.json` | Original content hashes, public destination paths and transformations for imported assets. |

Local validation records 253 passing tests with RL, or 247 passing tests and an
optional RL module skip in a core-only wheel environment. All 120 scenario
parameter sets and 360 published observations were checked across public renaming.
Working-file secret scans respect the public file set and Git ignore rules.
Manifest verification checks listed files, not every possible new unlisted file.

### Root files and automation

| File or path | Purpose |
|---|---|
| `README.md` | English installation, usage, architecture and repository guide. |
| `pyproject.toml` | Version, Python constraint, dependencies, optional extras, CLI and test settings. |
| `requirements-core.lock` | Tested core dependency versions; not a complete hash-locked dev/RL environment. |
| `MANIFEST.in` | Explicit source-distribution inclusion rules. |
| `release_manifest.json` | Current public file hashes after CRLF-to-LF normalization; excludes itself. |
| `.env.example` | Empty credential/endpoint fields for documentation; never automatically loaded. |
| `.gitignore` | Excludes credentials, environments, runtime outputs, checkpoints and generated archives. |
| `.gitattributes` | Normalizes text line endings across platforms. |
| `.githooks/pre-commit` | Local working-file/index scan; enable it with `core.hooksPath` after cloning. |
| `.github/workflows/ci.yml` | Windows/Linux core checks and builds, plus Linux optional RL checks, on push/PR. |
| `LICENSE` | MIT license text. |
| `CITATION.cff` | Software citation metadata; no manuscript DOI is asserted. |
| `CONTRIBUTING.md` | Development, validation and change-reporting conventions. |
| `SECURITY.md` | Credential handling, scan boundaries, reporting and incident guidance. |
| `CHANGELOG.md` | Source refactor and interface changes. |

### Local generated files

These can appear in a working directory but are not tracked public source.

| Path | Meaning |
|---|---|
| `.git/` | Local Git objects, commits, branches, tags and configuration; preserve it to retain local history. |
| `.venv/` | Rebuildable development/core-test environment. |
| `.venv-wheel/` | Separate environment used to validate wheel installation. |
| `.qa/` | Local build/install/commit logs, archive audit JSON and extracted validation copies. |
| `.pytest_cache/` | Test-run cache, not research observations. |
| `src/adaptive_feedback_sidelink.egg-info/` | Generated installation/build metadata; maintain `pyproject.toml` instead. |
| `__pycache__/` | Generated Python bytecode. |
| `dist/` | Built `.whl` and `.tar.gz` snapshots. |
| `.qa-build.log`, other `*.log` | Local tool output. |
| `build/`, `outputs/`, `runs/`, `logs/` | May be generated by future builds/runs; not necessarily present yet. |

Develop in the main source tree, not `.qa/extracted/` or `.qa/final-extracted/`.
Editing a README or source file does not update an existing archive or wheel
installation. Rebuild when a new distribution is needed. The wheel contains
the Python package/prompt assets; the source distribution also contains public
data, examples, tests and docs. Publish tracked source rather than an arbitrary
ZIP of the entire working folder.

The original training pools, trained weights, provider records, full scenario
generation campaign and manuscript build project are outside this release.
Scientific kernels retain their archived semantics; source provenance is recorded
in `docs/source_map.json`. See [CHANGELOG.md](CHANGELOG.md) for the refactor boundary.

## Before publishing changes

```sh
git config core.hooksPath .githooks
python tools/check_secrets.py --staged --history
python tools/verify_release.py
```

The hook scans staged **blob contents**, including force-added files. CI also
scans reachable history. No scanner guarantees that every possible secret is
detectable. Keep credentials and runtime records outside version control.

## License and citation

MIT; see [LICENSE](LICENSE). `CITATION.cff` describes this software release.
The manuscript is not distributed here and no publication DOI is asserted.
