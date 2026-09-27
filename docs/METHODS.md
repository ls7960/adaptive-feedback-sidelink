# Method, provenance and boundaries

## Manuscript identity

This code accompanies the bilingual manuscripts. The English
PDF SHA-256 is `3d808f2a5d923c5c64fbd611a699eb5e3323836900519d5d615bde6117ae8080`;
the Chinese PDF SHA-256 is
`e4abd4b7e1692ed255b2a0978923e03abe8f2cf94331408deb3eb2dac04cbc2c`.
The manuscripts and their editable archive are not included here.

The authoritative statistical input SHA-256 is
`d06b1ad153ff35782f7449a4cb4c3b0d572cfb689a3899cd8dbfb413c7523630`.
Public statistics retain its numerical fields, excluding private orchestration
paths. `source_map.json` records imported code/fixture assets by public destination and
original content hash, without private workspace paths. `release_manifest.json` fixes public file hashes.

## Scientific implementation

- `physics` and `evaluator`: per-link rates, path reliability, completion delay,
  resource feasibility and classified continuous objective. Every method uses
  the same evaluator.
- `prompts` and `solvers/llm`: the paper prompt, three candidates per round,
  evaluated history, best-feasible retention, a 20-round cap and three complete
  no-improvement rounds after positive feasibility. The public prompt bundle SHA-256 is
  `d35e6ad71900f775134f95707566448a85274defb5ae9f07fa096e919cacfad2`.
- The frozen implementation's positive/improvement comparisons retain their
  numerical tolerance. The paper uses strict mathematical inequalities; archived
  offline audits established equivalence for the reported histories. Equivalence
  is not assumed for every newly supplied floating-point input.
- The GA portfolio uses population/generation pairs (256,150), (512,450),
  (768,150), with 15 restarts each: 5,760,000 population evaluations per scenario,
  691,200,000 over the 120-scene study. Component names describe the balanced, deep-search and wide-search budgets. Highest objective wins across components with
  stable exact ties; within components the original comparator is retained.
- Full and Sparse RL preserve the finite decoder, masks, feature dimensions,
  message passing, complete-episode PPO and reward definitions. The formal study
  used hidden width 128, 4 PPO epochs, batch size 32, learning rate 0.0003,
  rollout target 512, validation interval 5000 and 200000 environment steps per
  seed. Five seeds were retained for each baseline.

## Refactoring boundary

Only `configs/paper.yaml` and the paper prompt are distributed. YAML is a strict
record of the paper parameters; tests compare it with runtime defaults. Provider
entry points use Flash, Thinking(low), 240-second timeouts and one attempt.
Formal GA components use 15 restarts each and only the three documented budgets.

Numerical physics, evaluation, feedback calculations, action decoding and PPO
operations are retained. Public class/resource names and scenario identifiers
have changed; the instruction text differs only in its format header. Prompt
bundle hashes therefore identify this public package, not an archived byte-for-byte
request. Renamed scenario IDs also participate in GA seed derivation: a new run
is not represented as an exact replay of the archived random stream or API output.
The numerical component master seeds are retained. Published observations are
linked consistently to the renamed IDs, with all parameter/result values checked
unchanged. The public GA wrapper executes the full fixed budget in process,
without private campaign restart/recovery orchestration.

`rl.training` is a portable new orchestration API. It selects checkpoints by
validation mean normalized objective with strict improvement and earliest ties,
but is not represented as a byte-identical replay of the historical five-seed
campaign. For the paper's single-source baseline, both training and checkpoint
selection used the registered source instance only, with no final-test access.
The public trainer requires disjoint training/validation manifests by default.
An explicit `--source-only` option accepts exactly one training source as both
inputs and uses the full architecture, matching the source-only selection design.

The historical 6,000/600 training/validation datasets, original trained weights,
provider transcripts, private feasibility anchors and campaign recovery state are
not bundled. The repository therefore does not promise regeneration of all reported
scores from an empty machine. It exposes the algorithms, test inputs and reported
statistics; future experiments must carry their own identities and budgets.

## Statistics

Three LLM repeats share the same 120 scenarios (40 each easy/medium/hard). For
each repeat/seed, average `100 * J_method / J_GA` over paired scenarios, then report
the mean and sample SD across repeats/seeds. GA is a finite-budget reference, not
a proof of global optimality. Do not clamp observations above GA to 100%.

Local work uses the fixed denominator 1876511.5097109186. Static path reference
work is counted once per instance. This accounting convention does not introduce
a new prompt-reference cache into the frozen implementation. LLM inference,
tokens, money, RL training and wall time are outside this structural metric.
