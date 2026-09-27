# Release validation — 2026-09-27

Validation was performed locally on Windows with Python 3.12.14. This document
records software checks, not new paper experiments.

| Check | Result |
|---|---|
| Full offline suite, including optional RL | 253 passed |
| Standalone installed wheel with core dependencies | 247 passed; optional RL module skipped |
| RL runtime | PyTorch 2.14.0+cpu; existing local runtime used without modifying it |
| Core installation | Fresh virtual environment; dependency versions in requirements-core.lock |
| Network isolation | Socket connections blocked by an autouse test fixture |
| Prompt identity | Public bundle identity fixed; instruction body and assembly template preserved |
| Public naming | Neutral configuration, component and scenario IDs; 120 scene contents and 360 observations checked unchanged |
| Public dataset | 120 unique domain-valid scenarios; 40 per difficulty; byte hashes checked |
| Published LLM aggregates | Recomputed from exported observations; mean and sample SD agree |
| Original mapped sources/assets | 66 original SHA-256 values rechecked unchanged |
| Provider security | Explicit network gate, HTTPS validation, key redaction, redirect rejection, single-attempt failure halt and escaped-key echo checks |
| Git secret scanner | Worktree, index and reachable-history fixtures; staged-content and deleted-secret cases verified |
| Source/wheel packages | Built in isolated build environments; every archive member scanned against token rules and known private values |

The optional RL tests cover complete episodes, masks, reward accounting, one
small PPO update, deterministic inference, safe checkpoint loading, step-budget
handling and the explicit single-source selection rule. They use tiny software
test budgets and do not update the published results.

No real LLM/API request, formal GA search, formal RL campaign, scenario generation
campaign or manuscript recompilation was run. Historical code, logs, datasets,
PDFs and ZIPs outside this release remain unchanged.

Windows/Linux GitHub Actions jobs are configured. Hosted CI has not run because
this repository has not been published. Linux execution is therefore not claimed
as locally verified. No third-party vulnerability audit or proof that every
possible secret encoding is detectable is claimed.
