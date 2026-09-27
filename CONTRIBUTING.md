# Contributing

Use Python 3.12 and install `.[dev]` in a virtual environment. Install `.[rl]`
when modifying the RL baseline. Run `python -m pytest`,
`python tools/verify_release.py` and `python tools/check_secrets.py --staged --history`
before a pull request. Tests must never need an API key or a paid model call.

Enable `.githooks` locally. Keep patches focused and explain changes to numerical
behavior, candidate selection, budgets, training/validation separation or prompt
content. Update provenance and tests when scientific behavior changes. Do not
replace published paper results with new runs; add a separately identified result.

Use explicit paths, exclusive output creation and importable functions. Keep
optional dependencies lazy: installing the base package must not require PyTorch.
Never add a private endpoint, raw request record, key, binary checkpoint or local
machine path. Report vulnerabilities as described in SECURITY.md.
