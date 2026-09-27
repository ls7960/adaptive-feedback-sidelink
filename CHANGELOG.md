# Changelog

## 0.1.0 — 2026-09-27

- Retained a single paper configuration with public component and scenario names.
- Extracted the scientific core and paper prompt rules into an installable
  source-layout package, preserving original scientific source outside this repo.
- Relocated full/sparse RL decoder, environments, policies and PPO into `rl/`;
  replaced dated campaign launchers and global monkey-patching with explicit
  variant selection and a portable training API.
- Added an explicit paper GA portfolio wrapper and full/one-round LLM workflows.
- Added offline CLI examples, a public 120-scenario test manifest, sanitized
  published statistics, provenance, MIT license, contributor and security docs.
- Hardened the provider boundary, added offline tests, staged/history secret
  scans, local hooks and Windows/Linux CI configuration.

This initial public release accompanies the paper. No real model call, formal
GA search or formal RL campaign was performed during preparation. Small software
tests are separate from the published experiments. Original private checkpoint
formats and campaign recovery paths are outside the public API.
