# Security policy

## Credentials

Supply keys through a process environment or secret manager. Do not place a real
key in source code, notebooks, shell command arguments, screenshots, issue reports
or `.env.example`. The CLI never loads a credential file automatically. For a
hidden Bash prompt: `read -rs LLM_API_KEY; export LLM_API_KEY`; clear it afterwards
with `unset LLM_API_KEY`. A PowerShell example appears in the Chinese README.

Transport settings redact keys in representations and exclude them from model
serialization. Validation errors exposed by the CLI are summarized by exception
type. HTTPS URLs with userinfo, query strings or fragments are rejected. Ambient
proxy variables and redirects are disabled. Custom transports can be injected
for tests without enabling a real socket. Raw HTTP failures are not logged.

The paper CLI uses one attempt per request, persists admission before dispatch,
stops dispatch after a provider failure, and has no automatic recovery. A timeout
does not prove that remote computation or billing stopped. Do not blindly rerun
an interrupted run. All output directories must be new.

## Repository controls

`.gitignore` excludes credentials, logs, request records in output directories,
checkpoints, databases and archives. `tools/check_secrets.py` scans UTF-8 files,
staged blobs and reachable Git history without printing matched values. It rejects
opaque binaries/archives in this source-only release, private filenames, common
provider token formats, private keys and credential-bearing URLs. An optional
`--known-env-file` argument checks known local values and simple encoded variants
in memory; that file is never copied or included in the report.

Enable the local hook using `git config core.hooksPath .githooks`. Hooks are not
installed by a clone and can be bypassed, so CI repeats the checks. Keep the test
suite offline: its socket fixture blocks outbound connections even when a shell
contains valid provider credentials. Unit tests use generated dummy credentials.

## Reporting and incidents

Do not disclose a credential in a public issue. If the hosting platform provides
private security reports, use that channel; otherwise contact the repository
owner privately through their listed contact. A contact address is not fabricated
in this release. If a key is exposed, revoke/rotate it at the provider before
cleaning Git history; adding the path to `.gitignore` does not remove old commits.

The scanner is a layered precaution, not a proof that all possible encodings or
secret types have been found. Review added files and dependency changes. The
included scientific inputs are synthetic; do not assume new user data is public.
