"""Fail closed on private files and common secrets, without echoing matched values.

Scan working files, the staged blob contents, and all reachable Git history.
Only Python's standard library is required. This is a guard, not a guarantee.
"""

import argparse
import base64
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".qa", ".venv", ".venv-qa", "__pycache__", ".pytest_cache", "build", "dist",
                 "outputs", "runs", "logs", ".ruff_cache"}
PATTERNS = {
    "provider-token": re.compile(rb"\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential-url": re.compile(rb"https?://[^\s/'\"]+:[^\s/'\"]+@"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{16,}\.eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"),
    "personal-workspace": re.compile(rb"(?i)(?:[A-Z]:[/\\](?:Users|Documents and Settings)[/\\][^/\\\r\n]+|/(?:home|Users)/[^/\s]+)"),
    "assigned-secret": re.compile(rb'''(?im)["']?(?:[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|CLIENT_SECRET))["']?\s*[:=]\s*["']([^\r\n"']{12,})["']'''),
    "unquoted-secret": re.compile(rb"(?im)^\s*(?:[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD|CLIENT_SECRET))\s*[:=]\s*([A-Za-z0-9/+_=-]{20,})\s*$"),
}
SAFE_LITERALS = {b"PASTE_API_KEY_HERE", b"LLM_API_KEY", b"credential-file value for {key} is empty"}


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def forbidden_name(name):
    parts = Path(name).parts
    lower = name.lower()
    leaf = Path(name).name.lower()
    if leaf.startswith(".env") and leaf != ".env.example":
        return True
    if any(p in {"runs", "logs", ".venv", ".ssh", "__pycache__"} for p in parts):
        return True
    if leaf in {".netrc", ".npmrc", ".pypirc"} or leaf.startswith(("id_rsa", "id_ed25519")):
        return True
    return any(fnmatch.fnmatch(lower, pattern) for pattern in
               ("*.pem", "*.key", "*.p12", "*.pfx", "*.pt", "*.pth", "*.ckpt", "*.sqlite*",
                "*.zip", "*.tar", "*.gz", "*.7z", "*.log", "*.pyc", "*credentials*"))


def inspect(name, raw, known=()):
    findings = []
    if forbidden_name(name):
        findings.append({"file": name, "line": 0, "rule": "private-or-binary-file"})
    # This source release deliberately contains only UTF-8 text. Reject opaque payloads.
    try:
        raw.decode("utf-8-sig")
        if b"\x00" in raw:
            raise UnicodeError()
    except UnicodeError:
        findings.append({"file": name, "line": 0, "rule": "opaque-binary"})
        return findings
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(raw):
            if rule == "assigned-secret" and match.group(1) in SAFE_LITERALS:
                continue
            findings.append({"file": name, "line": raw[:match.start()].count(b"\n") + 1, "rule": rule})
    for secret in known:
        if secret and secret in raw:
            findings.append({"file": name, "line": 0, "rule": "known-private-value"})
    return findings


def collect(mode, root=ROOT):
    root = Path(root).resolve()
    try:
        top = Path(git("rev-parse", "--show-toplevel", root=root).decode().strip()).resolve()
        owns_repo = top == root
    except (subprocess.CalledProcessError, FileNotFoundError):
        owns_repo = False
    if mode in {"staged", "history"} and not owns_repo:
        raise ValueError("scan root must be the repository root")
    if mode == "staged":
        for row in git("ls-files", "--stage", "-z", root=root).split(b"\0"):
            if row:
                info, path = row.split(b"\t", 1)
                filemode, oid, stage = info.split()
                name = path.decode("utf-8")
                if filemode == b"120000" or stage != b"0":
                    yield name + ".key", b"unsupported symlink or unmerged entry"
                else:
                    yield name, git("cat-file", "blob", oid.decode(), root=root)
    elif mode == "history":
        for line in git("rev-list", "--objects", "--all", root=root).splitlines():
            oid, _, path = line.partition(b" ")
            kind = git("cat-file", "-t", oid.decode(), root=root).strip()
            if kind in {b"blob", b"commit", b"tag"}:
                yield "history/" + (path.decode("utf-8") or oid.decode()), git("cat-file", "-p", oid.decode(), root=root)
    else:
        try:
            if not owns_repo:
                raise FileNotFoundError()
            names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", root=root).split(b"\0")
            paths = [root / n.decode("utf-8") for n in names if n]
        except (subprocess.CalledProcessError, FileNotFoundError):
            paths = [p for p in root.rglob("*") if p.is_file()
                     and not EXCLUDED_DIRS.intersection(p.relative_to(root).parts)
                     and not any(s.startswith(".venv") or s.endswith(".egg-info")
                                 for s in p.relative_to(root).parts)]
        for path in sorted(set(paths)):
            if path.is_symlink():
                yield path.relative_to(root).as_posix() + ".key", b"unsupported symlink"
            elif path.is_file():
                yield path.relative_to(root).as_posix(), path.read_bytes()


def known_values(path):
    values = []
    if path:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "BASE_URL")):
                value = value.strip().strip("\"'")
                if len(value) >= 8:
                    encoded = value.encode()
                    values.extend([encoded, base64.b64encode(encoded), quote(value, safe="").encode()])
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staged", action="store_true")
    p.add_argument("--history", action="store_true")
    p.add_argument("--known-env-file", type=Path, help="Read private values in memory; never print or copy them")
    args = p.parse_args(argv)
    modes = ["worktree"] + (["staged"] if args.staged else []) + (["history"] if args.history else [])
    known = known_values(args.known_env_file)
    count, findings = 0, []
    try:
        for mode in modes:
            for name, raw in collect(mode):
                count += 1
                findings.extend(dict(row, scope=mode) for row in inspect(name, raw, known))
    except Exception:
        print(json.dumps({"status": "error", "reason": "scan incomplete; publication blocked"}))
        return 2
    print(json.dumps({"status": "failed" if findings else "passed", "objects_scanned": count,
                      "findings": findings}, ensure_ascii=True, indent=2))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
