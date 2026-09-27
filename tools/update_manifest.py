"""Create the public source manifest from the same file set the secret guard scans."""

import hashlib
import json

from check_secrets import ROOT, collect, inspect


def main():
    files = {}
    for name, raw in collect("worktree"):
        if name == "release_manifest.json" or ".egg-info/" in name:
            continue
        if inspect(name, raw):
            raise SystemExit("Source manifest blocked by secret/file policy; run check_secrets.py")
        files[name] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    value = {"release": "0.1.0", "profile": "paper", "license": "MIT",
             "hash_policy": "SHA-256 over UTF-8 file bytes after CRLF-to-LF normalization",
             "files": dict(sorted(files.items()))}
    (ROOT / "release_manifest.json").write_text(json.dumps(value, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(f"Recorded {len(files)} public source files.")


if __name__ == "__main__":
    main()
