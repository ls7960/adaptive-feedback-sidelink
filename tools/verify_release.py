"""Check the source release manifest; this does not replay experiments."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(name)
            continue
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            failures.append(name)
    print(json.dumps({"checked": len(manifest["files"]), "failed": failures}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
