# Publishing this repository

The published repository and package documentation are English-only.

The prepared repository is local. No remote hosting account, repository URL or
publication is assumed. Publish only tracked source files, for example using
Git push after configuring your chosen remote; never upload the entire working
directory with its virtual environments and ignored runtime outputs.

Before the first push and after any change:

```sh
git status --short
python tools/check_secrets.py --staged --history
python -m pytest
python tools/verify_release.py
```

The release manifest hashes UTF-8 source text after CRLF-to-LF normalization,
matching `.gitattributes`. Its own file is excluded to avoid a circular digest.
When making a new release, regenerate the manifest using
`python tools/update_manifest.py`, review its changes and commit both source
and manifest. This operation checks the secret guard before writing.

`python -m build` produces a wheel and source distribution under ignored `dist/`.
The wheel contains code and prompt assets; the source distribution also includes
examples, public data, tests and documentation via an explicit inclusion list.
Do not distribute arbitrary ZIPs of the whole workspace. A local hook and CI
configuration are provided; hosted CI results exist only after a push.

The author/DOI fields for the unpublished manuscript are intentionally absent.
Update the software citation and public contact information when those details
are available. The copyright statement uses the project contributor identity.
