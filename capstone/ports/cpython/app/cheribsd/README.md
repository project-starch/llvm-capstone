# CPython 3.13.7 on CheriBSD purecap

`build.sh` cross-builds the complete interpreter with ordinary pymalloc from the pinned upstream
archive, for the platform's own libc revocation. Sources, the native build Python, cross-build,
stdlib zip and manifest stay under `$CAPSTONE_TMP_ROOT`. The recipe applies the interpreter's
pointer-layout patches except the Capstone-only thread-local workaround (`0006`) and the
Capstone Sublet protection (`0014`), then the CheriBSD patch in this directory. That patch admits
FreeBSD cross configuration and distinguishes the 64-bit address from CHERI's 16-byte `uintptr_t`
when CPython sizes its radix tree and hashes pointer identities. The actual capability storage
remains 16 bytes.

Set `CHERI_SDK` and `CHERI_SYSROOT`, source the project test environment, and run `build.sh`. An
existing `CPY_BUILD_PYTHON` may point to a native **3.13.7** interpreter; otherwise the recipe
builds one from the same source. Study builds require a fresh `CPY_CHERI_ROOT`. The result is
`python`, `pyhome/lib/python313.zip` and `manifest.json` in that root.

A guest smoke test, with the binary and zip staged as `/tmp/python-study` and
`/tmp/pyhome/lib/python313.zip`:

```sh
env -i PATH=/sbin:/bin:/usr/sbin:/usr/bin HOME=/root LC_ALL=C \
  PYTHONHOME=/tmp/pyhome /tmp/python-study -S -c 'print(6*7)'
```

The PoisonCap build mode (the component's PoisonCap lifetime backend and its deferred-free
patch) was removed on 2026-10-10. The results of the
[four-arm campaign](../../../../experiments/study/results/cpython-reuse-four-arm-20260928/README.md)
that used it remain as recorded.
