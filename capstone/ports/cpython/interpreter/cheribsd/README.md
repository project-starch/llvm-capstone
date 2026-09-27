# CPython 3.13.7 on CheriBSD purecap

`build.sh` cross-builds the complete interpreter with its ordinary pymalloc,
from the pinned upstream archive. Sources, the native build Python, cross-build,
stdlib zip and manifest stay under `$CAPSTONE_TMP_ROOT`. The recipe applies the
interpreter's existing pointer-layout patches except the Capstone-only
thread-local workaround (`0006`), then the CheriBSD patch in this directory.
That patch admits FreeBSD cross configuration and distinguishes the 64-bit
address from CHERI's 16-byte `uintptr_t` when CPython sizes its radix tree and
hashes pointer identities. The actual capability storage remains 16 bytes.

Set `CHERI_SDK` and `CHERI_SYSROOT`, source the project test environment, and
run `build.sh`. An existing `CPY_BUILD_PYTHON` may point to a native **3.13.7**
interpreter; otherwise the recipe builds one from the same source. Study builds
require a fresh `CPY_CHERI_ROOT`. The result is `python`,
`pyhome/lib/python313.zip` and `manifest.json` in that root. For a guest smoke
test, stage the binary and zip as `/tmp/python-study` and
`/tmp/pyhome/lib/python313.zip`, then run:

```sh
env -i PATH=/sbin:/bin:/usr/sbin:/usr/bin HOME=/root LC_ALL=C \
  PYTHONHOME=/tmp/pyhome _RUNTIME_REVOCATION_DISABLE=1 \
  /tmp/python-study -S -c 'print(6*7)'
```

The first complete application check used the existing JSON/GC `objects.py`
workload at `8 3 0` and produced `EXP-OK cpython 552` with the expected phase
sequence. This is a **spatial pymalloc** build and a functional qualification,
not a four-arm memory result. A PoisonCap pymalloc integration into the entire
interpreter, matched Capstone/CheriBSD build settings, memory ledgers and
repeated benchmark runs remain necessary before a CPython paper plot.
