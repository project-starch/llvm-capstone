# CPython 3.13.7 on CheriBSD purecap

`build.sh` cross-builds the complete interpreter with ordinary pymalloc
(`CPY_CHERI_MODE=spatial`, the default) or the existing PoisonCap pymalloc
component (`CPY_CHERI_MODE=poisoncap`),
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
`pyhome/lib/python313.zip` and `manifest.json` in that root. The PoisonCap
build selects interpreter patch `0014` in place of ordinary-pymalloc patch
`0009`, links the component's lifetime and external-metadata backends, and
reserves a 64 MiB payload region plus 16 MiB for allocator metadata. The same
binary selects its spatial adapter control (`PYM_POISONCAP_MODE=0`) or explicit
nested revocation (`PYM_POISONCAP_MODE=1`). It reports the mode and policy
operation counts and uses the shared application phase observer. The kernel's
ordinary libc revocation default is disabled for both modes; the explicit
pymalloc path still runs in mode 1.

For a PoisonCap mode-0 guest smoke test, stage the binary and zip as `/tmp/python-study` and
`/tmp/pyhome/lib/python313.zip`, then run:

```sh
env -i PATH=/sbin:/bin:/usr/sbin:/usr/bin HOME=/root LC_ALL=C \
  PYTHONHOME=/tmp/pyhome _RUNTIME_REVOCATION_DISABLE=1 \
  PYM_POISONCAP_MODE=0 /tmp/python-study -S -c 'print(6*7)'
```

The existing JSON/GC `objects.py 8 3 0` workload produces `EXP-OK cpython
552` in the ordinary spatial build and in three fresh-guest PoisonCap-adapter
mode-0 control processes. The protected mode-1 build links and reaches Python
startup but triggers `panic: share->excl` in the published CheriBSD kernel
during its first explicit nested revocation; it has no completed application
oracle. The outer jemalloc phase ledger does not include the separate pymalloc
regions. No four-arm CPython memory result is established yet.
