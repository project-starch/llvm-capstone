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
ordinary libc revocation is enabled in both modes for the published-policy
campaign. Earlier diagnostics explicitly disabled it and remain separate.
For a reuse study, `CPY_CHERI_GAP_OBSERVER=1` compiles an integer-only
observer into the inner pymalloc lifetime backend. Both process modes emit
one `PYM_REUSE_GAP` histogram with 32 logarithmic release-to-reissue bins.
The common CheriBSD runner validates that report when a point declares
`"reuse_gap": "PYM_REUSE_GAP"`; a nonzero observer error or inconsistent
counts invalidate the process. Its index counts successful handouts, and
the conditional histogram does not establish a fixed-horizon retirement
fraction on its own.

For a PoisonCap mode-0 guest smoke test, stage the binary and zip as `/tmp/python-study` and
`/tmp/pyhome/lib/python313.zip`, then run:

```sh
env -i PATH=/sbin:/bin:/usr/sbin:/usr/bin HOME=/root LC_ALL=C \
  PYTHONHOME=/tmp/pyhome _RUNTIME_REVOCATION_DISABLE=1 \
  PYM_POISONCAP_MODE=0 /tmp/python-study -S -c 'print(6*7)'
```

The [four-arm campaign](../../../../experiments/study/results/cpython-reuse-four-arm-20260928/README.md)
now passes the JSON/GC oracle and inner reuse checks in three processes per
arm. The protected mode uses deferred free-list publication with the published
SQLite thresholds transferred to pymalloc. It never makes a retired block
available before its sweep completes. A full queue holds 4,096 blocks; the
percentage trigger requires at least 16 MiB in live plus quarantined rounded
spans and at least one quarter quarantined. Realloc may move a block. Explicit
teardown drains precede interpreter-state destruction. The queue's static
metadata is charged in both arms.

This campaign requires the [VM-object poison-probe repair](../../../../experiments/study/patches/cheribsd-poison-object-probe.patch)
as well as the existing superpage/libc repairs. The original artifact's user
probe can enter `vm_fault()` recursively under the VM-map read lock, causing
`panic: share->excl`. No unresolved probe is silently ignored by the repair.
Unsupported VM objects and partial pages fail explicitly; swap-pressure stress
is still outstanding. The full successful campaign preserves its build and
runtime identities. Its outer jemalloc phase ledger excludes the separate
pymalloc regions; the reuse result is not a total-memory claim.
