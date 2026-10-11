# CPython complete-interpreter nested-allocator qualification

The CPython 3.13.7 interpreter passes **9/9 qualified processes** on the
complete `objects.py 8 3 0` JSON/GC workload: three Capstone spatial-pymalloc
processes, three Capstone per-block Sublet processes, and three CheriBSD
PoisonCap-adapter spatial processes. Every accepted process reports the same
nine phase markers and `EXP-OK cpython 552`; its complete stdout SHA-256 is
`badc8ab43190bf6ab70a21b566693f1c37076aaf482c224ef99ab306c39dd112`.
The six Capstone attempts share one persistent Linux VM boot. All three
CheriBSD spatial controls share one fresh guest. Source release, workload and
application optimization (`-O1`) are matched; the target ABI, libc, compiler
and OS necessarily differ. Each platform pair uses one binary with a runtime
spatial/protected mode switch.

The complete CheriBSD PoisonCap mode-1 interpreter **does not complete** the
workload. It reaches Python startup and then the published kernel panics with
`panic: share->excl` on its explicit nested revocation path. The panic occurs
both after a passing mode-0 control and when mode 1 runs first in a separate
fresh guest. Both interrupted processes are excluded from the nine accepted
runs; neither is a measured memory point. The serial backtrace enters
`vm_fault_trap`, `vm_fault`, `vm_map_lookup` and the VM-map lock upgrade. This
is a precise kernel failure site, not proof that a particular allocator
mapping or lock operation is the root cause.

The two Capstone modes have the same **outer-heap** peak of 835,056 B for
this workload. That ledger excludes the 64 MiB pymalloc payload region,
its separate 16 MiB metadata region and Capstone node storage. The node
high-water marks are about 57,230 in the spatial control and 207,222 with
Sublet, below the common 262,144-node capacity. The CheriBSD mode-0 backend
uses 2,159,424 B of its external metadata allocator; its jemalloc phase ledger
does not include the separate pymalloc mappings. These quantities cannot be
combined into a total-memory or cross-platform working-set ranking. There is
no complete four-arm CPython reuse, occupied-memory, or burst figure yet.

[summary.json](summary.json) lists the nine accepted process records, oracle
hash, policy and selected outer counters. [archive.json](archive.json)
identifies the 75-entry raw archive under `$CAPSTONE_TMP_ROOT`: process
transcripts, two excluded panic records and serial consoles, measured
binaries, standard-library zip, input, build manifests and logs. Its SHA-256
is `60e77322df147a0b7b1a298cf62cda4b613a85984cf35ac4db84555c5983060e`.
The [Capstone Sublet integration](../../../../ports/cpython/interpreter/patches/cpython-3.13.7-0014-pymalloc-under-sublet.patch),
[CheriBSD build recipe](../../../../ports/cpython/interpreter/cheribsd/build.sh)
and [shared guest runner](../../../applications/cheribsd-run.py) reproduce
the source and policy selection. Future memory figures need inner pymalloc
issue/release and metadata ledgers, a passing protected CheriBSD run and
matched repeated work schedules.
