<!-- The operating manual for the CheriBSD target. The corpus README
describes the corpus; this describes how to run it here. -->

# Running the corpus on stock CheriBSD

The same `case.c` builds as an ordinary CheriBSD purecap program against the
port's [stock build](../../../../ports/apr/pools/host/cheribsd/README.md):
every node from the platform's own `malloc`, back through its own `free`,
exactly as upstream APR does. No adapter authority, no protected mode. The
variable is libc revocation — CheriBSD's own, kernel-defaulted temporal
mechanism — and the question is whether it sees a consumer allocating through
a pool it destroyed.

    source capstone/tests/capstone-test-env.sh
    export CHERI_SDK=/tmp/capstone/poisoncap-work/sdk
    export CHERI_SYSROOT=/tmp/capstone/poisoncap-work/output/rootfs-riscv64-purecap
    BUILD=/tmp/capstone/apr-pool-repros/cheribsd
    bash capstone/bug-corpora/httpd/apr-pool-repros/shared/build-cases.sh cheribsd "$BUILD"

    python3 capstone/bug-corpora/httpd/apr-pool-repros/runners/cheribsd/run-defects.py \
      "$BUILD" /tmp/capstone/apr-defects-cheribsd-on \
      --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
      --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
      --runtime-revocation on

and, so the two configurations can be read against each other, the same with
`--runtime-revocation off` — and the control that makes either mean something,
which must exit 0:

    python3 .../runners/cheribsd/run-defects.py "$BUILD" /tmp/capstone/apr-defects-cheribsd-control \
      --sdk ... --rootfs ... --image ... --runtime-revocation on --negative-control

## What is measured, and what would have to be true for it to fire

| `--runtime-revocation` | revocation-control must | the case must |
|---|---|---|
| `on` (the shipping default) | FAULT at `apr_defect_read`: `SIGPROT`, `si_code == PROT_CHERI_TAG`, pc equal to the address the supervisor resolved | COMPLETE: exit 0, a 96-byte report reading `status=0 completed=1 count=1 mode=0`, no fault observed |
| `off` | COMPLETE | COMPLETE |

The case completing under `on` is the finding. It is not "there is no
mechanism": `revocation-control` frees a block, sweeps, and reads through the
old pointer at the very same labelled `clbu` — and faults. The mechanism is
active and can fire at this shape. It does not fire for the case because APR
files a destroyed pool's node on `allocator->free[index]` and pops it straight
back for the next pool; under `APR_ALLOCATOR_MAX_FREE_UNLIMITED` it never calls
`free()` on that path, so libc's quarantine never sees the node.

**Nothing here is a self-report.** The supervisor (the pymalloc corpus's
`observe/supervise.c`, built with this corpus's label) runs each program and
reports what the kernel says: signal, `si_code` and PC from `PT_LWPINFO` and
`PT_GETCAPREGS`, the expected address from the child's memory map and ELF.
`revocation-control` is required to pass before any case verdict is recorded;
if it does not, the run exits 75.

**`on` is CheriBSD's default.** The kernel decides it at exec —
`imgact_elf.c`, precedence procctl, ELF note, system default, and the system
default is `security_cheri_runtime_revocation_default = 1`. The common runner
always exports one of `_RUNTIME_REVOCATION_ENABLE`/`_DISABLE`, so the `on` arm
re-asserts the default explicitly, and the ABI control verifies it in the
guest through `malloc_revoke_enabled()`.

**Negative control.** `--negative-control` corrupts the fixture so the case's
`CHECK(700)` refuses it before any pool is created: no completed report. Every
oracle must report FAIL, and `revocation-control` — which takes no fixture —
must still pass.

**What this does not do.** There is no PoisonCap build of APR, so no protected
arm exists on this target; `--modes 1` is refused. The guest image is the
PoisonCap platform's with its libc patched (`../../../cpython/pymalloc-repros/platform/`);
without that patch a process with revocation on dies in libc's own start-up,
before any of this runs.
