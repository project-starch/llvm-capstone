<!-- The operating manual for the CheriBSD target. The corpus README
describes the corpus; this describes how to run it here. -->

# Running the corpus on PoisonCap/CheriBSD

The same `shared/defects.c`, and the same pinned CPython 3.13.7
`Objects/obmalloc.c`, also build as an ordinary CheriBSD purecap program against
the port's
[PoisonCap adapter](../../../../ports/cpython/pymalloc/host/cheribsd/poisoncap/README.md).
No case changes: the allocation sequences, the sizes and every `CHECK` are
shared between the two targets, and only the probe instructions, the markers and
the fault reporting are `#ifdef PYMALLOC_POISONCAP`-selected.

    source capstone/tests/capstone-test-env.sh
    export CHERI_SDK=/tmp/capstone/poisoncap-work/sdk
    export CHERI_SYSROOT=/tmp/capstone/poisoncap-work/output/rootfs-riscv64-purecap
    BUILD=/tmp/capstone/poisoncap-pymalloc-corpus-work/build/poisoncap
    bash capstone/ports/cpython/pymalloc/host/cheribsd/poisoncap/build.sh "$BUILD" \
      -DPY_CORPUS_SRC="$PWD/capstone/bug-corpora/cpython/pymalloc-repros/shared/defects.c"

    python3 capstone/bug-corpora/cpython/pymalloc-repros/cheribsd/run-poisoncap.py \
      "$BUILD" /tmp/capstone/pymalloc-defects-poisoncap-1 \
      --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
      --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
      --disable-default-revocation

and the control that makes the result mean something, which must exit 0:

    python3 capstone/bug-corpora/cpython/pymalloc-repros/cheribsd/run-poisoncap.py \
      "$BUILD" /tmp/capstone/pymalloc-defects-poisoncap-control-1 \
      --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image <image> \
      --disable-default-revocation --negative-control

The control's fixture declares two events and carries one, so the hosted entry
refuses the file (exit 3) before `pym_replay` and therefore before any
`defect()` branch: no ready marker, no fault line, no completion. The shared
runner reports every arm FAIL and exits non-zero, which is that control's
expected outcome rather than an infrastructure failure.

`--cases 5,10` takes a diagnostic subset and `--modes 0` a single arm. `--modes` also takes the arm names each `case.json` declares, so `--modes spatial,protected` and `--modes 0,1` are the same request; both are
recorded in `selection.json`, which also says whether the run was the complete
suite. Output directories must be new. All arms share ONE guest boot through
the common CheriBSD runner, and one arm failing no longer ends the boot, so a
single failure does not discard the arms behind it.

## Two modes, one binary Each case runs twice against the same `bin/defects`,
which picks its arm from the mode argument:

| mode | authority | required outcome |
|---|---|---|
| 0 | request-bounded spatial pointers, no per-object temporal invalidation | the sequence COMPLETES: `PYC_DEFECT case=N ready`, then `PYC_DEFECT case=N completed`, exit 0, and a 96-byte report reading `status=0 completed=1 count=1 mode=0` |
| 1 | PoisonCap lifetime invalidation on free and realloc | the stale read FAULTS: `SIGPROT`, `si_code == PROT_CHERI_TAG`, at the labelled `pyc_defect_read` instruction |

**The fault oracle is deliberately narrow, because exit 162 is not evidence.**
162 is the status of *every* `SIGPROT` on CheriBSD, so on its own it cannot tell
this corpus reproducing from an arbitrary crash, a bounds fault, a permission
fault, a tag fault somewhere else, a failure before the case marker, or the
allocator refusing the request. The program therefore installs its own
`SA_SIGINFO` handler before any case runs, reads the trap PC out of
`ucontext_t.uc_mcontext.mc_capregs.cp_sepcc`, and prints one line:

    PYC_DEFECT_FAULT case=5 signal=34 code=2 pc=0x98c4 expected=0x98c4 exact=1

`expected` is the address of the `pyc_defect_read` label itself, so a relink
cannot turn the check into a tautology. The runner requires the complete line,
with `signal=34`, `code=2`, the two PC fields identical as text, `exact=1`, the
case's ready marker before it, and exit 162 — every one of them, not any of
them. The handler then restores the default disposition and re-raises, so the
process still ends the ordinary CheriBSD way; a fault is never turned into a
clean exit. `cheribsd/test-run-poisoncap.py` is that oracle's own negative
control: it shows each of those rejections firing.

**Case 5 faults at the byte read, not at the pointer load.** The labelled byte
probe runs before `_odict_FOREACH`'s link load, so a protected run of case 5
shows the stale ACCESS refused and the pointer load never executing. The
wrong-node walk is observed only in the unprotected arm. Do not read a case-5
fault as the stale link having been followed and stopped.

**Scope, unchanged from the Capstone arms.** The allocator is real and the
consumers are reduced to the allocator calls each upstream defect makes, in the
same order. This is not a run of the CPython interpreter, its collector or its
extensions, and it says nothing about the free-list or `PyArena` layers above
pymalloc. The adapter is trusted and single-threaded; nothing here isolates a
hostile nested manager.

**Revocation configuration.** `--disable-default-revocation` turns the guest
libc's automatic revocation default off before SSH starts, which is the
documented workaround for the VM-locking failure of this published platform.
The adapter's own explicit PoisonCap sweeps stay on — they are what mode 1
measures — so this is not whole-process temporal protection.

Infrastructure failures are kept apart from measured case failures: a missing
guest summary, a boot that did not reach every arm, or a failing platform
control exits 75 rather than recording a case verdict. An application failure is
never retried.
