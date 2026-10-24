# The SQLite engine corpus on PoisonCap's published memsys5 (plan, 2026-10-24)

A side comparison, not an arm of the allocator study: the study excludes PoisonCap (decided
2026-10-07, because a port of it by us would be ours to defend). What differs here is that nothing
of PoisonCap's is ported. Their published SQLite and platform run unchanged; ours is the harness
around them. Whether the result enters the study is the project lead's call once it exists.

## Question

`bug-corpora/sqlite/engine-repros` holds 33 SQLite 3.22.0 cases served by memsys5. PoisonCap's
paper ports the same allocator in the same release (arXiv 2605.13210, §4.5 and §5.4) but runs no
security test through it: its Juliet and MSET results use the system allocator. Which of the cases
does their memsys5 catch, by the same rule our arms are scored by?

## Scope as run (decided 2026-10-24, before the corpus run)

Temporal cases only, PoisonCap's published arm only, lookaside on and off: two boots. Our side is
the recorded Sublet arm, not re-run. No spatial arm: a poison report (SIGPROT code 3 with the poison
line on their console) can only come from their mechanism, as cause 25 can only come from ours, so
each side is scored as "fault at site" and "mechanism-signed at site" without a control arm. The
`spatial` build and the attributable tier below remain available but are not part of this run.
Predictions: `bug-corpora/sqlite/engine-repros/results/2026-10-24-poisoncap/PREDICTIONS.md`.

## What the corpus can and cannot measure (read before any number)

- **27 qualified cases, not 33.** Cases 01, 04, 08, 09, 11 and 13 declare no `fault_sites`: host
  ASan is silent on their triggers, and case 01's record says its trigger never reaches the defect.
  They are reported as "no qualified trigger" on every arm and are outside the denominator.
- **Case 18 is a constructed demonstrator** of memsys5's in-band free list, not an upstream defect.
  PoisonCap moved exactly those links out of the heap (`mem5.c`, `mem5_link`). Its row is reported
  apart from the 26 upstream defects.
- **A fault in the declared function is not by itself a detection.** Our recorded Sublet arm
  (`results/2026-10-11-virtual-request-bounds`) scores 27 CAUGHT, but cases 02, 05, 19, 22 and 24
  also fault in the same function on stock memsys5 with no protection (`virtual-malloc`), and case
  02's host oracle is a plain SEGV, not a memory-safety report. A catch is therefore credited to a
  mechanism only against its matched control (below). On that reading our own arm's attributable
  catches are 22, not 27; this plan reports both numbers for every arm.

## Fairness rules

1. **Their mechanism unchanged.** Their `sqlite/` tree, QEMU, CheriBSD and LLVM, from the artifact
   pinned in `ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/platform.json`. Host-build
   workarounds (`ports/sqlite/poisoncap/build-platform.sh`) change no published source. A change to
   their platform's behaviour is a separately labelled arm, never folded in.
2. **Their heap backed as their driver backs it.** User data capabilities on their CheriBSD carry
   `CHERI_PERM_POISON` (`CHERI_PERMS_USERSPACE_DATA`, `sys/riscv/include/cherireg.h:136-138`), and
   their QEMU raises the poison fault
   only for a capability without it (`target/cheri-common/cheri-helper-utils.h:391`). A static
   array keeps the permission, and their `mem5.c` keeps the call that would strip it commented out
   (`memsys5Malloc`, `cclearpoisonperm`). Their `speedtest1` driver mallocs the heap, and their libc
   strips the permission on allocation. A static heap would switch their protection off, so their
   arm hands SQLite a heap capability without `CHERI_PERM_POISON`, the one change their `malloc`
   makes, in every case. Cases 07 and 22 pass the static `sqlite_heap` to `SQLITE_CONFIG_HEAP`
   themselves, so the mechanism has to reach them too (open design point for step 3: a malloc'd
   heap where the source allows it, or the same permission strip on the static array). A control
   first checks that the heap capability lacks the permission and that a planted memsys5
   use-after-free faults.
3. **The corpus's own cases and settings.** Each case's `case.c` from the corpus (not the older
   copies in `ports/sqlite/cheribsd/cases/`, 16 of which differ), the group flags read from
   `corpus322.sh` as `build-virtual.py` reads them, the heap sizes `build-virtual.py` gives (1 MiB
   from `repro322_common.h` unless a case's manifest entry says otherwise), the 64-byte minimum
   block, `-O0`, and the two source fixes every arm compiles (`fts5-azarg-patch.py`,
   `fts3-doclist-null-patch.py`; both apply once to their amalgamation).
4. **Lookaside rows.** Lookaside is set the same in every arm of a row. Their port does not cover
   it and ours does, so two rows: lookaside on (SQLite's default, our recorded setting), and off in
   every arm (`REPRO322_NO_LOOKASIDE`; ours `build-virtual.py --sublet --no-lookaside`). Two cases
   do not follow the switch and are recorded as such in both rows: 07 initializes SQLite itself and
   so bypasses it, 22 always turns lookaside off.
5. **Verdicts in three tiers, against matched controls.**
   - *fault at the site*: a fault whose PC is in a declared `fault_sites` function;
   - *attributable*: a fault at the site on the arm, and none at the site on its matched control,
     which differs in exactly the mechanism: `published` against `spatial` (their one
     `#define NESTED_TEMPORAL 1` removed) for temporal protection, `spatial` against plain memsys5
     on their platform for bounds; ours `virtual-nested-pools` against `virtual-malloc`;
   - *mechanism-signed*: attributable, and the fault is the mechanism's own report (their kernel's
     poison exception or a tag/bounds SIGPROT; ours cause 25, 28 or 29), not an untagged or
     misaligned dereference that a corrupted pointer produces anyway.
   The PC comes from a core file symbolized against the case binary. The CheriBSD harness's fault
   handler calls `_exit()` and so suppresses cores; this arm does not install it.
6. **Controls first**, in each arm: a planted memsys5 use-after-free and a planted memsys5 overflow
   must fault as the arm's design says, and an ordinary allocate/free/reallocate sequence must run
   clean, or the run is void.
7. **Predictions per case and arm committed before the run.** The run is repeated once.
8. **Their design properties are reported, not patched**, with a per-case counter where one can be
   read: bounds on the power-of-two block; the full quarantine (4,096 entries) freeing without
   revocation; revocation only past 16 MiB in use; the revoker's return value unchecked; on heap
   exhaustion `memsys5Malloc` calls `memsys5Size(NULL)`, which may fault instead of returning NULL.
9. **Allocation policies differ beyond lifetimes**, and both sides of that are recorded: ours bounds
   to the request (rounded to its representable size) and keeps a request table inside the heap;
   theirs bounds to the block and keeps its link table in a separate mapping. Usable atoms per heap
   are reported for each arm.

## Known risks, decided before the run

- **The published kernel panics in a normal boot** (`share->excl`, `vm_map.c:6103`, during rc after
  sshd starts; twice in two multi-user boots on 2026-10-24). Every run therefore boots the way all
  earlier PoisonCap runs did, through `ports/common/host/cheribsd/guest.py`: single-user, with
  `security.cheri.runtime_revocation_default=0` set on the console before sshd (the FFmpeg pilot's
  validated configuration, `ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot`).
  This turns off libc's automatic revocation for ordinary processes; memsys5's explicit
  `cheri_revoke` call is unaffected. It has two consequences that bear on rule 2 and on the libc
  defect below: with the default off, libc's malloc shim is not what strips `CHERI_PERM_POISON`, so
  the harness strips that one permission from the heap itself; and libc's never-removed poison on
  reallocated system-malloc blocks (`bug-corpora/cpython/pymalloc-repros/platform/`) cannot arise.
  Should a run need the default on, the libc fix (`mrs-poison-retire.patch`) is a separate,
  labelled arm applied to both PoisonCap arms alike.
- **Heap capacity.** Quarantined blocks stay out of memsys5 until a flush, so a case can run out
  of heap before its defect. Any SQLITE_NOMEM (or the exhaustion fault above) is NO-READING with
  its cause; a larger-heap row is a sensitivity check, not a substitute.

## Steps

| # | Step | State |
|---|---|---|
| 1 | `ports/sqlite/poisoncap/prepare-sqlite.py WORK`: artifact by its pin, `sqlite/` extracted, `mem5.c` checked against the audit hash, both amalgamations generated in a fixed environment | done |
| 2 | Platform on p13: `ports/sqlite/poisoncap/build-platform-docker.sh` (the recipe's `prepare.py` and `platform.sh` in a pinned container), then `guest-smoke.py` | done |
| 3 | `build-poisoncap.py`: the corpus `case.c` files for CheriBSD purecap against each amalgamation, malloc-backed heap | not written |
| 4 | Runner: one boot, controls first, per-case exit, signal, kernel report, core PC and fault site | not written |
| 5 | Predictions committed, then the run, twice | — |

Step 1, 2026-10-24 (WORK `/tmp/capstone/sqlite-poisoncap-corpus`, 65 s): archive sha256
`c5e8b07d4f70ed8c…` equal to the pin, 1,703 files under `sqlite/`, `mem5.c` `d8cb4c0e…`;
`sqlite3.c` `23a56929…` (published) and `05799d28…` (spatial), `sqlite3.h` `b02de0ea…` in both,
identical when regenerated in the fixed environment (cc 13.3.0, GNU Make 4.3, tclsh 8.6.14). The two
`sqlite3.c` differ in the removed define and in the line of SQLite's own `__LINE__` self-check.

Step 2, p13, 2026-10-24: built in the pinned Ubuntu 24.04 container with clang-18
(`ports/sqlite/poisoncap/build-platform-docker.sh`, no host workaround active): prepare 214 s and
verify 18 s (85,453 files), LLVM 385 s, QEMU 43 s, CheriBSD world and the CHERI-PURECAP-QEMU-POISON
kernel 806 s, image 9 s. An earlier attempt on the host toolchain (clang-21, GCC 15) built LLVM and
QEMU with two workarounds and then failed in CheriBSD's C++ bootstrap tools; it is not used. Boot
firmware: the stock CHERI BBL, sha256 `c8652f09…`, the file the earlier PoisonCap study recorded.
Kernel `ab2192d7…`, QEMU `35f1febd…`. Smoke (`ports/sqlite/poisoncap/guest-smoke.py`): booted
through `Guest`, `runtime_revocation_default` 0, `kern.features.cheri_revoke` 1, and the four
PoisonCap tests in their own cheribsdtest (`cheri_poison_mmap`, `cheri_poison_fault_read`,
`cheri_poison_revoke`, `cheri_poison_revoke_bounds`) each print PASS; no panic on the console.

Reviewed by Codex before any human review (2026-10-24): it found the poison-permission bypass on a
static heap, the six cases without fault sites, function matching crediting incidental crashes, the
wrong heap size (the first version of this plan said 256 KiB, which is the physical arm's), the two
cases outside the lookaside switch, and the unrecorded amalgamation environment. All are folded in
above.
