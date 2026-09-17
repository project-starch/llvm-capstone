# Compiler-lane handover to `apollo-compiler` (2026-09-17)

**For the successor session.** You take over the compiler lane: `llvm/` and codegen are yours by the
session split, along with the C-NN registry entries listed in §9. Everything below was re-read from
`origin/dev` at `80df35d54625` on the day of writing; line numbers rot, so re-read before citing.

**If you read only one section, read §2.** C-32's chosen fix is on dev and does not work. Everything
else can wait.

---

## 1. Read first, in this order

1. `capstone/docs/README.md`, then `docs/state/current-state.md` and `docs/state/current-next-step.md`.
2. `docs/ref/ISSUES.md:1448` — C-32, the live blocker.
3. `docs/ref/ISSUES.md:298` — Q-04, which is why no emulator pass can see C-32.
4. `docs/history/15-09-2026_02-40-00_c32-movc-scan-one-site-or-a-class.md` — carries its own
   retraction; read the retraction, not just the headline.
5. `docs/plans/bug-sweep-2026-09.md` — this lane's approved plan, four close-out blocks.
6. `docs/plans/2026-09-15-sublet-paper-follow-on.md:45-78` — F1, the task that produced §2.

---

## 2. THE LIVE BLOCKER — C-32 design A is on dev and does NOT fix `setupLookaside`

The lead chose design A (the rematerialisable bridge pseudo). It is implemented, gated, merged, and
**measured not to fix the site it was chosen for**.

### The measurement

`capstone/tests/movc-cfg-scan.py` over the Sublet cell ⑥ `-O2` image, before and after:

| | pre-fix `c506694f9f6f6889` | post-fix `113221f93b0ac994` |
|---|---|---|
| `setupLookaside` | `0x2679c` INT-ONLY | `0x267a0` INT-ONLY |
| its free path | `0x26a20` | `0x26a28` MIXED |
| `main` | `0x3aab4` MIXED | `0x3aabc` MIXED |
| `renameResolveTrigger` | `0x10b5b8` INT-ONLY | `0x10b5b8` INT-ONLY |

Same four functions, same count, same classifications, offsets shifted a few bytes. **Design A
removed nothing on this workload.** The board lane's independent scan found it; their gate is
worded by function precisely because offsets move between builds.

### The site, from the disassembly

    2677c: lcc  a0, t0, 0x3     inline asm: the lookaside base as an INTEGER
    2679c: mv   s3, a0          integer into callee-saved s3
    267a0: movc a0, s3          GPCR copy of an integer-defined value   <- the defect
    267a4: jalr a1
    267f4: mv   a0, s3          the source read again

On the RTL the `movc` nulls `s3`, so the read at `0x267f4` gets zero and the lookaside is silently
OFF. QEMU keeps the source (Q-04), so no emulator pass can see it.

### What is NOT the cause — tested, not assumed

The remat override removed in `46c53b7b6ae2` is **not** responsible. Rebuilt with it restored,
through the measure flow, the Sublet image is **byte-identical** (`113221f93b0ac994`) with the same
four sites. Removing it was correct, and that is now tested on an input that contains the shape.

### What is NOT known

**Why** design A misses this. The lit test's shape works — the bridged value stays in an integer
register and is re-bridged at each use (`mv`, not `movc`). This one does not: the value becomes a
capability vreg that register allocation copies. My hypothesis is that the copy is inserted by
`copyPhysReg` for the call argument *after* register allocation, too late for rematerialisation to
reach, which would make it the same family as the accepted PHI residue. **This is unproven and
should not be repeated as fact.** §10 says how to settle it.

### Why this is the lead's, not a bug to go fix quietly

Design A was the lead's choice, and the register-class alternative was rejected partly on design A
being sufficient here. That premise is now falsified, so the choice itself is back open.

### The registry is stale on both counts

`ISSUES.md:1448` still reads `DESIGN CHOICE STILL PENDING (lead)`. It never recorded that design A
was chosen, and does not record that it has been measured not to fix the site. `ISSUES.md` belongs
to the board lane under the session split — send them replacement text, do not edit it yourself.

### Three current records disagree, and none is a lie

* `ISSUES.md:1448` — design choice pending.
* `docs/state/current-next-step.md:25` — C-32 held by the author-line ruling.
* `docs/plans/2026-09-15-consolidated-board-queue.md:213` — ruling in, gated only on this lane's push.

The mechanical half is settled by git: the push happened, the fix is on dev, the author-line
question was waived by the lead (git authorship metadata only — no content, diff, token or email
hit). What is *not* settled is the design, which §2 reopens.

---

## 3. What is on dev, and what each commit does

* `46c53b7b6ae2` — `PseudoBRIDGE_CAP`: the `inttoptr` lowering
  (`CapstoneISelLowering.cpp`, the `VT == MVT::c128 && Op0VT == MVT::i128` case), the pseudo in
  `CapstoneInstrInfo.td`, and its expansion in `CapstoneExpandPseudoInsts.cpp`. **The expansion runs
  in `addPreEmitPass2`, NOT `CapstonePostRAExpandPseudo`** — MachineCopyPropagation is added by
  `addPreEmitPass` at -O2 and deletes `addi rd, rs, 0` when `rd == rs`, which would take the
  shadow-clearing write with it. The `rd == rs` case emits nothing, as `copyPhysReg` does.
* `19bc05cf21b1` — test-header correction (it had credited an override that was removed) plus the
  QEMU coverage record.
* `07bfcac91f14` — `SPEEDTEST1_SUBLET_ARENA` / `SPEEDTEST1_SUBLET_TABLES`. Without these the F1
  denominator cannot be regenerated by anyone: the board driver runs `--arena 2097152 --tables
  1750285`, arena raised with tables held, and both previously derived from one atom count.

Merged to dev at `e3bb47b43680`. Branch `compiler-validation-plan` is fully merged; working tree
clean; nothing unpushed. **`MOVC`'s definition is untouched** and must stay that way — C-46's box.

---

## 4. Artifacts, by hash (a rebuild is a new hash — cite hashes, never filenames)

| what | hash | where |
|---|---|---|
| cell ⑥ `-O2`, design A | `113221f93b0ac994` | `~/capstone-artifacts/c32-f1-2026-09-16/` |
| memsys5 `-O2` before / after | `ec061577fb008e18` / `a1f8f2093696d511` | `~/capstone-artifacts/c32-fix-2026-09-15/` |
| B6 set, `-O0`/`-O1`/`-O2` | `6cf8edf637f72063` / `50ca86aa70b3e425` / `ec061577fb008e18` | `~/capstone-artifacts/b6-2026-09-14/` |

Transfer branch `xfer/c32-f1-2026-09-16` at `d3dd7726028a` carries the cell ⑥ image and its two QEMU
records to apollo. **Never merge it.** Its logs are redacted — identifier substitution only, line
counts unchanged, pre-redaction hashes and the transformation recorded in its README. Those
artifacts remain valid as artifacts; they are simply artifacts of a fix that does not fix the site.

F1's two records, both of that one image: default arena 1,419,584 → 338,496,911 cycles / `HEAP
911104`; 2 MiB → 340,817,188 / `HEAP 1344064`, with sublet counters `5568/37966/32565/37966/5401`
matching the boot's pre-registration exactly.

---

## 5. Verification entry points

* **Freshness gate, before producing any artifact.**
  `python3 capstone/tests/toolchain-fresh.py --build llvm/cmake-build-debug` — 0 fresh, 1 STALE,
  **2 = cannot check, which is an error on purpose**. Read the status without a pipe; a pipe
  replaces `$?`.
* **Lit.** `llvm/cmake-build-debug/bin/llvm-lit -q llvm/test/MC/Capstone/ llvm/test/CodeGen/Capstone/
  llvm/test/MC/Disassembler/Capstone/` — 106 tests, all passing as of this handover.
* **QEMU suites.** `bash capstone/tests/run-nightly.sh --skip-build` (serial by design — they share
  the rootfs write lock; never run two). The single-domain smoke runner does **not** take the lock,
  so wrap it.
* **`capstone/tests/movc-cfg-scan.py` — the gate tool for C-32.** Takes path/label **PAIRS**; given a
  lone path it now errors instead of printing nothing. Run it from the repo root (it hard-codes its
  objdump path). **INT-ONLY = `len(strong) - mixed`** if reading an older copy whose summary line
  combined them.
* **`capstone/tests/c32-movc-scan.py`** — linear, **superseded for counts** (its own note says use
  the CFG tool). Kept for the def-side classification that separates the defect from the
  `real_cap_copy` control.
* **The F1 gate, in its corrected wording:** 0 integer-only sites *other than* `renameResolveTrigger`'s
  block-entry copy live around its back-edge, **named by function, never by offset**, with the mixed
  and opaque buckets reported beside. "0 integer-only sites" is unmeetable — design A's accepted PHI
  residue is itself an integer-only site.

---

## 6. THE APOLLO CONSTRAINT — you cannot build the SQLite domain images there

apollo's glibc cannot compile the SQLite domain TU. Mechanism:
`capstone/docs/design/hosted-libc-os-analysis.md:24-34` — the Capstone target exposes a capability
pointer model while the sysroot's glibc headers expect the ordinary RISC-V Linux ABI, and
`bits/wordsize.h` refuses.

Two things that look like escapes and are not:

* **`-DNDEBUG` does not help.** It is already passed (`build-sqlite-capstone.sh:111`). It governs
  what `assert()` expands to, not whether `<assert.h>` is reached.
* **`assert.h` is not special.** The amalgamation includes `stdio, stdlib, string, assert, stddef,
  ctype` unconditionally on consecutive lines; stubbing one moves the wall to the next.

Do **not** add `-I/usr/include/x86_64-linux-gnu`: it would bake x86_64 wordsize assumptions into a
`capstone64-unknown-elf` capability build and produce *a number instead of an error*, with nothing
in the output to reveal it. The board lane probed exactly that far and stopped, correctly.

**Consequence:** any C-32 measurement needs either a host where the build works, or images
transferred by hash. That is what `xfer/c32-f1-2026-09-16` exists for.

---

## 7. Rules that lived only in the previous session's memory files

apollo has none of these files. Each is one line; recreate the ones you want.

* **`negative-result-needs-a-loaded-input`** — the one that cost the most, see §8. "Flipping X
  changed nothing" proves nothing unless the input contained something X could change.
* **`one-arm-is-not-a-conclusion`** — the first finished arm of a multi-arm run cannot separate
  "this arm is special" from "every arm fails this way". Wait for all N.
* **`read-a-gates-intent-before-narrowing-it`** — a block that looks like a false positive is often
  deliberate. `precommit-scan` scans commit author lines **on purpose**; read a gate's comments
  before proposing it stop checking something.
* **`toolchain-binary-vs-source`** — after a merge the checkout binary is stale until rebuilt. Run
  the freshness gate before producing artifacts.
* **`no-toolchain-rebuild-during-suite`** — rebuilding swaps the shared-library compiler mid-run and
  voids the verdict.
* **`qemu-lock-is-one-constant-path`** — `CAPSTONE_QEMU_LOCK` from `capstone-test-env.sh`; set
  `CAPSTONE_QEMU_LOCK_HELD=1` for nested runners.
* **`gate-exit-status-and-removed-lines`** — `scan && commit`, never `scan; echo rc`. Invoke gates by
  absolute path: one that fails to start exits 127 and reads exactly like a pass.

---

## 8. What this session got wrong — one error, three times

Every validation of design A was measured on the **memsys5** image, which does not contain the
inline-asm bridge that C-32 was found on:

1. **"Design A closes the class."** Measured on memsys5: 1 integer-only site, down from 1 + 1 mixed.
   True of that image, and it does not generalise to the Sublet port.
2. **"The remat override is inert, remove it."** Byte-identical image — on memsys5. The conclusion
   happened to be right, confirmed later on the Sublet image, but the evidence did not support it
   at the time.
3. **The experiment meant to diagnose (1) and (2)** called `build-sqlite-silicon.sh` directly, which
   ignores `SPEEDTEST1_SUBLET`, and silently produced the memsys5 image *again*. Caught only because
   the scan output was digit-for-digit identical to the earlier memsys5 run.

**The Sublet cell ⑥ image is the only real test of a C-32 fix.** Build it through
`run-speedtest1-measure.sh` with `SPEEDTEST1_SUBLET=1` — that script applies the Sublet patch and
defines; `build-sqlite-silicon.sh` alone does not, and fails silently by giving you a valid image of
the wrong workload. Check the hash against `113221f93b0ac994`'s lineage before believing a scan.

Two earlier corrections in the same family are already in the record: the C-46 retraction
(`bug-sweep-2026-09.md:460`) and the `~190 sites` retraction
(`docs/history/15-09-2026_02-40-00_…:RETRACTED`, where a linear scan over-counted a CFG question by
two orders of magnitude).

---

## 9. The lane's registry entries, and the open lead decisions

**Structural trap:** only C-43, C-45, C-46 sit under `## Compiler / toolchain (ours)`. C-32, C-4,
C-14, C-17 are far above under `## RTL / FPGA`. Grepping the compiler section alone misses four of
seven. Line numbers are `origin/dev` at `80df35d54625`.

| ID | line | status |
|---|---|---|
| C-32 | 1448 | `OPEN — LIVE ON SILICON`, design pending per the file; **see §2** |
| C-4 | 3334 | `FIXED`, both halves re-verified; **not archived** |
| C-4a / C-4b | 3371 / 3405 | `FIXED 2026-07-28` |
| C-14 (superseded) | 4081 | `RETRACTED` |
| C-14 | 4137 | `FIXED`; two residuals rehomed → C-32 and C-46 |
| C-17 | 4348 | `LATENT BY DESIGN` — the crash is gone, a diagnostic stands in its place |
| C-46 | 5651 | `OPEN — LATENT HARDENING`; the fix shape the entry first implied is **WRONG** |
| C-45 | 5799 | `FIXED 2026-09-10` |
| C-43 | 5975 | `MITIGATED IN-BRANCH`; slot-allocated pools remain a **lead design item** |

**Open lead decisions this lane is waiting on:**

1. **C-32's design** — reopened by §2. The largest.
2. **C-43's slot-allocated pools** — a design item, not a bug.
3. **C-4's archiving** — blocked by `precommit-scan`: the commit that deletes the entry is blocked by
   the text it deletes. Narrowing the scan is a change to a release gate, so it is the lead's.
   There is also an unapplied retitle recommendation at `ISSUES.md:3360-3370`.
4. **Q-04** — whether a scalar source must be consumed is a spec question, and it is what makes
   every emulator pass blind to C-32.

**Closed — do not reopen.** B6 (lower the i128 `select_cc` as two i64 halves) was answered and
declined: the SQLite domain already builds and runs at `-O0`/`-O1`/`-O2`, and materialising a
>XLen constant would forge a capability. `bug-sweep-2026-09.md:512`.

**Two stale headers worth fixing when you touch them:**
`docs/plans/compiler-validation-plan.md:3` still says "PROPOSED … nothing here is executed yet"
above a 800-line execution log; `docs/plans/2026-09-15-sublet-paper-follow-on.md:75` still says the
fix is not on dev.

**Orphan:** `origin/backup/shrinkto-size-fix-2026-09-11` is an ungated, unsynthesised RTL change that
both the CHERI and compiler lanes have checked and disowned. Owner unidentified.

---

## 10. What I would do next — a recommendation, not an instruction

**Settle the mechanism of the `setupLookaside` miss before proposing any fix.** The question is
narrow: is the `movc` at `0x267a0` a copy the register allocator could have rematerialised, or one
`copyPhysReg` inserted for the call argument after allocation? Dump that TU's MIR after
`finalize-isel` and again after register allocation and look. The answer decides everything:

* If remat could reach it, design A is under-applied and may be extendable.
* If it is a post-RA `copyPhysReg` copy, design A **structurally cannot** cover this shape, it shares
  that limit with the PHI residue, and the register-class route the lead rejected deserves a second
  look.

Do not skip to a fix. Today's three wrong answers were all plausible, and the cost each time was a
rebuild plus a wrong claim in front of another lane.

**Who to talk to** (`ListAgents` names): `apollo-board` holds the board and F1's confirming boot —
they are holding it, correctly, on the strength of their own scan. The `board` sessions hold
`ISSUES.md` and the registry text. `apollo-paper` holds the numbers that F1 was going to de-caveat.
