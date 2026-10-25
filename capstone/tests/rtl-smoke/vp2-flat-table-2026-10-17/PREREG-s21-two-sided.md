# S-21 fix (a): the two-sided check ISSUES S-21 asks for, pre-registered 2026-10-26 before either boot

**Context.** Fix (a), the ruled module fix, has been on dev since ffb6fc71f523 (2026-10-20). CV_THREAD_CREATE
CSRETIREs the frame arena's ancestor once the page is pinned, refuses with -EACCES if the retire fails, and marks
`ancestor_retired` so the later unmap does not retire twice. It sits after the supervisor_slots -EBUSY refusal, so on
today's one-slot bitstreams it is unreachable and the board path is unchanged.

The five-item gate (RESULT-five-item-revision-gate.md, "Frame tests") read it one-sided only: `copy` READS the
domain's own copy after create. ISSUES S-21 asks for more: a domain that WRITES its child's frame+1024.. before the
ECALL reads 25 after the fix, and reaches the ECALL on the pre-fix module. This file is that check.

**The pair.** One variable, by source diff (`frame-tests/s21-prefix.patch`): module B is dev's capstone_vm.c with the 25-line retire
block at CV_THREAD_CREATE deleted, and nothing else.
- Module A is dev's capstone_vm.c (src e3da522ee472c743), built by the twin kit's recipe (18-bit flat). Its sections
  equal the kit's module 3325473537072a3a in 33 of 35. The two that differ are `.note.gnu.build-id` and
  `.rodata.str1.8`, which holds the one `__FILE__` build path.
- Module B builds with two expected warnings: the removed block's now-unused label and print counter.
- Both have 0 compressed instructions.

**The kits.**
- s21/kitA and s21/kitB are qkit7-b2twin3-q8 (QEMU 708bbff5, vexec fad465ce, images hard-linked), each with its
  wrapper appending `x-capstone-supervisor-slots=2`. They differ only in the module.
- Each boot insmods with `supervisor_slots=2 diag_emerg=1`.
- slot-match-gate.py must read OK before each boot.
- Without two slots a child frame never runs (Q-32 halts the guest).

**The test.** A new mode, `write1024`, in frame-tests/frame-contract.c:
- create a well-formed child frame (child_flag entry);
- WRITE markers through the domain's own copy at frame+1024, +1120 and +1408 (the snapshot's base, and the two
  readings of its sp slot);
- print WROTE;
- wait for the child to set child_ran before its exit ECALL.

The image is frame-contract.dom 527ae631233a93e1 (r60 SDK, -O1). Each boot runs `twice`, `copy`, `write1024`.

**Predicted readings:**

| mode | kitA (fix a) | kitB (no retire) |
|---|---|---|
| twice | first > 0, second = -16, child_ran=1 (as the five-item gate) | the same: the busy-frame check precedes the retire |
| copy | created > 0, then domain fault cause 25 | created > 0, then `SURVIVED read=3` (exit 1) |
| write1024 | created > 0, then domain fault cause 25 at the first write; no WROTE line | created > 0, WROTE, child_ran=1, exit 0 |

On both kits:
- the banner reads supervisor_slots=2;
- halts 0;
- `child frame grant not retired` 0.

**What counts against fix (a):**
- a WROTE line on kitA;
- any kitA fault cause other than 25;
- kitB faulting on the write. That would mean something other than the retire makes the frame unreachable, and fix
  (a) would be unmeasured by this check.

QEMU keeps the S caller internally, so on kitB the overwrite has no effect beyond landing. The check measures the
frame's REACHABILITY, which is fix (a)'s whole claim, not the RTL's restore.

## Read 2026-10-26 (frame-contract.dom 527ae631233a93e1): kitB's prediction FAILED, and the pair does not separate

| mode | kitA (fix a) | kitB (no retire) |
|---|---|---|
| twice | first=1 second=-16 child_ran=1 (as predicted) | the same (as predicted) |
| copy | created=1, cause 25 at frame+0x38 (as predicted) | **created=1, cause 25 at frame+0x38**, against the predicted `SURVIVED` |
| write1024 | created=1, cause 25 at frame+0x400 (as predicted) | **created=1, cause 25 at frame+0x400**, against the predicted WROTE |

On both kits: slot gate OK, banner `supervisor_slots=2`, halts 0, and `child frame grant not retired` 0.
- Module B was the one loaded: its staged hash is 51d59efd8b19c541, the same as the kit's.
- Module B does lack the retire: it has 2 CSRETIRE encodings to A's 3.

**RETRACTION.** The five-item gate's `copy` reading (RESULT-five-item-revision-gate.md, "Frame tests (S-21)",
`created=1, then cause=25`) has stood as evidence that fix (a) makes the domain's copy dead. Without the retire, the
same test reads the same 25. It does not measure fix (a), and neither does `write1024`.

**Why** (found with diagnostic modes after the read; diag.dom ad143468a3bc22599a50e6a82f0cf6e1, one boot per kit):

| diagnostic | kitA | kitB |
|---|---|---|
| prewrite (no create) | WROTE | WROTE |
| waitfirst (create, wait for the child to exit, write) | cause 25 | cause 25 |
| spin (create a child that never exits, write while it lives) | **cause 25** | **WROTE** |

When a child exits, the launcher unmaps its frame arena (runtime/virtual/exec.c, the CV_THREAD_EXIT path's
`unmap_arena(request.frame, 4096)`). That unmap's retire kills the domain's copy whatever the module did at create.
- `copy` and `write1024` use a child that exits at once, so they race the exit.
- The window fix (a) closes is create to child exit. Only a write while the child lives measures it, and there the
  modules separate.

## Pre-registered 2026-10-26, before its boots: the corrected two-sided check

frame-contract.dom b749e682daf358a0 adds two modes, run on fresh boots of both kits:
- `prewrite`: the control. Write the copy at +1024 with no create.
- `write-live`: create a child that never exits, wait for it to run, then write +1024, +1120 and +1408.

| mode | kitA (fix a) | kitB (no retire) |
|---|---|---|
| prewrite | WROTE | WROTE |
| write-live | created > 0, child_ran=1, then cause 25 at frame+0x400; no WROTE | created > 0, child_ran=1, WROTE |

The spin readings above are the same shape; these boots are the confirmation, since that finding was post hoc.

**What counts against fix (a):** a WROTE in kitA's write-live. **What voids the run:** prewrite not landing on either
kit, or kitB's write-live faulting.

## Read 2026-10-26 (frame-contract.dom b749e682daf358a0): the corrected pair separates, as pre-registered

| mode | kitA (fix a) | kitB (no retire) |
|---|---|---|
| prewrite | WROTE | WROTE |
| write-live | created=1 child_ran=1, then domain fault cause 25 at frame+0x400; no WROTE | created=1 child_ran=1, WROTE frame+1024, +1120, +1408 while the child lives |

On both kits: slot gate OK, banner `supervisor_slots=2`, halts 0, `child frame grant not retired` 0. With the
diagnostic spin boots above, this is N=2 per kit.

**So fix (a) holds where it matters.** From CV_THREAD_CREATE until the child exits, the domain cannot write its
child's frame at frame+1024.., the S caller's snapshot that every escape restores on the RTL. Without the retire it
can.

Scope:
- This is QEMU with two slots. QEMU keeps the caller internally, so it measures the frame's reachability, which is the
  fix's claim, not the RTL restore.
- On today's one-slot bitstreams the create is refused first, and the board path never reaches the retire.
