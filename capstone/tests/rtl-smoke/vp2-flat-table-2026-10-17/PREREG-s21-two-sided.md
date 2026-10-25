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
