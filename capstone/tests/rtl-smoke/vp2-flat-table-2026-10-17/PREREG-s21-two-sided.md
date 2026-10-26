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

## Pre-registered 2026-10-26, before any boot: the same pair on the collaborator PR stack (#250 #264 #258 #273 #274 #282)

Requested by the runtime lane before it lands the stack on dev. #250 rewrites the arena registry the retire looks up
(`find_arena`), #273 changes the image format (two load segments) and #282 the CV_STEP record. The retire block itself
is byte-identical: `s21-prefix.patch` applies to the merged source at an offset of 156 lines.

- **Tree:** the runtime lane's local merge chain fca004e7e063 on dev 8b2f10bed0c1, checked out clean in a worktree of
  this lane's. capstone_vm.c a4e5486fa5ce5b0a (dev's is e3da522ee472c743).
- **Modules** (18-bit flat, the twin recipe, QEMU headers = capstone-qemu 46b69c03a4's `cap_rev_table_abi.h`, no RVC):
  - prkitA: the merged source, module 9ab2eb23fac728ba.
  - prkitB: the same minus the retire block (source 97073d4f9fc46086, no `child frame grant not retired` line), module
    459a8096258e9afd.
- **Kits:** qkit7-b2twin3-q8 (QEMU 708bbff5663f0e86, firmware and images hard-linked) with two supervisor slots and the
  merged tree's launcher, capstone-vexec 62753dd693cc4a25 (equal to the runtime lane's adapter-merged build). They differ
  ONLY by the module.
- **Image:** frame-contract.c ba0b697e74e0289b (unchanged from dev), built with an SDK made from the merged tree (runtime
  43e6e03abba0dffc) and the frozen bitstream-2 compiler t3-freeze-r60: pr-frame-contract.dom 1ad2b7649cb31b96, two LOAD
  segments (R E, RW). The compiler PRs in the stack (#283, #275) are not in this image. The image is the same on both
  kits, so the compiler is not a variable here.
- **Run:** one boot per kit, insmod `supervisor_slots=2 diag_emerg=1`, modes in order twice, copy, prewrite, write-live
  (write-live last, because its child never exits). Same harness as above (virtualvm, unchanged between dev and the merge).

Predicted readings:

| mode | prkitA (fix a) | prkitB (no retire) |
|---|---|---|
| twice | first > 0, second = -16, child_ran=1, exit 0 | the same |
| copy | cause 25 (the exit-unmap race above: NOT separating) | cause 25 |
| prewrite | WROTE | WROTE |
| write-live | created > 0, child_ran=1, then cause 25 at frame+0x400; no WROTE | created > 0, child_ran=1, WROTE |

Must-fire lines on both: slot gate OK, banner `supervisor_slots=2`, `already backs a thread` 1 and `thread create
refused` 1 (from twice), `child frame grant not retired` 0, halts 0.

**What counts against fix (a) on the stack:** a WROTE in prkitA's write-live, or prkitA's write-live create refused
(created ≤ 0) when prkitB's succeeds, which would mean #250's registry no longer lets the retire find the frame arena.
**What voids the run:** prewrite not landing on either kit, prkitB's write-live faulting, or twice not reading -16 on
either kit.

### Addendum, before the re-run: the first boot pair was VOID on its image, which is rebuilt with the merged compiler

- **What the pair read** (pr-frame-contract.dom 1ad2b7649cb31b96): on BOTH kits, every step, prewrite included, died
  before `main` with cause 15 (store page fault) at `__capstone_cap_init`+0x38 (`stc a2, 0(a0)`), address
  `__atexit_lockptr`. That is musl's `const` capability-initialised holder, which this compiler put in `.rodata`, so in the
  R+X segment that the stack's launcher maps from the file without write permission. Prewrite did not land, so by the
  rule above the run is void. It says nothing about fix (a). Module lines: banner and `supervisor_slots=2` on both,
  halts 0.
- **The premise that failed:** "the compiler is not a variable here". The stack's `link.ld` relies on the compiler
  placing such holders in `.data` (#265, `49c1b69212eb`, on dev since 2026-10-21). t3-freeze-r60 (8807df747411) predates it,
  and so does the 2026-10-20 cc-dev (c0ccf4f36b32), whose musl object also has the holder in `.rodata`.
- **The change:** the SDK and image are rebuilt from the same tree with the stack's own compiler, cc-prmerged
  (clang fca004e7e063, binary b90e13006a016791, which carries #265, #283 and #275). Runtime 35c55ea403065902, and
  pr-frame-contract2.dom e06ee57a55ba7574, where `__atexit_lockptr` is in `.data` (the RW segment). Modules, kits,
  launcher, modes, predictions and voiding rules are unchanged.
- **For the stack itself:** under the stack's launcher, an image from a compiler without #265 dies at start-up whenever
  it carries a `const` capability-initialised holder, as this one does through musl's `__atexit_lockptr`. That may
  include bitstream-2 reference images built with t3-freeze-r60; it has not been checked image by image. This is a
  compatibility consequence of #273/#274, reported to the runtime lane, not a defect claim.

## Read 2026-10-26 (pr-frame-contract2.dom e06ee57a55ba7574): fix (a) separates on the stack; `copy` deviates on prkitB

| mode | prkitA (fix a) | prkitB (no retire) |
|---|---|---|
| twice | first=1 second=-16, child_ran=1 | first=1 second=-16, child_ran=1 |
| copy | cause 25 | **SURVIVED read=3 (exit 1)**, predicted 25 |
| prewrite | WROTE | WROTE |
| write-live | created=1 child_ran=1, then cause 25 at frame+0x400; no WROTE | created=1 child_ran=1, WROTE frame+1024, +1120, +1408 |

On both kits: slot gate OK, banner `supervisor_slots=2`, `already backs a thread` 1, `thread create refused` 1,
`child frame grant not retired` 0, halts 0. Every voiding rule passes and nothing counts against fix (a). **On the
stack, as on dev, the domain cannot write a live child's frame with the retire and can without it.**

**The deviation, reported as found:** `copy` on prkitB survived where 25 was predicted. `copy` reads with no wait, so
it races the launcher's exit unmap (`exec.c`, after CV_THREAD_EXIT, identical in both trees). Two readings fit and this
pair cannot tell them apart:
- the race moved (for example #282's shorter service round lets the parent read first);
- the stack's exit unmap no longer retires the frame. `unmap_arena(request.frame, 4096)` returning -EINVAL is
  tolerated silently, and #250 rewrote the registry it looks the frame up in.

### Pre-registered before any boot: `copy-exited`, the discriminator

- **New mode** in frame-contract.c (cd2c3d16290da162): a `child_flag` child. The parent waits for the flag, sleeps
  0.5 s and spins 4×10^8, then reads its copy. The child has then exited and the launcher has run its exit unmap.
- **Images:** pr-frame-contract3.dom 37c6f82232f3c078 (pr-sdk2, cc-prmerged, v7) for prkitA/prkitB, run as `copy`
  then `copy-exited`. dev-frame-contract3.dom 3dc08fb66c4566e6 (t3-freeze-nb-nomovc-cw-r60 SDK, v6, the SDK whose images
  ran on kitA/kitB above) for dev's kitA/kitB, run as `copy-exited`. One boot per kit, same insmod arguments.

| kit | copy-exited predicted |
|---|---|
| kitA (dev, fix a) | cause 25 |
| kitB (dev, no retire): **the control** that the wait is long enough | cause 25 (dev's exit unmap retires the frame) |
| prkitA (stack, fix a) | cause 25 |
| prkitB (stack, no retire) | 25 if the stack's exit unmap retires the frame; SURVIVED if it does not |

- **Void:** kitB SURVIVES (the wait is too short to show anything), or any create ≤ 0, or child_ran=0.
- **Reading:** prkitB 25 → `copy`'s survival was the race, and the stack's exit unmap works. prkitB SURVIVED with
  kitB 25 → the stack's child-exit unmap no longer retires the frame, a finding against #250/#264 for the runtime lane.
  `copy` on prkitB is repeated for N=2 and is reported either way.
