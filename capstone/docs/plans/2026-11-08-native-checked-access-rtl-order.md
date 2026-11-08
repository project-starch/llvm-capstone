# Native Linux tasks on silicon: the RTL side, in order (8 Nov 2026)

Scope set by the runtime lane after reading `native/linux-task-integration`.
- The saving is the direct "register" syscall ABI: 3.5 k against 15.5 k instructions per small call on QEMU.
- That route gets its addresses from CSCHECKR/W and the epoch/kinds words.
- Export/import (CSRUNV 4/5), paged tables, action 3, CLDRAW/CSTRAW/CPTAG and the copyable profile are needed only to
  load the collaborator's full adapter, not for the saving.
- The runtime lane is measuring the same saving in software first (in-module call servicing with software checks on
  today's silicon), and that result decides whether anything below is built.

Base: C.5a, capstone-ariane 1fc2cd5aa. Contract: capstone-qemu dd041bac5ad9 `tests/capstone-checked-copy/README.md`
and the handoff's section 3.

## What C.5a already has and lacks

**The frame.** The RTL switcher uses these offsets (`capstone_dom_switcher.anvil`, the runv walks):
- +0..+56 header;
- +64..+575 the context's registers (+64+16r);
- +576..+639 (written by the escape beside +224..+336);
- +656..+912 the context's FP registers and fcsr;
- +920..+944 mstatus and event words;
- +1024..+1976 the caller snapshot;
- +2048..+2312 the caller's FP state.

**Free: +640..+655 and +2320..+4095.**
- The native image's immutable GPR area (+2400+16r, 512 bytes) already fits.
- Its epoch (+656) and kinds bitmaps (+664) collide with the RTL's FP save.
- Moving them is a software ABI change (QEMU property, kernel, transport v6), e.g. to +2320/+2328. No RTL is needed
  for the move.

**What the check reads:**
- The DYN unit already has a liveness query against the lifetime table (`get_node_query_validity`, used by
  LDC/CDERIVE/CSRETIRE).
- The monitor already emulates one instruction S cannot execute (CSMINT, `capstone_csmint.S`, decoding mtval on the
  illegal-instruction trap).
- CSCHECKR/W themselves are illegal on C.5a (decoder default).

**The two properties the contract needs from the ESCAPE, and C.5a does not provide:**
1. A fresh, non-wrapping epoch published at every escape.
2. A HIDDEN escaped-register snapshot that trusted frame rewrites cannot change. C.5a's escape writes only the
   kernel-writable +64+16r slots.

## The order

| # | Item | Where | Storage | Logic / critical path | Bitstream? |
|---|---|---|---|---|---|
| 0 | Move epoch/kinds to a free offset (+2320/+2328) | QEMU property + kernel | none | none | no |
| 1 | **Measure first:** CSCHECKR/W emulated by the monitor on the illegal-instruction trap. It LDCs the slot at +64+16r from M, checks tag/type/perm/bounds, reads the 64-bit lifetime record, checks satp/srevroot/privilege, and writes the output scalar. The epoch is a kernel-written word | firmware | none | about 300-600 M-mode instructions per check plus a trap round trip, against the 12 k instructions the direct ABI saves per call | no |
| 2 | Escape publishes epoch + kinds: a 64-bit hart epoch counter (never wraps; refuse at saturation), and two 32-bit tag/raw-tag bitmaps built from the tags the escape SAVE walk already carries; two extra stores at the end of the walk | switcher (Anvil) | about 130 FF | small; the stores lengthen the walk by two beats | yes |
| 3 | Hidden escaped snapshot: 31 x (128-bit word + tag) captured during the escape SAVE walk; a 31:1 read port for the checker | switcher + new storage | about 4,000 FF (or LUTRAM: 31 x 129 bits) | about 1,300 LUT of read mux as flops, much less as LUTRAM; capture rides the existing walk | yes |
| 4 | CSCHECKR/W execution: decode (funct7 0x25/0x26); a multi-cycle DYN op that loads the 48-byte descriptor (3 beats), reads the snapshot (item 3) or the frame slot, decompresses its bounds, and does the offset+length vs bounds compare (64-bit add + 2 compares). Also: perm/type/tag, epoch/satp/srevroot/privilege compares (4 x 64-bit), the liveness query (existing port), the output store, rd = 0 or 24..29 | DYN unit (Anvil) + decoder | about 400-600 FF (FSM, descriptor words) | about 1,500-3,000 LUT. One more decompress instance is the wide function. Keep every compare behind flops (FSM states), never on the result endpoint: that endpoint is the DYN priority chain already blamed for IRO growth | yes |

**Gates per the RTL rules:**
- lint, plus claim-auditor;
- the 95-sweep;
- directed arms mirroring QEMU `run.py`: decode, causes 24..29, immutable-frame, replay/epoch, the atomic-record
  injection;
- a synthesis-only size gate with the per-module LUT delta, BEFORE a full build.

The design is at 169 k LUT (b3 placed) against two routing failures near 178 k. Items 2-4 add about 2-5 k LUT and
4.5 k FF (item 3 as flops), so they fit only if item 3 goes to LUTRAM or the snapshot stays in frame RAM at
+2400+16r. Frame RAM is not hidden: the kernel can rewrite it, which the contract forbids for authority.

## Recommendation

Do 0 and 1 first. Item 1 answers "is the direct ABI's saving real on silicon" with no bitstream, and so without
re-measuring every Plain denominator. If the saving holds, build 2-4 as ONE bitstream (batching rule), with item 3 in
LUTRAM, and pre-register its gates. If the runtime lane's software-check measurement already answers the question,
item 1 is redundant and the decision moves straight to 2-4.

Not in this order: CSRUNV 3/4/5, paged tables, CLDRAW/CSTRAW/CPTAG, the copyable-profile decoder changes. They are
needed for the collaborator's full adapter, not for the saving. Size them only if that adapter becomes the goal.

Estimates are desk estimates from the C.5a source, not synthesis readings.

## Addendum (8 Nov): steps 0 and 1 run now, in parallel, handed to their own agent

The lead's word, relayed by the runtime lane: run steps 0-1 now, in parallel with the optimisation line, not gated on
any other result. Steps 2-4 stay gated on step 1's number. Anchors for whoever takes 0-1:

**Where the words are read and written today**
- Kernel: `native/linux-task-integration`, `capstone/runtime/virtual/module/checked-copy-dom.h:125-134`. It reads the
  epoch as `frame[82]` (+656) and issues `.insn r 0x5b, 1, 0x26` (CSCHECKW) / `0x25` (CSCHECKR), rd = status,
  rs1 = the frame's physical address, rs2 = the 48-byte descriptor's physical address. The module parameter is
  `checked_copy`, default off.
- QEMU: capstone-qemu dd041bac5ad9, property `x-capstone-checked-copy`. Each escape publishes the epoch at +656 and
  the tag/raw-tag bitmaps at +664. The semantics are in `tests/capstone-checked-copy/README.md` (descriptor words
  0-5, causes 24..29, the checks). The check reads a HIDDEN escaped snapshot.

**Step 0 (software only)**
- Move the epoch to +2320 and the bitmaps to +2328: the QEMU property's publication, and the kernel's `frame[82]`
  -> `frame[290]` (and the bitmap reader).
- **Second collision, outside the saving:** transport v5/v6 puts the escape epoch at +1072 and the caller privilege
  at +1080. Both are inside the RTL's caller snapshot (+1024..+1976). This matters for export/import images only,
  not for the direct ABI; record it beside the move and leave it.

**Step 1 (firmware emulation, measurement only)**
- Model it on the monitor's CSMINT emulation: caplifive-sbi `vp2/csmint-emu` 868dd5f (`capstone_csmint.S`), plan
  `docs/plans/2026-10-17-vp2-monitor-csmint-port.md`; a kit copy is
  `tests/rtl-smoke/vp2-bare-2026-10-17/tests/inc/capstone_csmint.S`. Decode mtval on the illegal-instruction trap
  from S (funct7 0x25/0x26), emulate, write rd, and step mepc.
- The emulation reads the MUTABLE slot +64+16r (C.5a has no hidden snapshot) and takes the epoch from the moved
  software word. Both are measurement-only shortcuts; say so wherever the number is cited.
- M in capability mode needs a capability base for every access (an integer base reads cause 5 on a load and 24 on
  a store). Mint over the frame, the descriptor and the lifetime table.
- On QEMU, run with `x-capstone-checked-copy=false` so the instructions really trap to the firmware. Positive
  control: with the property ON, QEMU executes them natively and the firmware is never entered. Count both.

**The number wanted:** exits and instructions per delegated call, on QEMU, for perlbench test and gcc train01, on
the exchange route (before) and the direct ABI with the emulated check (after). On silicon, the firmware trap adds a
trap round trip per check; that cost is measured on the board once QEMU says the saving is worth it.
