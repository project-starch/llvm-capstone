# On silicon, bounds only (2026-10-16)

`silicon-mruby-bounds`: the corpus on `caplifive_sublet_d93a8d66b`, one boot. The board lane ran it and recorded
the run in `bug-corpora/silicon-runs/2026-10-16-mruby-bounds` (dev 701bfbb399c2). It was pre-registered at lane
13f97a853606 + 3bff031359fd. `probe/fold-silicon.py` re-read the run from the board transcript and wrote
`silicon-mruby-bounds/matrix.tsv` and each case.json's arm.

**What ran:**
- B5's `mruby.dom` (a7ac283d7050a5d8): mruby 4.0.0-rc2 as a silicon delegated application, the same image for every
  case. Each case is the image run on its own `trigger.rb`.
- The allocator is `ports/mruby/app/silicon-alloc.c`: size classes, every object narrowed to exactly its request,
  nothing revoked. This is the corpus's `sysalloc-bounds` configuration on a different allocator.
- The gate: smoke.rb printed SMOKE_DONE and exited 0, and the tool's four controls read as declared.

**The rule,** pre-registered by the board lane, with the rows it did not cover agreed before its results:
- A spatial row is CAUGHT on a capability fault.
- A temporal row is MISSED as a detection, whatever it reads, because the arm revokes nothing. The note records how
  the defect shows.
- 17, both spatial and temporal, is CAUGHT only with cause 28.
- The other classes use this corpus's own rule and count in neither total.

**The readings:**

| | cases | silicon |
|---|---|---|
| spatial | 12-16 | 5 CAUGHT, cause 28 (ar_get, mrb_pack_pack, mrb_obj_ceqq, str_prepend, ary_fill_exec) |
| temporal | 01-04, 06-10 | 9 MISSED as detections. Of those, 01 and 07 faulted 24 in mrb_vformat, 02 faulted 24 in mrb_vm_exec, and 06 faulted 28 in kh_get_set_val (a declared site). 03, 04, 08 and 10 gave a wrong answer, and 09 PASSed |
| both | 17 | MISSED: 24 in mrb_func_basic_p |
| null dereference, type confusion | 18-21, 22 | a fault, cause 24 (mt_put; mrb_proc_parameters): CAUGHT under the corpus's rule, outside the totals |
| not run | 05, 11, 23 | 05 needs mruby-task; 11 is a C-API case; 23 does not terminate |

03 differs from the corpus's recorded `sysalloc-bounds` reading (PASS): B5's size classes reuse the stale format
string's memory, so the stale read sees the change. The board lane predicted this before the boot from the
silicon-mimic emulator, which reads the same.

**Two things recorded as read, not explained:**
- Every faulting run outlived the runner's 120 s watchdog after printing its fault line. Its block carries a WATCHDOG
  line before its END, and the row is read from the fault line.
- Case 22's fault address reads 0x7d8000 on silicon and 0x0 on the emulator, in the same function.

**Checked independently:**
- all 25 blocks are verbatim in the raw capture: one OpenSBI banner, one Linux banner, and the bitstream
  caplifive_sublet_d93a8d66b.bit;
- the 15 faulting runs' printed image hashes match;
- all 23 staged files (mruby.dom, controls.dom, the 20 triggers, smoke.rb) are in the booted Image's initramfs
  (af665ea2487e980a, firmware 6bf139245abaabc1) by sha256;
- every staged trigger is byte-equal to its case's `trigger.rb`.

N = 1 per case.
