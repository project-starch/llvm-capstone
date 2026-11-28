# mruby/release-differential on virtual Capstone with the one-operand lifetime instructions -- 2026-11-11

The two virtual arms of [2026-10-11-virtual-capi](../2026-10-11-virtual-capi/README.md), run again after the lifetime
instructions changed: `CREVOKE` takes the object alone. Patch 0008 (`MRBD_SUBLET=1`) was converted: every GC object
slot is a `CDERIVE` child of its page's own pointer, without the page's intermediate lifetime, and the sweep ends it
with `capstone_cap_release`. Runner: `probe/run-virtual.py`, one `capstone-vexec` process per case, 120 s per case,
case 11 through its C-API driver.

| arm | image | control `smoke.rb` | fault | completed |
|---|---|---|---:|---:|
| `virtual-malloc` | `e8593fdae83a8215` (stock) | completed, `SMOKE_DONE` | 20 | 3 |
| `virtual-nested-pools` | `0391d7d629e25700` (`MRBD_SUBLET=1`, patch 0008) | completed, `SMOKE_DONE` | 23 | 0 |

**44 of the 46 cells have the outcome, the cause and the function of the 2026-10-11 record**, case 11's `reached` and
`in_fault_sites` included. The three GC-slot rows that only the protected arm catches are 04, 05 and 08, each cause 25,
as before; 02 is cause 24 on stock and cause 25 in `mrb_gc_protect` on the protected arm, as before.

## The cell that differs on both arms: case 23, the function only

Case 23 faults with cause 28 on both arms, as on 2026-10-11, in `mrb_ary_new_capa` where the record has
`mrb_obj_alloc`. It is not the instruction change and not patch 0008: the stock arm reads the same, and so does mruby
built from unmodified dev (f8d6cb8bc, the two-operand patch) on the QEMU dev pins, with the compiler of this run
(cause 28 in `mrb_ary_new_capa`+12). The 2026-10-11 images were built with another compiler.

## Runs, and a process that stops making progress

The two bundles here are the second run of each arm, in two VMs side by side. The first run of `virtual-malloc`
(`virtual-malloc-first-run/`) has the same 23 rows.

The protected arm needed four attempts, and the reason is recorded because it is not the patch's conversion:

| attempt | what happened |
|---|---|
| 1 | 24 QEMU sessions of the same campaign shared the host; `smoke.rb` ran past the runner's 120 s, so no row is a reading. Stopped |
| 2 | 01 to 07 as recorded; 08 ran into the bound with no output. Stopped |
| 3 | 01 to 05, 07 to 13 and 15 as recorded; 06, 14, 16 and 17 ran into the bound. Stopped at 18 |
| 4 | all 23 rows; the bundle of record |

- A case that ran into the bound had printed nothing and raised no fault: the QEMU log has no capability fault for it.
  Its process was runnable and spent its time in the kernel (attempt 2, case 08: 28 s of user time against 185 s of
  system time when sampled).
- It is not tied to a case. 08 faults with cause 25 in `mrb_vm_exec` in 11 s and in 19 s when run again on fresh VMs,
  after 09 alone and after `smoke.rb` and 01 to 07; 06 faults with cause 25 in `kh_get_set_val` in attempts 2 and 4.
- **It happens on unmodified dev as well.** The same sequence with mruby built from dev (f8d6cb8bc, the two-operand
  patch 0008) on the QEMU dev pins had case 07 in that state for 333 s, 41 s of user time against 270 s of system
  time, until it was killed; 01 to 06 and 08 faulted with the recorded cause there.
- The module of this platform has one context, so a process the runner's bound has ended on the host keeps the
  context in the guest and every later case waits for it. In attempts 3 and 4 a watcher beside the runner killed such
  a process in the guest. The runner and its bound were not changed.
- The stock arm did not show it in two runs of 24 processes each. What the processes do in the kernel was not
  established. The host's load average was between 47 and 77 on 80 cores during all of this.

## Platform and inputs

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. Patch 0008 is 31c512ae806b....
- Builds: `ports/mruby/app/build-mruby-domain.sh`, `MRBD_PIN=4.0.0-rc2`, `MRBD_SDK` naming a copy of the corpus SDK
  (`CAPSTONE_SDK_CORPUS=1`, effective `-fno-builtin`), which the runner requires since the 2026-10-11 record.
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters (pinned,
  one context), `capstone_vm up --profile virtual --exact-bounds --cpu-prop x-capstone-test-csrs=true`. Each bundle's
  `inputs.json` carries the image and platform hashes; it does not record the CPU property.
- Lifetime instruction words in the images: stock 6 `CDERIVE` and 3 `CREVOKE` (the C library's mallocng), protected 8
  and 5. No `CRENEW`.

N = 1 per case cell in each bundle.
