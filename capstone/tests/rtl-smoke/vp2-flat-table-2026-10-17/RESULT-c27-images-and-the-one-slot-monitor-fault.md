# The C-27 fix on real images — validated; and a SECOND defect the new QEMU pin exposes

Two findings, from one 2x2 matrix. The first closes C-27 on images. The second is new, belongs to
the monitor, and it changes what boot 3 can read.

## The matrix — one variable per comparison

| images x QEMU pin | result | reading |
|---|---:|---|
| OLD `bf433e10` x OLD `f2f15f8fc70c` | 41/41 | the superseded kit record |
| OLD `bf433e10` x NEW `852c5bc76774` | 9/41 | **every** step exits 139 — the delin, cause 26 |
| NEW `f2017064` x NEW `852c5bc76774` | 1/41 | step 1 passes, then a **monitor** fault, cause 24 |
| NEW `f2017064` x OLD `f2f15f8fc70c` | **41/41, zero false** | **the control: the fix is good** |

Rows 3 and 4 differ in exactly one thing — the QEMU binary. Rows 2 and 3 differ in exactly one
thing — the images. That is what makes each reading attributable.

## Finding 1 — C-27 is fixed on images, and the gate is not vacuous

Census on the rebuilt images (board lane's `delin-census.py`, `vp2-tier12 2a01393a18e4`), reported
with TOTALS beside the zeros because a window count alone can pass vacuously:

| image | gp-derived delin | TOTAL delin | was |
|---|---:|---:|---:|
| contract.dom `f2017064bea77dd2` | **0** | **5** | 530 |
| sqlite3.dom `233640f7193b034c` | **0** | **5** | 5,988 |
| mruby `bba891bc16576d98` | **0** | **4** | 9,863 |

The totals fell by 525, 5,983 and 9,859 — exactly the old gp-derived counts. So the delins are
GONE, not moved outside the 5-instruction window, which is the vacuous-pass the board lane warned
about. The residuals are unchanged: the same 5/5/4 `ldc`-fed sites in the allocator glue.

And the rebuilt images score **41/41 with zero false tests** on the old pin, with the full expected
VM_EXIT vector — nine deliberate 139s, `preempt:143`, and `parallel_one:0 parallel_two:0
cleanup:0`. So the fix removes 16,367 delins across three images and changes nothing else.

Compiler identity, with a feature probe beside the version because a version string does not track
incremental rebuilds: clang 22.0.0git at `01b62884c0c7`; on the same input the default ABI emits
`delin` x1 and `-capstone-image-gp` emits x0.

## Finding 2 — the one-slot default faults the MONITOR, at step 2, not at the parallel pair

On `852c5bc76774` the rebuilt images pass step 1 and then die in `contract.dom threads` — the
SECOND step, long before the parallel pair:

```
[CAPSTONE] Print = Scalar(0xdeadbeef)
[CAPSTONE] Print = Scalar(0x1d)
[CAPSTONE] Cap mem access requires capability: pc = 8002522a, rs1 = x7, imm = 0, value = 0
[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x8002522a, tval = 0x0,
           badaddr = 0x3f98000010
```

`0x1d` is **29** — the CSRUNV refusal. `pc 0x8002522a` lies inside the monitor's own range
(`pc_cap = C(8002006c [8001a1d0,8002fe28) type 0)`). So the chain is:

1. one supervisor slot by default;
2. the second CSRUNV is refused, cause **29** — the monitor prints it;
3. the monitor's refusal path then dereferences an **untagged** value (`rs1 = x7`, `value = 0`) and
   takes cause **24** at its own pc;
4. the domain is halted, and the suite ends at step 2.

The control settles that this is the pin and not the images: the same images, adapter, module,
kernel and firmware on `f2f15f8fc70c` print `Scalar(0x1d)` **0 times**, `cause = 24` **0 times**,
and `threads` exits 0. One variable.

### What this changes for boot 3

* **The failure is at `threads`, step 2 — not at the parallel pair.** So moving the pair last does
  not rescue the run, and `--parallel-last` is not the mitigation it was meant to be here. The
  derived expectation of 34/41 with seven false does not hold: the reading is **1/41**, with only
  `normal` true.
* This is the M-mode analogue of what was predicted for S: a component that relies on the
  hardware's refusal rather than knowing the slot count. The kernel module needs the serialization
  already designed (one context outstanding, held from first CSRUNV to termination, blocking ioctl,
  `-EBUSY` only where blocking is not allowed) — and, separately, **the monitor must not fault on a
  refused CSRUNV.** The second is not fixed by serializing the module, because a monitor that
  dereferences an untagged value on that path is one bug away from the same fault whenever a
  refusal happens for any other reason.
* So the useful boot-3 prediction on `ed24f827b5beb373` is 1/41 ending at `threads`, not 34/41
  ending at cleanup.

## Provenance

Records `<v8-kit>/out/result-{new,old,newimg-oldpin}.json`; logs `runpy-*.log`; status files
`status`, `status-runpy`, `status-control`. Module `ed24f827b5beb373` and adapter-flat
`c4440adde0d02347` REUSED unchanged throughout — C-27 is codegen, and the ko is built by the
buildroot cross-gcc, so no module input moved. `cap_rev_table_abi.h` is byte-identical across both
pins (`257fbd0ea0ca8844`), so the ABI did not move either.
