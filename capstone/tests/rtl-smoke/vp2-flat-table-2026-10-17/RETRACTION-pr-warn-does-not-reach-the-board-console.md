# Retraction: "the board console drops KERN_INFO, so pr_warn shows" — wrong, and it was mine to check

## What is withdrawn

`PREREGISTRATION-resolve-refusal-sites.md:38` states:

> **`pr_warn`, not `pr_info`.** The board console drops `KERN_INFO`; that is what voided r7's
> diagnostic…

The reasoning is **withdrawn**. The level of those lines was never the variable.

The pre-registration itself is **left unedited on purpose** — it was written and pushed before its
run, and a pre-registration that gets corrected after the fact stops being evidence. This document
is the correction.

## The actual cause

The board driver's `cold_boot` runs `echo 1 > /proc/sys/kernel/printk` in its quiet setup before
the first rung. From that point the console carries **KERN_EMERG and nothing else**. So:

- r7's `pr_info` (level 6) could not show;
- r11's `pr_warn` (level 4) could not show **either**;
- both were voided by the *same* cause, the console level — not by a difference between the two
  printk levels.

Confirmed against the built object: every diagnostic line in this module is level **4**, and the
one `pr_err` is level **3**. **None of them can reach that console.** The `pr_warn` lines cited as
proof that `pr_warn` shows ("remote fence extension is not available") were all printed during
boot, before the quieting.

**Consequence: r11's absent banner and absent site lines say nothing.** They are VOID, not a
reading. The same applies to any earlier conclusion that rested on a module line being absent in a
boot where that quiet setup ran — the audit does not stop at r8.

## How this got into my own work

The claim arrived from a peer lane and I repeated it into a pre-registration, a commit message and
a module comment without checking it. The project rule is explicit that a peer's report is a claim
and not a fact, and this is exactly the shape it warns about: the premise was checkable in two
places I had open at the time — the driver's quiet setup, and the module's own level prefixes in
the built object. **The banner design was right; its stated reason was not**, and a correct
mechanism resting on a wrong reason is not a safe thing to leave standing, because the next person
to reuse the reason will apply it where it does not hold.

## What changed in the module, and what deliberately did not

- **The load banner is now `pr_emerg`** (`CV_DIAG_TAG resolve-sites-4`). It is the one line that
  survives a console at level 1, and it is affordable there precisely because it fires once per
  `insmod`, before the rungs begin, where it cannot interleave with the ladder's markers.
- **The five site lines stay at `pr_warn`** and are read from the kernel ring buffer with `dmesg`,
  which the console level does not filter. Raising them to `KERN_EMERG` was considered and
  rejected: the driver quiets that console on purpose, and an unexpected line mid-stream can break
  the harness's own parsing.

The readings this gives are now unambiguous in both directions: **banner present** means the module
loaded at this revision and an absent site line really did not fire; **banner absent** means the
module did not load at all, which is a louder and different failure than a filtered print.

Board module: `06d663f30bf29825…` (`vermagic=6.4.14`, RVC 0, banner level 0, five site lines level
4, no banner left at level 4 — all checked in the object).

## Verified for the dmesg route, since r12 depends on it

- `CONFIG_LOG_BUF_SHIFT=17` → a **128 KB** ring buffer, and `CONFIG_PRINTK=y`. Confirmed in the
  board kernel's own `.config`.
- `dmesg` on the target is a **busybox** symlink, not util-linux; `CONFIG_DMESG=y` and the binary
  carries `klogctl`, so the applet is present and functional.
- **`dmesg -c` could NOT be confirmed from the binary** — the option spec is not a standalone
  string and the usage text is not adjacent — so it should be exercised on the board rather than
  assumed. A read of the whole buffer with no `-c`, delimited by a marker, needs no such option and
  fits comfortably in 128 KB.
- The positive control r12 relies on is real: `capstone_vm: thread create refused, 1 supervisor
  slot in use` fires **3 times** in each of the two QEMU controls, through the production harness.

## Channel audit of my own QEMU records — asked for by both lanes, and they stand

The same `dmesg -n 1` mechanism exists on the QEMU side (`tools/virtualvm.py:71` and others), so a
module line absent from a QEMU serial log is void the same way. Audited rather than assumed:

* **`run.py`, which produced all three of my controls, contains ZERO `dmesg -n` or `printk`
  references, and so does the `gate.sh` it bakes.** The console level is never lowered there. The
  six runners that do lower it are `run-malloc.py:68`, `run-hosted.py`, `run-ports.py`,
  `run-safety.py`, `run-fp.py` and `run-pthreads.py`.
* **Direction matters and it is favourable here.** My readings were *positive* — banner, six site
  lines, three `thread create refused` — and an observed line cannot have been suppressed. Their
  presence at level 4, at t≈271 s in the same log, is itself an internal control showing level-4
  output was reaching serial in that window.
* **The one absence claim I made — "zero `remap` lines" — therefore stands**, on both grounds: the
  harness never lowered the level, and level-4 lines demonstrably arrived during the window.
* **`run-malloc.py` is a harness I did use earlier**, so its records were checked too. The rows
  they rest on (`CAPSTONE_VM_ADD_REFUSED` and friends) come from `exec.c:210` via
  `fprintf(stderr, ...)` — launcher stderr, which `dmesg -n 1` cannot touch. `ea37cbd801e6`'s
  "the both-satisfied cell is empty" is a stderr-based absence and is unaffected.

**Defence in depth, since the above is one edit from being false:** `run.py` now reads the module's
lines from the ring buffer (`dmesg | grep 'capstone_vm:'`, prefixed `KMSG:`) after the `insmod` and
again before the `rmmod`, so QEMU and board records are read the same way. The banner is that
read's own control and it **reports** rather than printing nothing:
`KMSG_CONTROL_FAILED: no load banner in the ring buffer -- module-line absences in this run are
VOID`. Negative-tested both ways before committing: it fires on a banner-less buffer and is silent
on one with a banner.

**A commit title of mine that now rests on the withdrawn premise:** `69db2f133b93` *"Board-facing
refusals must be pr_warn: KERN_INFO is invisible in a board transcript"*. The conclusion it reached
(do not use `pr_info` for a board-facing refusal) is still right; its stated reason is not, for the
reason given above. It is on a lane branch and rewriting it would be a force-push, so it is
recorded here instead — which is where a reader following the trail will arrive.
