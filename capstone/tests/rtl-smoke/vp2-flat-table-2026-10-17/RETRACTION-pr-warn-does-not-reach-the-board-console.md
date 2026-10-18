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
