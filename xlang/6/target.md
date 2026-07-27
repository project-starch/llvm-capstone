# target.md — CVE-2026-1979 / mruby #6701 (Row 6)

* **CVE:** CVE-2026-1979
* **Upstream issue:** mruby #6701
* **Product:** mruby ≤ 3.4.0
* **Vulnerability Type:** Compiler bytecode corruption in the pattern-matching
  `JMPNOT`→`JMPIF` peephole optimization, yielding an out-of-bounds register write
* **Status:** REPRODUCED — but **not as a use-after-free**; see below
* **Vulnerable Tag/Commit:** `cda2567c36ca33cd404908ce2fa7bd55ea2a8ed9` (3.4.0-1476)
* **Fix Commit:** `e50f15c1c6e131fa7934355eb02b8173b13df415`
  ("mruby-compiler: fix bytecode corruption in pattern matching optimization")
* **Corruption Site:** `mrbgems/mruby-compiler/core/codegen.c:6632` —
  `s->iseq[fail_pos - 2] = OP_JMPIF;`
* **Crash Site:** `src/vm.c:1788` in `mrb_vm_exec` — the out-of-range register store
* **ASan Verdict:** `heap-buffer-overflow`, **`WRITE of size 8`**, 32 bytes past a
  1024-byte VM stack region
* **Determinism:** aborts on 10/10 consecutive native-ASan runs.
* **RISC-V QEMU:** **SIGSEGV, exit 139, on 3/3 runs.** Both legs of the deliverable
  are present for this row.

## Mechanism

The `expr in pattern` / `expr => pattern` codegen has a peephole optimization: when
a pattern has exactly one failure exit and that jump is the last thing emitted, it
rewrites the jump to its inverse and drops a redundant `JMP`:

```c
s->iseq[fail_pos - 2] = OP_JMPIF;
```

`fail_pos` is the position of the jump's 2-byte operand, so `fail_pos - 2` assumes
a **4-byte** `OP_JMPNOT` (opcode, reg, offset-hi, offset-lo).

But `NODE_PAT_PIN` has two paths. With the pinned variable **defined**
(`codegen.c:4519`) it emits a 4-byte `OP_JMPNOT` and the arithmetic is right. With
the pinned variable **undefined** — `lv_idx()` returns 0, `codegen.c:4524` — it
emits a **3-byte** `OP_JMP`. Now `fail_pos - 2` points one byte *before* the jump,
into the last byte of the **preceding instruction**, and the store overwrites it
with `38` (`OP_JMPIF`'s opcode number, `include/mruby/ops.h`).

Three conditions must hold together, and the trigger is shaped by each:

1. **Undefined pin** — selects the 3-byte `OP_JMP` path.
2. **Statement position** (value discarded). If the match's value is used, codegen
   emits `OP_LOADT` first, which breaks the optimization's `fail_pos + 2 == s->pc`
   guard and nothing corrupts. This is why `victim` ends in `nil`.
3. **A 2-byte preceding instruction**, so its single operand is the last byte.
   `5` compiles to `LOADI_5 R1` = `[opcode][dest-reg]`, so the store lands on the
   **destination** register: `LOADI_5 R38` — an out-of-bounds *write*. A 3-byte
   predecessor (e.g. a local value compiling to `MOVE`) corrupts the *source*
   operand instead and only misreads.

Confirmed directly by disassembly: `mrbc -v` on the trigger's method shows
`LOADI_5 R38` in a frame with `nregs=4`.

The recursion is needed because the top-level frame carries ~128 slots of slack, so
a stray write to R38 lands inside the allocation. Recursing marches the frame base
up the VM stack until `base+38` passes `stend`. Depth **29** is the first that
crosses; 29–31, 80 and 118 all fault.

## Bug-class discrepancy — needs a decision

The spec §6 and the benchmark table classify this row as a **use-after-free /
temporal borrow** ("UAF in `mrb_vm_exec`"). It reproduces as
**`heap-buffer-overflow` (WRITE)** — a *spatial* violation. Nothing is freed and
reused; a corrupted instruction indexes a register past the end of the current VM
stack allocation.

This is the same finding as Row 11, and it matters for the same reason: the
companion note claims "every defect here is a temporal borrow" and "There is no
spatial or single-domain row, by selection". Two of the fifteen rows do not hold to
that. See Row 11's `target.md` for the options; they apply identically here.

One nuance specific to this row: because the corruption is a *write* to an
out-of-range offset, it is arguably a more severe primitive than the temporal rows
— but it is still spatial, and bounds rather than revocation is what stops it.

## Relationship to Row 7 — they are NOT the same bug

An earlier version of this file asserted that "CVE-2026-1979 is the official CVE
identifier assigned to mruby issue #6701 (which is also listed as Row 7)" and used
that equivalence to skip both rows on one rationale.

The upstream fix commit `e50f15c1` says **"Fixes #6701"** and changes exactly the
pattern-matching optimization described above, so **CVE-2026-1979 and mruby #6701
are the same defect — this row**. But the spec §6 and the benchmark table describe
Row 7 as something different: *"UAF in `mrb_bint_reduce` (bigint gem), read by the
VM"*. That is not what `e50f15c1` fixes.

So the two rows were conflated in the wrong direction: the shared identifier is
real, but Row 7's *described* bug (a bigint GCD free) is a separate issue from the
pattern-matching corruption. See `../7/target.md`.

> **Supersedes an earlier SKIPPED filing.** The previous rationale claimed that
> "compiling and running mruby 3.4.0 with ASan triggers severe boot-time GC
> stack-scanning mismatches ... preventing the ASan-instrumented interpreter from
> booting successfully". That is false at this commit: the ASan build boots fine
> and runs scripts normally. The old `trigger.rb` was the single bare token `EOF`
> — a heredoc whose body was never written — so nothing was ever exercised.
