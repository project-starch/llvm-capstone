# mruby freestanding: two faults, one dead control, one refuted class

The freestanding mruby port builds and runs again, reaches `mrb_open_core`, and dies there.
This note records what was measured getting that far, including one retraction.

## The port was not buildable and two repairs were needed

* **Foreign headers.** `build-mruby-silicon.sh` took `string.h` and friends from
  `benchmarks/micropython/adapted/include`, `capstone_setjmp.h` from MicroPython's port and
  `inttypes.h` from WAMR. On a branch carrying only mruby none of them exist. The port now
  has its own `adapted/` tree and is self-contained.
* **A duplicate builtin.** The shared softfloat list took `floatunsisf` on 2026-09-05;
  mruby's own loop still compiled it under the same object name and appended it twice, so
  `ld.lld` reported a duplicate symbol against the object itself. The port had not been
  linked since that change.

## Where it dies

Five rungs return, the sixth does not:

| call | what | result |
|---|---|---|
| 1 | anchor | load address |
| 2 | entry, cap-init, return channel | OK |
| 3 | outer allocator, malloc/realloc/free | OK |
| 4 | **narrowing control** | 64, so the outer allocator does narrow |
| 5 | probe self-test | fires |
| 6 | `mrb_open_core` | **cause 7** |

```
Cap mem access OOB: pc = 101727cdc, rs1 = x10, cursor = 1021722d0, imm = -16,
                    addr = 1021722c0, size = 8, bounds = (10216e0c0, 10216e110)
```

Mapped by hand: `domain_main` is at image offset 0x58378 and the anchor reports 0x1748378,
so the load base is 0x16f0000 and the pc offset is **0x37cdc, inside `mrb_vm_run`**
(0x37adc..0x37d78). The faulting instruction is not the reported pc, which is the
translation block entry; it is the `sd zero, -0x10(a0)` of the stack clear at
`src/vm.c:1808`, `stack_clear(c->ci->stack + stack_keep, nregs - stack_keep)`.

The capability in `c->ci->stack` carries **80 bytes**, and `sizeof(struct RProc)` is 80
while `sizeof(mrb_callinfo)` is 96. Every 80-byte object in the image is an RProc.

## RETRACTED: the narrowing arm never differed

The previous reading excluded the allocator with `MRUBY_NO_NARROW=1` and stated the knob was
real because the narrowed image carried two `shrink` instructions and the wide one none.

Both images are **byte-identical**, md5 `4f369a5d283b`, and both carry two `shrink`.
`-DCAPSTONE_HEAP_NO_NARROW` appeared **exactly once in the whole repository**: in the line
that defines it. `cap_narrow` never read it.

`cap_heap.c` now honours the macro. With it the image does change, and the `shrink` that
disappears is the one inlined into the allocator at 0x10f40; the one that remains at 0x57f68
belongs to the probe's own self-test, so the arm is clean.

Everything that leaned on that arm has to be taken again: the allocator is **not**
exonerated, and neither is `envadjust`, which was excluded "for the same reason". And since
`cap_narrow` shrinks to exactly the requested size, an 80-byte capability is precisely what
this allocator hands back for an 80-byte request.

## REFUTED, with a control that fires: the struct-copy class

`tests/runtime-qemu/silicon-ladder/cistk_*` reproduces `mrb_callinfo` (96 bytes, fields at
the same offsets), the `static const` copy, the proc/blk stores and the `cipush` derivation,
and returns the LENGTH of the capability that lands in the stack slot.

* stage 3, the full sequence: **1024**, correct.
* stage 4, the control, RProc capability put there on purpose: **80**.

So the shape is not the defect, and the clean answer means something because the control
produces the other one. `cipush` is separately exonerated by its own disassembly: it loads
`ci[-1].stack` at -0x30, adds `push_stacks*32` with `cincoffset` and stores at 0x30.

## The double-ldc arm moves the fault

`MRUBY_DOUBLE_LDC=1` is a real knob: 22,412 `ldc` become 38,640 and the image differs.
With it the stack clear **completes** and a different fault appears further on:

```
Cap mem access OOB: rs1 = x10, cursor = 10256e020, imm = 160, addr = 10256e0c0,
                    size = 16, bounds = (10256e020, 10256e0c0)
```

160 bytes of bounds, a 16-byte access at exactly the end. The faulting instruction is
`stc s9, 0xa0(a0)` at 0x3a9a8, part of a register save that writes ra, s0..s11 and sp at
16-byte strides up to 0xd0, i.e. **setjmp saving into a jmp_buf**. Measured with the
compiler, `sizeof(jmp_buf)` and `sizeof(struct mrb_jmpbuf)` are both **224**. The buffer is
the right size; the capability addressing it carries 160.

Both faults therefore have the same shape: **a capability whose bounds are too small for the
object it addresses**, once for a VM stack pointer and once for a jmp_buf.

## What the probe says, and why that is the sharpest open question

The probe at `src/vm.c:1800` reads `c->ci->stack` twice, checks that its bounds contain the
range the clear will write and that they lie inside `stbase`, and escapes by longjmp on
`bad || moved`. **It never escaped.** So at probe time, a few instructions before the store,
the capability looked sound and unchanged between two reads -- and the store then faulted
through it.

The reporting rungs sit at calls 15 and above and the domain dies at call 6, so the recorded
geometry has never come back. `MRUBY_ESCAPE_AFTER=1` forces the escape at the first frame
and is the next run.


## ROOT CAUSE CANDIDATE: the value is re-loaded after the probe validates it

The disassembly at the fault settles why the probe and the hardware disagreed. The probe is
a CALL, and both operands are reloaded after it returns:

```
37cf8: jalr a5              the probe call, returning nregs
37cfc: sub  a1, a0, s4      nregs - stack_keep
37d00: beqz a1, ...         skip if the count is zero
37d04: ldc  a1, 0x30(s7)    c->ci      RE-LOADED
37d0c: ldc  a1, 0x30(a1)    ci->stack  RE-LOADED
37d1c: cincoffset a0, a1, s4
37d24: sd zero, -0x10(a0)   the faulting store
```

The probe validates the pointer it was PASSED; the clear writes through a pointer it reads
back from memory afterwards. They are different loads of the same address, and they do not
agree.

Four measurements fit that and nothing else:

* the probe reports `ci->stack` at 4096 bytes, `nregs` 4, `stack_keep` 0, **0 of 1 clears out
  of bounds** -- the first load is right;
* the fault reports 80 bytes, `sizeof(struct RProc)`, immediately after the reload;
* `-capstone-double-ldc` makes this fault disappear and the clear complete, which is what a
  doubled load does when the first of the pair is the wrong one;
* `MD_ESCAPE_AFTER=1` survives and answers all 60 rungs while `=2` dies, so the clear is
  reached exactly once and jumping out before the reload is what saves it.

So the defect is a LOAD, not mruby and not the allocator. That is the S-12 / R-20 family,
already in the registry. mruby is the vehicle that exposes it, not the subject.

**Not proven, and the next thing to attack:** doubling does not rescue the second fault. In
the double-ldc image the jmp_buf pointer is itself loaded twice (`ldc a0, 0x10(s6)` at
0x3a95c and 0x3a960) and still carries 160 bytes where the object is 224. Either the wrong
value is what sits in memory there, or the defect is not simply "the first read of a pair".

## RETRACTED: neither fault is a load defect. Both are one inlined naked function

The section above is wrong. The defect is not S-12 / R-20 and it is not a load. Both faults
come from a single cause, and it is entirely inside this port.

**`setjmp` is `naked`, and mruby gets it INLINED.**

```
remark: 'setjmp' inlined into 'mrb_vm_exec' with (cost=always):
        always inline attribute at callsite mrb_vm_exec:178:3
```

The chain, each link checked against its source:

1. mruby marks `mrb_vm_exec` `MRB_FLATTEN` = `__attribute__((flatten))` (`src/vm.c:2491`), so
   its extracted opcode handlers are forced back inline.
2. `flatten` makes clang stamp `alwaysinline` on **every** call site in that function --
   263 of them in the frontend IR, including `MRB_TRY`'s `setjmp((&c_jmp)->impl)`.
3. `getAttributeBasedInliningDecision` honours a call-site `alwaysinline` at
   `InlineCost.cpp:3217`, **before** it looks at the callee's `noinline` at 3246, and
   `isInlineViable` does not reject a `naked` callee. The IR carries both attributes and the
   inline happens anyway: `attributes #9 = { naked noinline nounwind returns_twice ... }`.
4. A `naked` function's body is inline asm with no operand constraints. It addresses its
   argument as `a0` because that is where the ABI puts it. Inlined, nothing puts `&c_jmp`
   into `a0`, so the fourteen `stc`s write 224 bytes through whatever `a0` happened to hold.

The image proves it: the save sequence appears **three** times -- once as the real `setjmp`,
twice spliced into callers. At both splice sites `a0` holds a `mrb_callinfo *`:

```
37d4c: ldc a0, 0x10(s1)   mrb->c
37d50: ldc a0, 0x30(a0)   c->ci
37d5c: stc s2, 0x10(a0)   ci->proc = begin_proc
37d60: stc a1, 0x40(a0)   ci->pc   = irep->iseq
37d64: stc ra, 0x0(a0)    setjmp's body, a0 still ci
   ...
37d98: stc sp, 0xd0(a0)   224 bytes into a 96-byte callinfo
```

That single mechanism produces both faults:

* **Fault 2** is the save itself running off the end of whatever `a0` addresses -- 160 bytes
  in the double-ldc image (`irep->iseq`), 96 in the plain one (`mrb_callinfo`). The buffer was
  never too small; the pointer was never the buffer.
* **Fault 1** is its wreckage. `stc s2, 0x30(a0)` puts `s2` -- `begin_proc`, a `struct RProc *`
  -- into `ci->stack`. `sizeof(struct RProc)` is **80**, which is the capability the stack
  clear then faults on. The 80 bytes were never an allocator bound and never a bad load: they
  are an RProc pointer written into the stack slot by a register save that thought it was
  writing a jmp_buf.

So the earlier reading has to go in full: the probe was right that the frame is healthy when
it looks, the clear was right to reload, and `-capstone-double-ldc` never repaired a load --
it moved the addresses enough to change which fault came first.

**Fix.** `port/capstone_setjmp.c` gets its own translation unit; separate objects cannot be
inlined across without LTO. It owns no globals, so the gp-captable single-owner gate is
unaffected. The build now counts the save sequence in the linked image and fails unless there
is exactly one copy -- positive control: before the split it printed 3.

**Still open, and it is a compiler question, not an mruby one:** `isInlineViable` should
refuse a `naked` callee outright. Inlining one is always wrong, because its asm binds its
operands by ABI register and inlining is exactly what removes the ABI. The port-level split
fixes mruby; the next port that puts `flatten` or `always_inline` near naked asm gets the same
silent miscompile.
