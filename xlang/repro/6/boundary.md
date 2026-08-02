# Boundary Violation — CVE-2026-1979 / mruby #6701 (Row 6)

> **Read `target.md` first.** This row reproduces as a **spatial**
> heap-buffer-overflow *write*, not a temporal use-after-free, and the fault is
> internal to the mruby toolchain rather than a managed↔native FFI crossing. The
> annotation is kept in the §8 shape for consistency, but no pointer crosses a
> language boundary here.

### The object involved

The **instruction stream** (`codegen_scope::iseq`) of a compiled method, and then
the **VM register frame** that the corrupted instruction indexes. The compiler
writes one byte outside the instruction it intended to modify; the VM later trusts
that byte as a register number and stores through it.

### Owner vs. borrower

Both sides are the mruby C runtime, in two different phases:

- **The compiler** (`mrbgems/mruby-compiler/core/codegen.c`) owns `iseq` and
  performs the errant store during the peephole optimization.
- **The VM** (`src/vm.c`) is the borrower: it consumes `iseq` as trusted input and
  applies the corrupted operand without bounds-checking it against the frame's
  `nregs`.

The Ruby script only chooses the *shape* of the source (an undefined pin, in
statement position, after a 2-byte instruction). It holds no pointer and calls no
foreign function. There is no native-extension participant, which is why this row
does not fit the cross-language framing the other rows share.

### Corruption site

`codegen(...)`, the pattern-match optimization —
`mrbgems/mruby-compiler/core/codegen.c:6632`

```c
if ((int32_t)(fail_pos + 2) + (int16_t)PEEK_S(s->iseq+fail_pos) == 0 &&
    fail_pos + 2 == s->pc) {           /* <- no check of the actual opcode */
  s->iseq[fail_pos - 2] = OP_JMPIF;    /* assumes a 4-byte OP_JMPNOT */
```

With `NODE_PAT_PIN`'s undefined-variable path (`codegen.c:4524`) emitting a 3-byte
`OP_JMP`, `fail_pos - 2` lands in the preceding instruction.

### Fault site

`mrb_vm_exec` — `src/vm.c:1788`

The corrupted `LOADI_5 R38` executes in a frame with `nregs=4`. ASan:
`heap-buffer-overflow`, **`WRITE of size 8`**, 32 bytes past a 1024-byte region
allocated by the VM stack. Under RISC-V QEMU the same store reaches an unmapped
page and the process takes SIGSEGV.

See `bytecode-diff.txt` for the side-by-side disassembly against a fixed compiler.

### The rule that is violated

A peephole rewrite may only edit an instruction it has positively identified. The
optimization inferred the instruction's *position* from a fixed offset
(`fail_pos - 2`) that is only correct for one opcode width, and never checked
which opcode was actually there. The upstream fix adds exactly that check:

```c
s->iseq[fail_pos - 2] == OP_JMPNOT
```

Secondarily, the VM applies a register operand from `iseq` without bounding it
against the frame's `nregs` — so a single corrupted byte becomes an out-of-frame
store rather than a caught error.

### Note on the capability phase (not implemented here)

Revocation does not address this row — nothing is freed. **Bounds** do: a
capability for the register frame carrying its true `nregs` extent would fault on
the store to R38 regardless of how the operand became 38. That is a spatial
guarantee, which is why this row sits outside the temporal-borrow argument.

There is a second, arguably more interesting framing: the corrupted object is
*code*. A sealed-code capability over `iseq` would make the compiler's own late
rewrite impossible — but mruby legitimately rewrites `iseq` throughout codegen, so
the seal would have to be applied at the codegen→execute transition rather than at
allocation.
