# Boundary Violation — CVE-2018-10191 (Row 11)

> **Read `target.md` first.** This row reproduces as a **spatial**
> heap-buffer-overflow, not a temporal use-after-free, and it is a within-runtime
> defect rather than a managed↔native FFI crossing. Both points are documented
> there. The annotation below is kept in the §8 shape for consistency, but the
> honest summary is: no pointer crosses the language boundary in this row.

### The object involved

`REnv::stack` — the register array of an mruby scope's environment. For a
stack-shared environment it points into the VM's register stack; for a closed one
it points at heap storage sized to that scope. The trigger reads
`e->stack[b]` where `b` was computed for a *different, wider* scope than the `e`
actually resolved.

### Owner vs. borrower

Both sides are the mruby C runtime. The compiler (`codegen.c`) decides `b` and
`lv`; the VM (`vm.c`) resolves `e` from `lv` and indexes with `b`. The Ruby script
only supplies nesting depth and local count — it holds no pointer and crosses no
FFI shim. There is no native-extension or foreign-language participant, which is
why this row does not fit the cross-language framing the other rows share.

### Fault site

`mrb_vm_exec`, `CASE(OP_GETUPVAR)` — `src/vm.c:1208`

```c
struct REnv *e = uvenv(mrb, c);     /* c truncated to 7 bits by codegen */
...
*regs_a = e->stack[b];              /* b is the *outer* scope's index */
```

ASan: `heap-buffer-overflow`, `READ of size 16`, 528 bytes past a 4096-byte
region allocated by `stack_extend_alloc` (`src/vm.c:203`).

### The rule that is violated

The scope level `lv` and the register index `b` must describe the *same* scope.
`codegen.c:2191` breaks that pairing by writing `lv` into a 7-bit field without a
range check, so beyond 128 levels of nesting the VM resolves an environment that
does not match the index it is about to apply — and `OP_GETUPVAR` indexes the
resolved environment with no bound against its actual register count.

Note the two independent failures: the *truncation* (a compiler bug — no
diagnostic for an out-of-range level) and the *unchecked index* (a VM bug — no
bound on `b` against `e`'s size). Either check alone would prevent the overflow.

### Note on the capability phase (not implemented here)

Unlike the temporal rows, revocation does not address this one — nothing is freed.
What would stop it is **bounds**: a capability for `e->stack` carrying the
environment's true register count would fault on the `b`-th access when `b`
exceeds it, regardless of which environment `lv` resolved to. That is a spatial
guarantee, which is why this row sits outside the temporal-borrow argument.
