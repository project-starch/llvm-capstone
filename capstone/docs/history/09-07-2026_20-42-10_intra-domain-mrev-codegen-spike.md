# Codegen spike: can domain C mint + revoke a capability intra-domain?

*2026-07-09, B-lane (task `agentB-005`). Answers the one narrow codegen unknown
left open by A's firmware spike
(`history/09-07-2026_18-10-43_option-b-intra-domain-revoke-firmware-spike.md`).
Status: **all three questions answered; the Option B mechanism is a normal probe
build, but three codegen defects were found, one of which changes how Option B
must be written.** No compiler or emulator code changed.*

## The question

A's spike established that the Option B "gold standard" (literal single-domain
BORROW-REVOKE) needs none of the gated `start.S` firmware cycle, *provided* one
codegen assumption holds:

> When domain C does
> `R = __builtin_capstone_cap_mrev(arena); … cap_revoke(R); v = *alias;`,
> does the Capstone clang route the `*alias` deref through the held capability
> (so revoke faults it), or re-materialise an ambient pointer (so the revoke is
> silently missed)?

Task 003 proved the fault at the **instruction** level. This spike asks whether
**compiled domain C at `-O2`** preserves it.

## Answers

1. **Deref-through-held-cap after revoke: FAULTS.** Not missed. Verified on
   three independent paths — a register-held alias, an alias passed across the C
   ABI into a non-inlined callee, and an alias spilled to memory and reloaded.
   The `-O2` asm confirms the mechanism directly: `.insn` mints the cap into
   `a1`, and the post-revoke load is `lbu a1, 8(a1)` — still based on the held
   capability, no `auipc`/`gp` re-materialisation of the `arena` symbol.

2. **`MREV` RETAINS its source.** `helper_csmrev` copies `rs1` into `rd`,
   retypes **`rd`** to `CAP_TYPE_REV`, and gives it a fresh node via
   `cap_rev_tree_mrev` (which splices a node in *before* the source's and
   increments the source's `depth`). `rs1` is never nulled — `MREV` is one of
   the few cap-moving ops that does not consume its source. So **one linear
   grant mints many** revocation handles, nested: each `MREV` inserts another
   senior node, and revoking any handle invalidates the source.

3. **A linear sub-cap IS `MREV`-able — via `SPLIT`, not `SHRINK`.** `cssplit`
   gives the upper half a *fresh* node at the same depth as the lower half, so
   `mrev(hi)` + `revoke` walks only `hi`'s deeper run and stops before `lo`.
   Independence is structural. `SHRINK`/`SHRINKTO` copy `rev_node_id` unchanged,
   so a shrunk sub-cap shares the arena's node and cannot be revoked
   independently of it. **One arena can protect several sub-buffers
   independently, if carved with `SPLIT`.**

## Three codegen defects found

### C1 — `fastcc` + capability argument ICEs the compiler

Every pointer is `ptr addrspace(200)`, lowered as `MVT::i128`. `CC_Capstone`
(`CapstoneCallingConv.cpp:432`) has an explicit `ValVT == MVT::i128` case giving
a capability one argument GPR or a 16-byte stack slot. **`CC_Capstone_FastCC`
(`:639`) has no such case**: it falls through to `return true` ("CC didn't
match"), which `analyzeOutputArgs`/`analyzeInputArgs`
(`CapstoneISelLowering.cpp:23820`) turn into `llvm_unreachable(nullptr)`.

GlobalOpt promotes internal-linkage functions with all callers known to
`CallingConv::Fast` at `-O1`+. **So any domain C translation unit with a
`static` function that takes or returns a pointer and is not inlined fails to
compile at `-O1`/`-O2`.** `-O0` compiles; external linkage compiles. It has gone
unnoticed because existing probes are small enough that everything inlines.

Isolated by taking the `-O1` IR, deleting the two `fastcc` tokens, and running
`llc`: it lowers cleanly. The calling convention is the only variable.

**This matters for A**: SQLite domain C is full of non-inlined static functions
taking pointers. The Option B *mechanism* does not need this fixed; the SQLite
*integration* does.

Fix (~10 lines, B's lane): copy the `MVT::i128` block from `CC_Capstone` into
`CC_Capstone_FastCC`, allocating from `getFastCCArgGPRs(ABI)`. Not done — it
changes the shared LLVM tree that A also builds, and this task was scoped to
report. Repro committed as `fastcc_cap_arg_repro.c`.

### C2 — `cap_mrev` is marked pure but mutates the revocation tree

`BuiltinsCapstone.td:188` puts `cap_mrev` under `Attributes = [NoThrow, Const]`;
`IntrinsicsCapstone.td` marks `int_capstone_cap_mrev` `[IntrNoMem]`. But
`helper_csmrev` allocates a node in `env->cr_tree` and increments the source
node's depth — global side effects. Observed at `-O2`:

- an `MREV` whose result is unused is **dead-code-eliminated** (0 instructions);
- two `MREV`s of the same SSA value are **CSE'd into one**, though each must
  produce a distinct revocation node at a distinct depth.

`Const`/`IntrNoMem` further permits hoisting an `MREV` out of a loop or moving
it across a `REVOKE`. `cap_delin` has the same defect: it is `Const`, yet
`cap_rev_tree_delin` clears the node's `linear` flag, which decides whether a
later `REVOKE` yields `LIN` (data retained) or `UNINIT`.

Fix: drop `Const` from `cap_mrev`/`cap_delin` and give the intrinsics
`IntrHasSideEffects`, as `cap_drop`/`cap_revoke` already have.

### C3 — passing a LINEAR capability by value silently consumes it

`captype_is_copyable` (`cap.h:122`) is `type == CAP_TYPE_NONLIN`. Every
capability-moving instruction — `movc`, `cincoffset`, `cincoffsetimm`, `csscc`,
`csseal`, `csccsrrw` — NULLs its **source register** when the source is not
`NONLIN` and `rd != rs1`. That is correct linear-capability semantics.

C has no notion of linearity. `touch(buf)` is an ordinary pointer copy; clang
lowers it to `movc a0, s2`, which nulls `s2`. The first call works, the second
faults with cause 24 on an untagged null. **Nothing in the C source says the
first call destroyed `buf`, and there is no diagnostic.**

The memory path is asymmetric: `helper_compress_cap` does not null its source,
so `stc` duplicates even a `LINEAR` cap.

**Consequence for Option B — the one that changes the recipe:** a domain that
wants to pass its arena alias around in ordinary compiled C must **`delin` the
working alias first**. That is safe: `cap_rev_tree_delin` only clears the node's
`linear` flag, so the alias stays revocable, and the owner's `REVOKE` then
returns a `LIN` handle (data retained) rather than `UNINIT` — which is exactly
what an arena-reuse loop wants. This is the same reason `start.S` delinearises
`sp`/`gp`, generalised to every linear cap a domain holds.

The Option B sequence is therefore:

```
arena  = <linear grant>              // monitor, or csdebuggencap in a probe
R      = mrev(arena)                 // owner keeps R; arena is RETAINED (finding 2)
alias  = delin(arena)                // copyable, still revocable (finding C3)
   … domain uses alias freely, passes it to functions, spills it …
revoke(R)                            // lifecycle point; R comes back as LIN
   … any surviving alias now faults: cause 25 held / cause 24 reloaded …
```

## The provenance rule, restated at C level

`mrev_ambient_miss.c` revokes the arena and then touches the *same bytes* via
the ambient symbol. It does **not** fault: the domain reaches `.bss` through its
own `gp`/`pcc`-derived authority, whose revocation node is not a descendant of
the `gencap` node. `capstone_cap_revoked` only flags caps whose *tracked
lineage* was revoked.

**The revocable arena must be reachable only through the tracked capability.** A
buffer that is also an ordinary domain global has a second, un-revocable path to
the same memory. Same constraint as task 004, now demonstrated in compiled C.

## Reproduction

`capstone/capstone-qemu/tests/capstone-mrev-codegen/run-mrev-codegen-probes.sh`
— nine probes, one boot each (a faulted domain poisons later domain creation).
Firmware-free: the linear cap is minted with `csdebuggencap`. GREEN 9/9 on
submodule `fd4bc0c0`.

| Probe | Asserts | Outcome |
|---|---|---|
| `mrev_held_reg` | register-held alias | fault, cause 25 |
| `mrev_call_alias` | `delin`'d alias across the C ABI | fault, cause 25 |
| `mrev_mem_alias` | alias spilled to memory, reloaded | fault, cause 24 |
| `mrev_mem_control` | same, no revoke — isolates the above | `0x2227005E` |
| `linear_move_consumed` | LINEAR cap passed by value, no revoke (C3) | fault, cause 24 |
| `mrev_ambient_miss` | ambient deref of the same bytes after revoke | `0x22230041` |
| `mrev_src_retained` | source usable after `MREV`, `MREV`-able again | `0x22240011` |
| `mrev_subcap_lo_ok` | sibling of a revoked `SPLIT` half survives | `0x22250044` |
| `mrev_subcap_hi_fault` | the revoked `SPLIT` half itself dies | fault, cause 25 |

### Harness lesson: assert the *cause*, not just "a fault"

The first `mrev_call_alias` draft passed a LINEAR cap by value and faulted — but
with cause **24**, not the expected **25**. The fault was real and had nothing to
do with the revoke: `movc` had consumed the cap on the first call (C3). A driver
that only greps for "domain halted by capability fault" would have scored this a
pass and reported a validated Option B mechanism on the strength of a fault
caused by an unrelated compiler defect.

Cause 25 means the tag was **intact** and the revoked rev-tree node stopped the
deref. Cause 24 means the tag was **gone** — a reload of a revoked cap, *or* a
consumed linear cap. Only cause 25 is self-proving; every cause-24 expectation
needs a no-revoke control (`mrev_mem_control`, `linear_move_consumed`). The
driver now asserts the exact cause and fails on a mismatch.

## Bottom line for the A/B decision

**The Option B mechanism is a normal probe build — no firmware, no codegen fix
required**, provided the domain `delin`s its working alias (C3) and the arena is
not reachable ambiently. A can proceed with the single-domain probe.

The **SQLite integration** is a different matter: C1 blocks compiling any
realistic domain C at `-O1`/`-O2`, and C2 makes `MREV` unsafe under the
optimizer. Both are contained, both are B's lane, neither is on the critical
path for the mechanism proof.

Not addressed (out of scope): SQLite's `memsys5` arena being linear-backed —
still a separate, contained heap-backing question, and still not the `start.S`
ABI change. Subobject-bounds increment 2 remains lead-gated.
