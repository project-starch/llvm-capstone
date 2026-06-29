# Threat model: manufacturing capabilities, integer/capability confusion, and bounds

*Status: analysis for the paper's threat model. Grounded in what our Capstone
LLVM fork actually emits (probed 2026-06-24), not in what the architecture
promises in principle. Not yet a committed design decision — it is the input to
one.*

> **Update (2026-06-27) — T3 is now substantially mitigated.** Since this doc was
> written, object-granularity bounds narrowing has been **implemented and
> validated**. The compiler now emits **`SHRINK`** (the real Capstone op —
> *not* CHERI's `CSetBounds`/`scbnds`; `SPLIT`/`SHRINKTO` exist in the ISA but are
> unwired) at object materialization: **globals** (`CapstoneISelDAGToDAG.cpp`
> `selectLGA`, `-capstone-shrink-globals`, **default on**), **heap**
> (`rv8_malloc.c` + dtoa, `cap_shrink`, default on), **stack**
> (`ISD::FrameIndex`, `-capstone-shrink-stack`, **gated off** — whole-object
> only). Measured reality also corrects two claims below: un-narrowed bounds are
> **segment-granular**, and the domain image is a single `PT_LOAD`, so the
> inherited bound ≈ the **whole image** (cross-segment over-reads already trapped);
> precision after narrowing is byte-exact < 4 KiB, power-of-two grain above
> (`capability-bounds-model.md`). Evidence: the runtime authority suite
> `../../tests/capstone-authority/` (20/20; `global_oob`/`heap_oob`/`stack_oob`
> now **bounds-fault**), lit `cap-shrink-{globals,stack}.ll`, and a **real OOB
> write found** in rijndael (`char r[4]` written as 8 bytes). The residual T3 gap
> is now **subobject** bounds, stack **varargs/dynamic-alloca**, and
> **inter-procedural** provenance — see the inline "→ now:" notes below.

This document answers the questions raised after the reviewer's note on
"manufacturing capabilities out of thin air" and integer↔capability confusion:
what the concern means, how it maps onto **our** compiler/runtime, which threats
are real for us, what is already mitigated, what is not, and what to do.

---

## 1. What the reviewer actually means (restated precisely)

The claim is **not** "capabilities/CHERI are useless." It is a warning about the
**gap between a theorem and an implementation**:

> If you prove memory safety under the assumption *"integers and capabilities are
> never mixed"*, but the real system mixes them (by forging, by stripping tags,
> or by handing out capabilities that are far wider than the object they name),
> then the proof guards an assumption the implementation violates, and offers
> little real security.

There are three distinct ways the assumption gets violated in practice. It is
worth separating them because **they have different defenses and our system sits
differently against each one:**

1. **Forging a capability from an integer (provenance violation).** Code takes an
   attacker-controlled integer and gets a *usable* (tagged) pointer out of it —
   directly (a bad `inttoptr`), or by punning bytes (union, `memcpy`), or by a
   compiler bug that re-tags integer-derived bits.

2. **Demoting a capability to an integer and silently re-promoting it.** A real
   capability is moved through an integer-typed path (an integer register, an
   `i64` ABI slot, an integer store) that drops the tag, and then the value is
   used as a capability again. If the tag is *re-attached* by deriving from an
   ambient root, the demotion→promotion round-trip has effectively forged a
   capability. If the tag is *not* re-attached, the program faults (fail-safe) —
   but a fault on a load-bearing path is still a correctness/availability bug, and
   the dangerous case is the one where it does **not** fault.

3. **Over-broad capability + attacker-controlled offset ("base + offset" with no
   bounds check).** The capability is genuine and tagged, the hardware *does*
   check its bounds — but the bounds cover much more than the intended object, so
   an attacker-controlled index or offset stays *in bounds* yet reaches a
   *different* object. The tag check passes; the isolation is gone. This is the
   case in the V8/WASM sandbox-escape examples the reviewer linked: the runtime
   holds a legitimate, powerful pointer in a slot the attacker can influence, and
   a **logical** (type-confusion) bug supplies an attacker-chosen offset to it.
   Hardware tags do nothing here, because nothing is forged.

The interpreter/runtime scenario the reviewer described is **(3) layered on a
type-confusion bug**: security metadata (a type tag) outside the cage gates access
to real pointers; a logical type-confusion error lets attacker data be read as a
pointer, or lets attacker data choose the offset applied to a real pointer.

**Key consequence for us:** memory-safety claims must be stated against a threat
model that explicitly covers (1), (2), and (3). Our headline guarantee — "every
pointer is a capability, so spatial/temporal safety is enforced in hardware" —
is only as strong as (a) our refusal to forge, (b) our discipline about
tag-stripping integer paths, and (c) **how tightly we bound capabilities.** We
measured all three.

---

## 2. The three layers, and where each guarantee must live

| Layer | Guarantee it owns | In our project |
|-------|-------------------|----------------|
| **Architecture** (Capstone HW / QEMU) | Tags are unforgeable in unprivileged code; capability-producing ops are monotonic (cannot grow bounds/perms); bounds & perms are checked on every access; LINEAR capabilities are unique. | The CHERI-like ISA we target. We *rely* on it; we do not implement it. |
| **Compiler** (this LLVM fork) | Lower pointer ops to **tag-preserving** instructions; lower integer ops to instructions that **do not** fabricate tags; **bound** capabilities to the objects they name. | **This is our contribution and our risk surface.** A compiler bug here silently breaks the architectural guarantee. |
| **Workload** (BEEBS/RV8 today; SQLite, language runtimes, WASM-like later) | Do not place powerful capabilities in attacker-reachable slots; keep type metadata sound; do not feed attacker offsets into pointer arithmetic. | Out of our hands for third-party code, but we choose what we run and how we compile it, and the over-broad-bounds problem (3) determines how badly a workload bug escalates. |

The reviewer's concern is precisely that **a memory-safety theorem is usually
proved at the architecture layer, but threats (1)–(3) are introduced at the
compiler and workload layers.** Our paper's defensible contribution is the
**compiler layer**: showing (and where necessary, enforcing) that our lowering
does not manufacture, does not carelessly demote, and bounds tightly.

---

## 3. Threats, mapped to our compiler — with evidence

The following is what our backend emits today (`-O1`, `capstone64`,
representative probes; reproduce with the snippets in §7).

### T1 — Forging a capability from an integer  →  **mitigated (architecture + compiler does not forge)**

`int *forge(unsigned long x){ return (int*)x; }` compiles to essentially a
register move: the integer `x` is returned in the pointer position **with no
tag-setting instruction** (no `cincoffset` from a valid root). The result is
**untagged** (tag = 0). Dereferencing it faults. There is no unprivileged
instruction that sets a tag from arbitrary bits, so neither C-level `inttoptr`,
union punning, nor `memcpy` of pointer bytes (which moves bytes, not the
out-of-band tag) can produce a usable capability. **(1) is closed by
construction**, and our compiler does not undermine it.

> Caveat to verify for the paper: confirm that *byte-wise* copies of a capability
> (`memcpy`/`memmove` over a `void*` buffer) move data **without** the tag (so the
> copy is untagged), i.e. that our `memcpy` uses integer (`ld`/`sd`) not
> capability (`ldc`/`stc`) moves. Our freestanding `memcpy` is byte-oriented, so
> it strips tags — which is the *safe* behavior for forging, but note it also
> means **capabilities cannot be copied through `memcpy`** (a correctness corner
> for any workload that `memcpy`s a struct containing pointers).

### T2 — Demote-to-integer then re-promote (tag strip via integer ops)  →  **mostly mitigated, but this is the live compiler-bug class**

The whole point of the `cincoffset` vs `addi` split is this threat:

* **`cincoffset` / `cincoffsetimm`** = capability-aware pointer arithmetic;
  **preserves the tag** (subject to representability).
* **`addi` / `ISD::ADD` on the address** = integer arithmetic; the result is a
  plain integer, **tag stripped**.

If the compiler lowers a *pointer* operation with the integer instruction, the
capability is silently demoted to an integer. Whether that is fail-safe or
exploitable depends on what happens next: a deref of the demoted value faults
(safe); a *re-derivation from an ambient root* (see T3) would re-tag it and
forge.

**We have a real, fixed case study of exactly this.** Earlier this session,
stack-passed capability arguments (the 9th+ pointer argument, spilled to the
stack) were "delivered untagged": `LowerCall` computed the outgoing stack-slot
address with `ISD::ADD` (an `i64` integer add) instead of `CIncOffset` from the
stack capability, so the store target was an integer-typed address and the
capability argument lost its tag in transit (commit "fix tag loss for
stack-passed capability arguments"). This is **the reviewer's threat (2) as an
actual compiler defect** — and it shows the risk is not hypothetical: it is one
mis-typed DAG node away, and it had escaped notice because the common (≤8-arg)
path was correct.

**Implication for the paper:** "we never mix integers and capabilities" is a
property of the *lowering*, and lowering bugs violate it. We should not assert it
— we should **audit and test for it** (§7). The existence of the stack-arg bug is
itself evidence for the reviewer's point and a good motivating example.

### T3 — Over-broad bounds + attacker offset ("base + offset")  →  **was the central gap; now mitigated for globals + heap (default on), stack opt-in**

This was the most important finding *at the time of writing*. The analysis below
describes the **pre-narrowing** behavior. **→ now:** the compiler emits `SHRINK`
to narrow each materialized object to its bounds (globals + heap by default,
stack via `-capstone-shrink-stack`); the un-narrowed picture below is reproducible
with `-mllvm -capstone-shrink-globals=false`. The original text:

**[pre-narrowing] Our compiler never narrowed capability
bounds.** The backend emitted no bounds-narrowing op for object addresses; every
pointer inherited the bounds of the root it was derived from:

```
take_global(i):   &g[i]          g is int[64]
    auipc a1, %pcrel_hi(g); addi a1, a1, %pcrel_lo
    cincoffset a1, gp, a1      ; a1 = capability to g, bounds INHERITED FROM gp
    delin a1
    cincoffset a0, a1, a0      ; + i*4  — NO bounds check against g's 256 bytes

take_stack(i):    &loc[i]        loc is int[64] on the stack
    cincoffsetimm a1, s0, -288 ; frame slot, bounds INHERITED FROM the stack cap
    cincoffset    a0, a1, a0   ; + i  — NO bounds check against loc

take_heap(i):     &h[i], h = malloc(256)
    cincoffset a0, a0, s1      ; + i on the malloc'd pointer (bounds = whatever
                               ;   malloc returned — our bump allocator returns
                               ;   gp-derived, whole-arena/whole-domain caps)
```

`gp` is an **ambient, near-omnipotent root**: `cincoffset(gp, X)` yields a valid
tagged capability whose address is `X` for any `X` spanning at least the whole
domain image (this is *why* our globals/function-pointers/`__capstone_cap_init`
table all work by deriving from `gp`). Consequently:

* `&g[i]` for attacker-controlled `i` was [pre-narrowing] a **genuine, tagged**
  capability that could address **anything in the segment**, not just `g`. The
  bounds check passed because the bounds were the whole segment. **Inter-object
  spatial safety inside a domain was not enforced.** (**→ now:** `&g` is `SHRINK`'d
  to `[g, g+sizeof(g))`, so `&g[i]` leaving `g` faults.)
* The same held for stack objects (bounded to the whole stack/frame region) and
  heap objects (bounded to the allocator arena) — both now narrowed (stack via
  the opt-in flag).

This was **exactly threat (3)** in the pre-narrowing output. Our scheme gives:

* **Inter-domain** isolation — enforced by the monitor / the `/dev/capstone`
  loading model bounding each domain's root (and page tables/PMP). *(Strong, but
  owned by the architecture layer, not us.)*
* **Provenance / unforgeability** — T1/T2 above. *(Compiler-owned, in good shape
  modulo audit.)*
* **Intra-domain, object-granularity spatial safety** — **[pre-narrowing] absent.**
  A buffer overflow or attacker-controlled index within a domain was *not* caught
  by capabilities, because the capability covered far more than the buffer.
  **→ now: enforced for globals and heap (default) and stack (opt-in)** — an
  over-read leaving the object faults (authority `global_oob`/`heap_oob`/`stack_oob`).
  Note the un-narrowed bound was **segment-granular** (the image is a single
  `PT_LOAD`, ≈ whole image), so a cross-*segment* over-read already trapped; the
  gap was cross-*object within the segment*, which narrowing now closes.

So a memory-safety claim of the form "every access is capability-checked,
therefore object bounds are enforced" **was false for the pre-narrowing output**
and is **now true for narrowed objects** (globals/heap default, stack opt-in),
modulo the residual gap (subobject, varargs/dynamic-alloca on stack,
inter-procedural). We enforce that accesses go *through* capabilities, and now
also that compiler-materialized object capabilities are *tight*. Remaining work
is stated precisely in §6–§7.

### T4 — Real capability in an attacker-controlled slot + type confusion (the interpreter/WASM case)  →  **future workloads; severity governed by T3**

Today's workloads (BEEBS, RV8, CoreMark) are not interpreters and do not store
capabilities in attacker-reachable, type-tagged slots, so the V8/WASM scenario
does not yet apply. But the roadmap (SQLite, then "real software," plausibly
language runtimes) walks straight into it. Two sub-cases:

* **Attacker data read as a pointer (no real cap involved):** if the runtime
  reads an attacker-controlled slot as a pointer, it gets an **untagged** value →
  fault (T1 protects us). *Hardware tags help here, contra a naïve reading of the
  reviewer's note — provided the slot really held attacker data and not a real
  capability.*
* **Real cap in the slot, attacker controls the offset:** the runtime stores a
  genuine pointer and a type-confusion bug lets the attacker pick the offset
  applied to it. **Tags do not help; bounds are the only defense — and per T3 our
  bounds are domain-wide.** This is the dangerous case, and it is dangerous *for
  us specifically* until T3 is addressed.

The defense the reviewer implicitly endorses (and WASM uses) — keep real
capabilities in a trusted region *outside* the attacker cage — is a **workload/
runtime** obligation. Our **compiler** obligation is to make sure that when the
runtime does hold a pointer, that pointer is bounded to its object, so an
attacker-chosen offset cannot wander.

### T5 — Control-flow integrity (function pointers)  →  **not addressed**

Function pointers are materialized the same way (`cincoffset gp, <fn addr>`),
inheriting `gp`'s executable extent. There is no **sealing** of code pointers and
no CFI. An attacker who controls a function-pointer-typed slot's *offset* (T3 in
the control-flow dimension) can redirect a call to any executable address in the
domain. Sealed/sentry capabilities (a CHERI/Capstone feature) are the standard
defense and we use none today.

### T6 — Tag in the static image  →  **mitigated (and the reason the cap-globals work exists)**

A capability tag cannot live in a static ELF image, so initialized capability
globals load **untagged**. We do **not** "re-forge" them from the stored bits;
we re-materialize them at runtime by deriving from `gp` (`__capstone_cap_init`
constructor + the PC-relative `.capstone_cap_init` table; see
`capability-globals-init-decision.md`). This is provenance-correct (derives from
a valid root, does not fabricate a tag). **T3 caveat [pre-narrowing]:** those
re-materialized globals also inherited `gp`'s wide bounds — tagged and genuine,
but not bounded to the object. **→ now:** the cap-globals path also narrows —
both the `@tab` store base and the stored element pointers are `SHRINK`'d to their
objects (see `static-cap-global-init.ll`), so T6 no longer inherits T3.

---

## 4. Summary table

| # | Threat | Arises from | Impacts us? | Status |
|---|--------|-------------|-------------|--------|
| T1 | Forge cap from integer | `inttoptr`, byte punning, `memcpy` | No — produces untagged, faults on use | **Mitigated** (HW + compiler doesn't forge) |
| T2 | Demote→re-promote (tag strip via integer ops) | Mis-typed lowering (`addi`/`ISD::ADD` on a pointer) | Yes — *was* a live bug (stack-arg) | **Mostly mitigated; audit/test needed** |
| T3 | Over-broad bounds + attacker offset | Was: everything derived from ambient `gp`/stack root, no narrowing | Was the central gap | **Now: `SHRINK` narrows globals+heap (default) / stack (opt-in); residual = subobject, varargs/alloca, inter-procedural** |
| T4 | Real cap in attacker slot + type confusion | Interpreters / language runtimes we will run | Future; severity set by T3 | **Open (workload + T3)** |
| T5 | Control-flow redirection via cap offset | Unsealed function pointers, no CFI | Yes | **Not addressed** |
| T6 | Tag in static image | Capability globals in ELF | No (re-materialized from `gp`) | **Mitigated** (inherits T3) |

---

## 5. What is genuinely already mitigated (and defensible in the paper)

1. **Unforgeability of tags** — architectural, and our compiler never emits a
   tag-from-integer path (T1, evidenced).
2. **Provenance discipline in pointer arithmetic** — the `cincoffset` vs `addi`
   separation is real and is the mechanism that *can* keep (2) honest; the
   recently fixed stack-arg bug shows we are actively maintaining it.
3. **No tag smuggling through the static image** — capability globals are
   re-derived from a live root, not reconstructed from stored bits (T6).
4. **Inter-domain isolation** — each domain runs under a monitor-bounded root;
   one domain cannot fabricate a capability into another (relies on the
   architecture + loader, not on our codegen).
5. **Fail-safe on misinterpretation** — reading attacker/integer data as a
   pointer yields an untagged value that faults rather than silently aliasing
   (the *helpful* half of the WASM scenario).

## 6. What is NOT mitigated — risks to put on the record

1. **(Was highest) Object-granularity bounds (T3).** **→ now largely addressed:**
   globals + heap are `SHRINK`-narrowed by default and stack opt-in, so intra-domain
   overflows and attacker indices on those objects **are** caught. Residual: stack
   off by default (whole-object spike), **subobject** bounds, stack
   **varargs/dynamic-alloca**, and **inter-procedural** provenance. The
   memory-safety claim can now include object-granularity spatial safety for
   compiler-materialized globals/heap (and stack with the flag), scoped by that
   residual — no longer only "inter-domain + provenance."
2. **Lowering-bug fragility (T2).** Provenance correctness rests on every pointer
   operation being lowered to the tag-preserving instruction. We have no
   systematic check; we found one violation by accident. There are surely more
   corners (varargs, `memcpy` of structs-with-pointers, `setjmp`, atomics,
   inline asm, `va_arg`, bitfield/union pointer storage).
3. **No sealing / CFI (T5).** Control-flow targets are as wide as `gp`'s
   executable extent.
4. **`gp` is a near-omnipotent ambient root.** Convenient for codegen; "derived
   from `gp`" carries no object-level authority *until narrowed* (now done at
   materialization). Measured: the un-narrowed bound is **segment-granular**, and
   the domain image is a single `PT_LOAD`, so it ≈ the whole image (confirmed via
   the `Cap mem access OOB` bounds reported by the authority suite — not "wider
   than the domain"). Still worth a monitor/loader cross-check for the inter-domain
   argument.
5. **[Fixed] Allocator returned wide capabilities.** **→ now:** the bump `malloc`
   (`rv8_malloc.c`) and dtoa's `malloc_beebs` `cap_shrink` each allocation to the
   requested size. (trio's `realloc_beebs` is left un-narrowed — it deliberately
   over-reads the old allocation; a documented latent over-read.)
6. **Sub-object safety** (struct fields) is out of reach even with per-object
   bounds — a known CHERI limitation worth stating, not solving.

---

## 7. Concrete steps

### For the system (compiler/runtime work)

1. **Add bounds narrowing (the big one). → DONE (globals + heap; stack opt-in).**
   The backend emits **`SHRINK`** (rounded to representable-bounds granularity by
   the hardware; byte-exact < 4 KiB) where objects are materialized:
   * **Globals:** narrow the `cincoffset gp, &g` result to `sizeof(g)`
     (`CapstoneISelDAGToDAG.cpp` `selectLGA`; `-capstone-shrink-globals`, default
     on). ✓
   * **Stack:** narrow address-taken whole locals to the object size
     (`ISD::FrameIndex`; `-capstone-shrink-stack`, **default off** — whole-object
     only, not interior/varargs/dynamic-alloca yet). ◑
   * **Heap:** `cap_shrink` in `malloc` (`rv8_malloc.c`, dtoa) to the requested
     size. ✓
   The correctness-fallout + cost experiment was run: **CoreMark ✓, RV8 7/7 ✓,
   BEEBS 82/82 ✓** with narrowing on, and it **found a real OOB write** (rijndael
   `char r[4]` written 8 bytes — now patched). Evidence/regression:
   `../../tests/capstone-authority/`, `cap-shrink-{globals,stack}.ll`. **Remaining:**
   stack default-on (subobject/varargs/alloca), and the runtime gap that a
   domain-mode fault currently aborts the QEMU model.
2. **Provenance audit + a "capability hygiene" test suite (closes T2 honestly).**
   Enumerate every lowering that touches a capability-typed value and assert it
   uses tag-preserving ops. Build a negative test battery that *attempts* to
   manufacture/smuggle a capability through every C construct — `inttoptr`, union
   pun, `memcpy` of a pointer, `va_arg`, `longjmp`, struct return, >8 pointer args
   (the fixed bug), atomics, inline asm — and assert each yields **untagged**
   (faults) or is correctly preserved. This battery is reusable as a regression
   gate and as a paper artifact ("we systematically tested the no-forging
   property").
3. **Confirm and document `gp`'s bounds** against the monitor/loader. Establish
   the inter-domain isolation argument rigorously (what bounds the root, who sets
   it, can it be widened).
4. **Sealing / CFI for code pointers (T5)** — design item; sentry capabilities
   for function pointers and return addresses.
5. **Exploit linearity as a lever (Capstone-specific).** LINEAR capabilities
   (and `delin`) give *unique ownership*: a linear capability cannot be
   duplicated into an attacker-reachable slot while also being held in a trusted
   region. This is a Capstone feature beyond baseline CHERI and a natural answer
   to T4 (keeping a runtime's real pointers un-copyable). Worth a design pass and
   likely a paper differentiator.

### For the paper

6. **State the threat model in the (1)/(2)/(3) taxonomy** from §1 explicitly, and
   for each, say what the architecture guarantees, what the compiler guarantees,
   and what is assumed of the workload. This directly answers the reviewer:
   the proof's "no integer/capability mixing" assumption is **discharged by the
   compiler**, so the paper must show *how*.
7. **Use the stack-arg bug as a motivating example** of threat (2) — a concrete,
   found-and-fixed provenance defect that justifies systematic testing over
   "by inspection" claims.
8. **Scope the safety claim precisely.** Until step 1 lands: claim provenance +
   inter-domain isolation, *not* intra-domain spatial safety. After step 1:
   claim object-granularity spatial safety with measured overhead. Do not
   overclaim — that is the whole content of the reviewer's warning.
9. **Add an interpreter/runtime case study** (even a minimal one) to engage T4
   directly: show that (a) reading attacker data as a pointer faults (tags help),
   and (b) only tight bounds (step 1) prevent the "real cap + attacker offset"
   escalation — quantifying the residual risk the reviewer is worried about.

---

## 8. One-paragraph answer to "could this impact our system?"

Yes — partially, and in a way that is precisely the reviewer's point. We are
**safe against forging** a capability from an integer (T1) and we maintain the
integer/capability lowering split that keeps that honest (T2 — though it is a
lowering-bug surface, as our fixed stack-argument defect proves). On the
"base + attacker-controlled offset" class (T3): **this is now mitigated for
compiler-materialized globals and heap objects (default on) and stack objects
(opt-in)** — the compiler emits `SHRINK` to bound each to its object, so an
attacker-chosen offset that leaves the object faults. (Pre-narrowing, every
pointer carried segment-wide bounds — the image is a single `PT_LOAD`, ≈ whole
image — derived from the ambient `gp`/stack root; that is reproducible with
`-capstone-shrink-globals=false`.) Hardware tag checks enforce *that accesses go
through capabilities* and *inter-domain* isolation; narrowing now adds
**object-level spatial safety inside a domain** for those materialization sites —
the protection an interpreter type-confusion exploit (T4) needs. The honest claim
today is "provenance + inter-domain isolation **+ object-granularity spatial
safety for narrowed globals/heap (and stack with the flag)**," with the residual
being subobject bounds, stack varargs/dynamic-alloca, and inter-procedural
provenance. The narrowing experiment also delivered the paper-worthy
overhead + bugs-found result (rijndael). The provenance test battery (step 2) is
realized as `../../tests/capstone-authority/`; the systematic provenance *verifier*
is proposed in `c2-provenance-verifier-proposal.md`.
