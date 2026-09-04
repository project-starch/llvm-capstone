# Capability granularity, provenance, and the two paper contributions

*Status: analysis + answers to the questions raised in the review discussion. Every
claim about "what our compiler does today" is backed by an actual codegen probe
(target `capstone64-unknown-elf`, `-O1`, commit `dcc9c0cee120`); the probe
sources and disassembly are in §11. This is the design input for the two
proposed paper contributions, not yet a committed implementation plan.*

Companion documents:
- `capability-provenance-threat-model.md` — the T1–T6 threat taxonomy this builds on.
- `capability-bounds-model.md` — the measured bounds model (precision, `SHRINK`).
- `c2-provenance-verifier-proposal.md` — the proposed provenance verifier (C2).
- `syscalls-and-hostcall-abi.md` — the syscall/HostCall explainer for a collaborator.
- `../plans/capability-authority-audit.md` — the provenance-audit track skeleton.

> **Update (2026-06-27) — C1 granularity is now implemented.** This doc's
> "what our compiler does today" answers describe the **pre-narrowing** baseline
> (probe commit `dcc9c0cee120`). Since then, object-granularity bounds narrowing
> landed: the compiler emits **`SHRINK`** at materialization for **globals**
> (`-capstone-shrink-globals`, default on), **heap** (`malloc` `cap_shrink`,
> default on), and **stack** (`-capstone-shrink-stack`, gated off). So the "we
> never narrow / no per-object bounds / both run without
> faulting" answers below are **superseded** for narrowed objects — the array
> example now **traps**; see the per-row "→ now:" notes. Measured detail: bounds
> are **segment-granular** when un-narrowed (single `PT_LOAD` ≈ whole image), and
> `SPLIT`/`SHRINKTO` exist in the ISA (so "no splitting" was about the compiler,
> not the hardware). Evidence: `../../tests/capstone-authority/` (26 domains),
> `cap-shrink-{globals,stack,dynalloca}.ll`, `capability-bounds-model.md`. The
> "before" picture is reproducible with `-mllvm -capstone-shrink-globals=false`.
>
> **Update (2026-07-03) — stack narrowing coverage extended (still gated off).**
> The stack arm is no longer whole-object-only. It now narrows **interior pointers
> + load/store bases** through fixed frame objects (shared
> `narrowToFrameObjectBounds`, 2026-07-01), the **varargs save area** (via the
> fixed-object path), and **dynamic (runtime-sized) allocas** —
> `lowerDYNAMIC_STACKALLOC` shrinks the returned pointer to `[cursor, cursor+size)`
> while `sp`/X2 keeps broad bounds (2026-07-03; probes
> `stack_dynalloca_{inbounds,oob}`, lit `cap-shrink-dynalloca.ll`). So the
> per-row "whole-object only" caveats below are themselves now dated;
> `-capstone-shrink-stack` stays **default off** pending a clean default-on matrix
> (the one prior `-O0` regression, rijndael, was a genuine LP64 over-read the
> narrowing caught — now fixed).

---

## 0. TL;DR for each question

| # | review question | Short answer |
|---|-------------|--------------|
| 1 | Spilled capability readable by an attacker? | Spills use `stc` and stay *tagged* and usable. **Pre-narrowing:** any in-domain pointer could name any spill slot (no intra-domain bounds). **→ now:** object narrowing (C1) shrinks the blast radius — an attacker holding a pointer to object *X* can no longer name the spill slot for cap *Y* (their cap is bounded to *X*). Residual: caps spilled within a frame whose own bound still covers them; full mitigation also wants linearity / don't-spill policy. |
| 2 | Distinguish int vs cap; MAC/checksum; "disjoint" | We use a **hardware out-of-band tag bit**, not a MAC. That is *stronger* than a checksum (unforgeable, not probabilistic). A MAC only becomes relevant if caps must cross **untagged storage** (disk/network/persistence). |
| 3 | How is `ptr→int→ptr` compiled? | Through **integer** instructions (`mv`/`addi`); the result is **untagged → faults on deref** (fail-safe). Also: our `uintptr_t` is **64-bit**, narrower than the 128-bit cap — a cap cannot even round-trip through it. |
| 4 | `p - q` into an integer? | Extract both cursors (`lcc`), integer `sub`, then signed element scaling (`srai` for exact power-of-two factors). Pure integer result, no tag. **Fixed after the 2026-06-29 audit:** `low - high == -7` now passes; genuine logical shifts remain `srli`. |
| 5 | `malloc` now / in the suites / how to do it | No OS heap, and **no general libc allocator**. **Per-allocator, benchmark-local:** `rv8_malloc.c` and dtoa's `malloc_beebs` `cap_shrink` each allocation (dtoa to the 16-rounded size, not byte-exact); **trio's `realloc_beebs` is left un-narrowed** (it over-reads the old block); CoreMark uses stack storage. So this is *not* a compiler-wide "heap default-on" policy — it is two prototype allocators. |
| 6 | How are caps created / bounds assigned? | Derive from a root (`gp`/`sp`) via `cincoffset`. **Pre-narrowing:** bounds were inherited from the root (no per-object bounds). **→ now:** materialization narrows to the object with `SHRINK` (globals/heap default, stack opt-in). |
| 7 | Do we do capability splitting? | **Compiler:** pre-narrowing did the anti-pattern (root cap + move cursor); **→ now** it narrows each object to its bounds (`SHRINK`). **Hardware:** a real `SPLIT` (and `SHRINKTO`) instruction *does* exist in the ISA — just not wired into LLVM yet. So "no splitting" was a statement about the compiler, and is now outdated. |
| 8 | The `&a[10]; p+5; p+25; *q` example | **Pre-narrowing:** both ran without faulting (cap covered the whole segment). **→ now (default):** `p+25` leaving the object **traps** — `a`'s capability is `SHRINK`'d to its bounds (authority `global_oob`/`stack_oob`). `-capstone-shrink-globals=false` reproduces the old no-trap behavior. |
| 9 | The two contributions (granularity + provenance) | C1 (granularity): **initial slices implemented** (globals default-on; two benchmark allocators; stack opt-in), **functionally** validated across CoreMark/BEEBS/RV8, rijndael OOB found — but **overhead is NOT yet measured** and coverage is partial (no subobject, broad roots remain). C2 (provenance): primitives exist; the **proposed** verifier is a hygiene checker, not a proof (audit). The audit argues the distinctive Capstone angle is **attenuation + root-elimination** (linearity/`SPLIT`), not re-deriving CHERI bounds — a framing to agree with the reviewer. See §9–§10. |

---

## 1. If we spill a capability, can an attacker read it and gain authority?

**What the compiler does:** capabilities are spilled with the tag-preserving
capability store `stc` and reloaded with `ldc`. Every probe shows it, e.g. the
prologue `stc ra, 16(sp)` / epilogue `ldc ra, 16(sp)`. So a spilled capability in
memory is a **genuine, tagged, immediately usable capability** — the bytes in the
slot *are* a working pointer, with its tag intact in the out-of-band bit.

**So the answer depends entirely on who can read that slot:**

- **Another domain:** cannot. Each domain runs under a monitor-bounded root; it
  has no capability that covers another domain's stack/heap, so it cannot even
  name the slot, let alone load it. *Inter-domain isolation holds* (architecture +
  loader, not our codegen).
- **Code within the same domain:** **[pre-narrowing] could.** This was the
  uncomfortable answer: with no per-object bounds, every pointer was derived from
  `gp`/`sp` and covered essentially the whole segment, so an attacker with *any*
  pointer could compute a spill slot's address and `ldc` the spilled capability,
  tag and all. **→ now:** per-object narrowing (C1, §6–§8) bounds each pointer to
  its object, so a pointer to object *X* can no longer name the spill slot for cap
  *Y*. Residual: caps spilled into a frame whose own bound still covers them, and
  the stack-narrowing default-off — full intra-domain spill confidentiality also
  wants linearity / a don't-spill-sensitive-caps policy.

**Guarantees, stated honestly:**
- We guarantee a spilled cap is **not silently downgraded** to forgeable bytes (it
  stays a real cap — `stc`/`ldc`, not `sd`/`ld`).
- We guarantee **inter-domain** secrecy of spilled caps.
- We do **not** guarantee **intra-domain** secrecy. An in-domain memory-disclosure
  bug leaks usable capabilities.

**Implications / what to do.** This is precisely why the reviewer's concern matters and
why both contributions are needed:
1. **Bounds narrowing (C1)** shrinks the blast radius: if each cap is bounded to
   its object, an attacker with a pointer to buffer *X* still cannot name the spill
   slot for cap *Y*.
2. **Linearity (Capstone-specific lever).** A *linear* capability cannot be
   duplicated; if a sensitive cap is held linearly in a trusted region, it cannot
   simultaneously sit in an attacker-readable slot. Worth a design pass for the
   "don't spill the crown jewels" pattern (and a differentiator vs baseline CHERI).
3. **Don't spill sensitive caps at all** where avoidable (keep them in callee-saved
   regs / a trusted region), an ABI/codegen policy question.

---

## 2. Distinguishing integers from capabilities — the tag vs the reviewer's MAC idea

**How we distinguish them today: an out-of-band hardware tag bit.** Each
128-bit (16-byte) capability-aligned memory slot and each register carries **one
extra bit** that is *not part of the addressable bytes*. It is set only by
capability-producing instructions (`cincoffset` from a valid cap, `ldc` from a
tagged slot, the boot root caps) and is cleared by any integer write to the slot.
Hardware/QEMU checks it on every capability use (`ldc`/`stc`/`cjalr`).

This already delivers the property the reviewer wants — **capabilities are disjoint from
integer representations** — and it does so *more strongly than a checksum/MAC*:

| Property | Out-of-band tag (ours) | MAC / checksum (the reviewer's idea) |
|----------|------------------------|---------------------------------|
| Forge by writing bytes | **Impossible** (tag not addressable) | Possible if MAC key/algorithm leaks; probabilistic |
| Cost | 1 bit per 16 B + tagged memory | CPU to compute/verify per use; key management |
| Survives copy through untagged medium (disk/net) | **No** — tag is lost | **Yes** — that is its whole point |
| Needs trusted memory with tag storage | Yes | No (works on plain RAM) |

**Empirical confirmation (the "forge" probe, §11):** `int *forge(unsigned long x)
{ return (int*)x; }` compiles to a bare register move with **no tag-setting
instruction**; the result is **untagged** and faults on deref. There is no
unprivileged instruction that turns arbitrary bits into a tagged capability. So
**forging from an integer is closed by construction** (threat T1).

**When the MAC idea becomes relevant for us.** A MAC is the right tool exactly
where the out-of-band tag *can't* travel: serializing a capability to a medium
without tag bits — **persistence (disk), networking, or a non-tagged shared
region.** The moment we want SQLite to put a pointer in a file, or to swap pages
to untagged storage, the tag is gone and we would need a cryptographic
authenticator (MAC, à la ARM PAC / "cryptographic capabilities") to detect
tampering on reload. So: not needed for in-memory safety (the tag is better), but
**a real design point for the SQLite-on-disk / persistence story.** Worth one
paragraph in the paper to show we understand both regimes and chose the right one.

---

## 3. How is `ptr → int → ptr` compiled? What should we do?

**Probe `roundtrip` (§11):** `(int*)((uintptr_t)p + 4)` compiles to:
```
mv   a0, a0      ; ptrtoint  — take the address bits (no tag carried out)
addi a0, a0, 4   ; INTEGER add — result is a plain integer, tag = 0
mv   a0, a0      ; inttoptr  — no tag-setting instruction
```
There is **no `cincoffset`** anywhere: the value travelled entirely through
integer instructions, so the returned pointer is **untagged** and a deref
**faults**. This is **fail-safe** behaviour (threat T2): a `ptr→int→ptr`
round-trip silently *loses* authority rather than silently *forging* it.

**Two extra facts worth knowing:**
1. **`uintptr_t` is 64-bit on our target** (the compiler even warns: *"cast to
   smaller integer type"*). A 128-bit capability **cannot** survive a trip through
   `uintptr_t` — you lose the tag *and* the upper metadata. This is a real
   divergence from purecap CHERI, where `uintptr_t`/`__intcap` is a 128-bit
   capability that *preserves* the tag across int arithmetic.
2. Consequently any idiom that relies on `intptr` round-tripping (XOR-linked
   lists, low-bit pointer tagging, hashing a pointer then reusing it) **breaks**
   on our target. Mostly that's fine (and even desirable for provenance), but it
   needs to be a *decision*, not an accident.

**What we should do — a provenance design decision (feeds contribution C2):**
- **Default: keep it fail-safe** (the current behaviour). It is exactly the
  provenance discipline "a pointer must come from a pointer." Document it and test
  it (the negative-test battery in `capability-authority-audit.md`).
- **Decide the `uintptr_t` model explicitly.** Either (a) keep `uintptr_t` scalar
  and tell programmers "round-trips drop authority" (simple, CHERI-purecap
  forbids most of these anyway), or (b) adopt a CHERI-style `__intcap` that is a
  full capability and preserves provenance through integer ops (more source
  compatibility, more lowering work). **Recommendation: (a) for the paper**, with
  (b) noted as future work — (a) is the cleaner provenance story and matches our
  64-bit `uintptr_t` reality.

---

## 4. How is the difference of two pointers computed?

**Probe `pdiff` (§11):** `long pdiff(int *a, int *b){ return a - b; }`:
```
lcc a1, a1, 2    ; read field 2 (cursor/address) of b  -> integer
lcc a0, a0, 2    ; read field 2 (cursor/address) of a  -> integer
sub a0, a0, a1   ; integer subtraction of the two addresses
srai a0, a0, 2   ; exact signed divide by sizeof(int) = 4
```
`lcc rd, rs, 2` is the capability **field-query** instruction reading the *cursor*
(address) field (fields: 0=tag, 2=cursor, 3=base, 4=end, 5=perms). So a pointer
difference **projects both capabilities down to their integer addresses and
subtracts** — the result is a plain integer with no tag (correct: a pointer
difference is a number, not a capability; no tag/forging concern).

> **Resolved after the 2026-06-29 audit:** lowering recognizes the unlowered
> capability-cursor `SUB` before truncating its i128 carrier to XLEN. If exact
> signed scaling was represented as `SRL` by demanded-bits simplification, the
> XLEN operation is restored to `SRA`; genuine logical shifts remain `SRL`.
> `pointer_diff` and `pointer_diff_neg` both pass at runtime, and
> `ptr-diff-signed.ll` covers power-of-two, 12-byte, and logical-shift cases.
> (Cross-object subtraction is separately UB, as in standard C/CHERI.)

---

## 5. `malloc`: how it works now, what the suites use, how to do it properly

**There is no OS heap and no `brk`/`mmap`.** Domains are freestanding (§ a collaborator's
doc). The three suites:
- **CoreMark:** static memory (no `malloc`).
- **BEEBS:** almost all benchmarks are static; `dtoa` needed a heap → a static
  bump arena (`malloc_beebs`).
- **RV8:** several benchmarks (`dhrystone`, `primes`, `miniz`, `aes`, `norx`,
  `qsort`, `sha512` tails) call `malloc`/`realloc` → we link a shared static bump
  allocator `capstone/benchmarks/rv8/adapted/rv8_malloc.c`.

**How `rv8_malloc.c` works:** a 16-byte-aligned static array (`rv8_heap[]`), a
bump offset, a 16-byte header per allocation recording the requested size (so
`realloc` can copy); `free` is a no-op; `rv8_arena_init()` resets it. **16-byte
alignment is mandatory** — an allocation that stores a capability field must be
16-aligned or the `stc` drops the tag (this is what bit dtoa's bigint arena).

**Security-relevant gap [pre-narrowing]:** the returned pointer was `gp`-derived
and un-narrowed — segment/whole-image bounds, not the requested size, so
`malloc(16)` handed back a pointer that could roam the whole arena.

**→ now (contribution C1, heap arm — implemented):** the allocator narrows each
return:
1. Carve the requested (16-rounded) range from the arena.
2. **`cap_shrink` the returned capability to `[base, base+size)`**
   (`__builtin_capstone_cap_get_cursor` + `__builtin_capstone_cap_shrink`) so the
   caller gets a cap bounded to *its* allocation — done in `rv8_malloc.c` and
   dtoa's `malloc_beebs`. (`realloc` recovers its size header through the *wide*
   arena cap, since the narrowed user pointer can't reach `p-16`.)
3. Precision is the hardware's: **byte-exact < 4 KiB**, power-of-two grain above
   (`capability-bounds-model.md`) — exact in-register, may round on store/reload.
4. `free` is still a no-op; revocation (`cap_revoke`/`mrev`, temporal safety) is a
   separate, harder track.

Done + measured: RV8 7/7 ✓ and BEEBS `dtoa` ✓ with the narrowing allocator; trio's
`realloc_beebs` is intentionally *not* narrowed (it over-reads the old block — a
documented latent over-read, like the rijndael find).

---

## 6. How are capabilities created, and how are bounds assigned?

**Creation is always *derivation from an existing capability* — never from
integers.** The mechanism, seen in every probe (e.g. `take_global`, §11):
```
auipc a1, %pcrel_hi(g) ; addi a1, a1, %pcrel_lo   ; compute g's address (integer)
cincoffset a1, gp, a1                              ; DERIVE a cap to g from gp root
delin      a1                                      ; de-linearise (make it usable/non-linear)
cincoffset a0, a1, a0                              ; add the (scaled) index — tag preserved
```
- **Roots.** `gp` is the global/code/data root capability; `sp` is the stack root.
  These are handed to the domain at entry (`start.S` sets them up from the boot
  caps via `ccsrrw`/`scc`/`delin`).
- **Address arithmetic.** `cincoffset(cap, n)` moves the *cursor* by `n` and
  **preserves the tag, base, and end** (it is monotonic — it cannot grow
  authority). This is how *all* pointer arithmetic is lowered.
- **Bounds.** Set by the `SHRINK` instruction (`rd = shrink(cap, base, end)`),
  queryable with `lcc` fields 3/4 (base/end). **Pre-narrowing we never emitted
  `SHRINK` for object materialization**, so bounds were inherited from the root.
  **→ now:** object materialization emits `SHRINK` to narrow to the object —
  globals (`selectLGA`, default on), heap (`malloc`), stack (opt-in) — so a
  global cap is bounded to `sizeof(g)`, a heap cap to the request, an
  address-taken local to its size (when stack narrowing is on).

So: **creation = `cincoffset` from a root; bounds assignment = `SHRINK` to the
object at materialization** (was: none/inherited). The primitive
(`llvm.capstone.cap.shrink` → `selectShrink` → `SHRINK`) is now invoked by normal
object lowering. *This is contribution C1 — implemented; see
`capability-bounds-model.md`.*

---

## 7. Do we do capability splitting? How well?

Two senses to separate. **Compiler narrowing (the C1 sense the reviewer meant):**
*pre-narrowing* we did the anti-pattern — "one capability for (almost) all of
memory, move the cursor": every pointer was `cincoffset(gp/sp, …)` inheriting the
root's bounds. **→ now:** we narrow each materialized object to its bounds with
`SHRINK` (globals/heap default, stack opt-in). **Hardware capability *splitting*:**
the ISA has a distinct `SPLIT` instruction (split a cap into two adjacent halves)
and `SHRINKTO`; **neither is wired into LLVM** — we narrow with `SHRINK`, we do
not emit `SPLIT`. So "no splitting" conflated the two: the compiler now narrows;
the `SPLIT` primitive exists but is unused.

Evidence: *pre-narrowing*, across all probes there was **not a single `SHRINK`**;
**now** `SHRINK` is emitted at every sized-object materialization (lit
`cap-shrink-{globals,stack}.ll`; authority suite). `gp` is an ambient root
spanning the segment (single `PT_LOAD` ≈ whole image) — this is *why* globals,
function pointers, and the `__capstone_cap_init` table derive from it (see
`capability-globals-init-decision.md`); the derived object caps are then
`SHRINK`'d.

**Initial slices (the C1 work):** `SHRINK` at materialization sites —
- **globals:** narrow `cincoffset(gp,&g)` to `sizeof(g)` ✓ (default on);
- **stack:** `FrameIndex` narrowing incl. interior/varargs/dynamic-alloca ◑ (`-capstone-shrink-stack`, off);
- **heap:** `cap_shrink` in **two benchmark allocators** ◑ (rv8/dtoa; not libc-wide);
— exact in this QEMU (representability not observable). **Functional** validation
only across CoreMark/BEEBS/RV8; rijndael OOB found. **Overhead NOT measured.**
Remaining (audit): coverage matrix, overhead numbers, stack default-on, subobject,
permissions/W^X, `SPLIT`/root-elimination.

---

## 8. The array example: `char *p = &a[10]; q = p+5 (ok); q = p+25 (?); *q`

I reproduced it with a **16-byte** object (`char a[16]`) so that `p+25` (→ `a[35]`)
genuinely leaves the object, matching the reviewer's intent (§11, `arr.c`).

**1. Does it compile?** **Yes — both, with no error and no warning.** The
front-end and backend treat pointer arithmetic uniformly; nothing rejects an
offset that leaves the object.

**2. Does it fault at runtime?** **Pre-narrowing: no — neither.** Both compiled to
a `gp`-derived capability to `a` plus a load at the constant offset:
```
in_obj :  cincoffset a0, gp, &a ; delin a0 ; lb a0, 15(a0)   ; a[15]
out_obj:  cincoffset a0, gp, &a ; delin a0 ; lb a0, 35(a0)   ; a[35] — OUT of a[16]
```
Because the capability's bounds were `gp`'s (segment ≈ whole image), the bounds
check passed for `a[35]` even though it is past the 16-byte object.

**→ now (default):** materialization inserts `SHRINK` —
`cincoffset a0, gp, &a ; delin a0 ; lcc/add ; shrink a0, &a, &a+16 ; …` — so the
cap to `a` is bounded to `[a, a+16)`, and the `a[35]` load **traps**
(`Cap mem access OOB`). The reviewer's expectation ("`p+25` does not work") **now holds**.
This is the authority suite's `global_oob`/`stack_oob` before/after (the "before"
is `-mllvm -capstone-shrink-globals=false`). *(One nuance: a **constant** OOB
offset folded into a non-narrowed interior pointer can still avoid the trap; the
runtime-index/whole-object cases trap. Subobject + constant-interior cases are the
documented residual.)*

**3. How is pointer arithmetic implemented?**
- **Constant offset** (`p+5`, `p+25`): folded into the load's immediate
  (`lb …, 35(a0)`) or into a single `cincoffset` — the offset is known at compile
  time.
- **Runtime offset** (`p+i`, attacker-controlled, probe `deref_attacker`): emitted
  as `cincoffset a0, <cap>, a0` — **tag-preserving, monotonic** address arithmetic,
  then the access. The tag is kept; only the cursor moves.

**With contribution C1 (bounds on `a`):** materializing `a` as
`SHRINK(cincoffset(gp,&a), &a, &a+16)` makes `a`'s capability `[a, a+16)`; then
`a[35]` is **out of bounds** and `*q` **traps** — which is the spatial-safety
guarantee the reviewer wants. `a[15]` still works. This example becomes the paper's
"before/after" figure.

---

## 9. Contribution C1 — spatial safety via best/near-ideal granularity

**Claim to prove:** *every memory access through a compiler-generated capability
is bounded to the object it names (modulo a stated, measured representability
rounding), so intra-domain spatial violations trap.*

**What already exists (lowers the risk):**
- The narrowing primitive: `SHRINK` instruction + `llvm.capstone.cap.shrink`
  intrinsic + `selectShrink` in ISel; bounds queries `lcc base/end`.
- The derivation discipline: `cincoffset` is monotonic (can't grow bounds).
- A measurement harness: CoreMark + BEEBS (82) + RV8 (7) as the regression/perf
  baseline; tight bounds that break a benchmark are *findings*, not just failures.

**What was built (→ now):**
1. **Globals bounds** ✓ — narrowed at materialization (`selectLGA`) incl. the
   `__capstone_cap_init` path; `-capstone-shrink-globals`, default on.
2. **Stack bounds** ◑ — `FrameIndex` narrowing incl. interior pointers/load-store
   bases, varargs save area, and dynamic (runtime-sized) allocas;
   `-capstone-shrink-stack`, **default off** pending a clean default-on matrix.
3. **Heap bounds** ✓ — `cap_shrink` in the allocator (§5; `rv8_malloc.c`, dtoa).
4. **Representability** ✓ — confirmed (`capability-bounds-model.md`): byte-exact
   < 4 KiB, power-of-two grain above.
5. **Sub-object** — explicitly out of scope (a known CHERI limitation); stated.

**The "prove" angle.** Materialization sites are narrowed by construction; the
regression evidence is the authority suite + `cap-shrink-{globals,stack}.ll`. A
systematic *checker pass* (proving no object cap is wider than its object) is the
C2 verifier proposal. The *measurement* landed: **CoreMark/RV8 7-7/BEEBS 82-82
green** with narrowing on, and a **real OOB write found** (rijndael) — that
exposure is itself a result.

**Scoping now:** claim *provenance + inter-domain isolation* **+
object-granularity spatial safety for narrowed globals/heap (stack with the flag)*,
with residual = subobject / stack-default / inter-procedural. (See threat-model §6.)

---

## 10. Contribution C2 — provenance ("every pointer comes from a pointer")

**Claim to prove:** *no compiler-generated path produces a tagged capability from
non-capability (integer) data; authority only ever flows by derivation from an
existing capability.*

**What already supports it:**
- **Forging is closed** (T1): `inttoptr`, union/byte punning, and byte-wise
  `memcpy` all yield **untagged** values (probe `forge`; the freestanding
  `memcpy`/`memmove` are byte-oriented, so they move data *without* the tag).
- **The `cincoffset` vs `addi` split** is the mechanism that keeps derivation
  honest (probes `roundtrip` vs `take_global`).
- **A real, found-and-fixed defect** to motivate systematic testing over
  "by inspection": stack-passed capability arguments were once delivered untagged
  because the outgoing slot address used integer `ISD::ADD` instead of
  `CIncOffset` (see `research-decisions-log.md`). Also `va_list` lowering and
  sub-capability `memcpy`. These are perfect paper examples of threat T2.

**What to build (per `capability-authority-audit.md`):**
1. **Authority-constructor inventory** — enumerate every legal tag-producing/
   preserving site (roots, `cincoffset`, `ldc` from tagged memory, `delin`,
   reviewed intrinsics) and assert nothing else sets a tag.
2. **A provenance lattice + a checker** — `ScalarOnly → … → CapabilityDerived`,
   forbidden transitions; a verifier pass (or MIR check) that flags any tagged
   value whose definition isn't a legal constructor.
3. **A negative/laundering test battery** — `inttoptr`, union pun, `memcpy` of a
   pointer, `va_arg`, `setjmp`/`longjmp`, struct-return, >8 pointer args, atomics,
   inline asm — each must yield untagged (fault) or be correctly preserved. This
   doubles as a regression gate **and** a paper artifact.

**The "prove" angle.** State the property as a lowering invariant over the lattice
and discharge it by (constructor inventory) + (checker) + (exhaustive negative
tests). The `uintptr_t` decision (§3) is part of this contribution's scope.

---

## 11. Reproduction — probe sources and disassembly

Compiler: `$CAPSTONE_CLANG -target capstone64-unknown-elf -Xclang -target-feature
-Xclang +m -ffreestanding -fno-builtin -O1 -S`.

```c
int  *forge(unsigned long x)      { return (int*)x; }                 // → untagged
int  *roundtrip(int *p)           { uintptr_t x=(uintptr_t)p; x+=4; return (int*)x; }
long  pdiff(int *a, int *b)       { return a - b; }
char a[16];
char in_obj(void)  { char *p=&a[10]; return *(p+5);  }                // a[15] in
char out_obj(void) { char *p=&a[10]; return *(p+25); }                // a[35] OUT
int *take_global(long i)          { extern int g[64]; return &g[i]; }
```

Key emitted sequences (full listings: `scratchpad/probes.s`, `arr.s`):
```
forge:        mv a0, a0                                  ; no tag set → untagged
roundtrip:    mv a0,a0 ; addi a0,a0,4 ; mv a0,a0         ; integer path → untagged
pdiff:        lcc a1,a1,2 ; lcc a0,a0,2 ; sub ; srai a0,a0,2   ; signed integer result
out_obj:      auipc/addi &a ; cincoffset a0,gp,a0 ; delin a0 ; lb a0,35(a0)  ; NO bounds check
take_global:  slli a0,a0,2 ; cincoffset a1,gp,&g ; delin a1 ; cincoffset a0,a1,a0  ; NO SHRINK
```
Cross-checks in the backend: `CapstoneInstrInfo.td` defines `SHRINK` (bounds
narrowing), `lcc` field map (0=tag,2=cursor,3=base,4=end,5=perms), `cincoffset`,
`delin`, `seal`/`tighten`; `IntrinsicsCapstone.td` exposes `cap_shrink`,
`cap_get_base/end`, `cap_seal`, plus the domain-transition intrinsics
(`cap_enter/return/exit/ccsrrw`). **[Pre-narrowing baseline, commit
`dcc9c0cee120`: no automatic `SHRINK` emission.]** Since then, object
materialization emits `SHRINK` automatically (globals/heap default,
stack opt-in) — see the banners at the top and the "→ now:" notes.

---

## 12. Suggested next steps (engineering, ordered)

*Status (2026-07-03): items 2 and the heap half of 4 are **done** — the C1 slices
that were "next steps" when this was written have landed. Rewritten to reflect
what remains.*

1. **Authority-constructor inventory + negative-test battery** (C2 foundation;
   cheap; turns the found bugs into a permanent gate). *Also the artifact to hand
   a collaborator.* — **partly realized** as `../../tests/capstone-authority/`; the
   systematic *checker* is still the `c2-provenance-verifier-proposal.md` work.
2. ~~**Globals bounds via `SHRINK`** — smallest end-to-end C1 slice.~~ **DONE**
   (`selectLGA`, `-capstone-shrink-globals`, default on; three suites re-run green;
   rijndael OOB found). What's still open is **overhead measurement** (the
   instruction-count proxy exists; cycle-accurate is pending the RTL vehicle).
3. **Confirm `gp`'s exact bounds and the representable-bounds rule** against the
   monitor/loader and the ISA (both underpin every claim). — bounds model settled
   in `capability-bounds-model.md`; the monitor/loader cross-check remains.
4. ~~**Stack + heap bounds**~~ → **heap DONE** (two benchmark allocators);
   **stack implemented but default-off** pending a clean default-on matrix; then the
   **provenance checker** pass (still open, C2).
5. **Decide the `uintptr_t`/`__intcap` model** (§3) and the **persistence-MAC**
   question (§2) — both are short written decisions, not large builds. (Still open.)
