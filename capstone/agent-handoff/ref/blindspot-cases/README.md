# Blind-spot cases: the index

Cases where **standard CHERI cannot see the bug, because of how a nested allocator
is structured.** One file per piece of software, worked through in order of yield.

## The criterion, restated so nobody has to re-derive it

CHERI narrows a capability at `malloc()`. A nested allocator calls `malloc` once for
a pool or a page and then carves every object out of it by pointer arithmetic, with
no `csetbounds` per object. Inside that region, therefore:

* an overflow from one sub-object into the next is **in bounds** -- nothing raised;
* a use-after-free yields a pointer that is still **tagged and in bounds**, and no
  `free()` ever reached the system allocator, so revocation has nothing to revoke;
* the allocator's own metadata usually lives in the same region, so corrupting it
  is in bounds too.

**The test for every case: does the bad access stay inside the region the nested
allocator owns?** A bug that overflows a standalone `malloc` buffer is the opposite
of what belongs here, because CHERI catches those trivially. Those are recorded as
class C and rejected, not silently dropped -- knowing what was rejected is what
makes the count meaningful.

| class | where the bad access lands | purecap | + revocation |
|---|---|---|---|
| **A** | inside a live pool/page, on the allocator's own free list | blind | **blind** |
| **B** | the whole pool was returned to `malloc` | blind | catches |
| C | a standalone `malloc` buffer | catches | catches |

Class B becomes class A whenever the pool can be made never-freed, which several of
these allocators support directly.

## Two traps that make a clean result meaningless

**A sanitizer cannot see class A.** ASAN only ever observes `malloc` and `free`. An
allocator that recycles a slot on its own free list performs neither, so the bug it
is hiding produces no report. Where the harness is a sanitizer, the measurement is
of the sanitizer. Build the oracle around a **wrong answer** instead.

**Pool allocators round the request up.** SQLite's `memsys5Roundup`, TLSF, EMS and
umm all do. If the capability handed back carries the BLOCK's bounds rather than the
REQUESTED size, a small overflow is invisible for a second, unrelated reason, and a
"not caught" verdict says nothing. Narrow to the request, and negative-test that
narrowing before trusting any verdict.

## The order of work, by yield

| # | software | usable cases | with a script | harness state | file |
|---|---|---|---|---|---|
| 1 | **mruby** | **36** (26 A + 10 B) | 21 | CHERI port RUNS; Capstone port ~11 census errors | [mruby.md](mruby.md) |
| 2 | MicroPython | ~22 | few | **runs today** on both | to write |
| 3 | standalone allocators (TLSF, umm, tinyalloc, dlmalloc, Contiki, Zephyr) | ~14 | some | no harness yet, but they ARE the pool, so it is small | to write |
| 4 | WAMR EMS | 5 | 1 in-tree C reproducer | **runs today**, stage 40 built | to write |
| 5 | SQLite (intra-pool subset only) | ~9 | most | runs today; needs an older amalgamation per bug | to write |
| — | JerryScript | ~92 issues, the purest source | many | **BLOCKED**, see below | not usable |

**JerryScript is excluded and the reason is a result, not an excuse.** Its heap is a
single `uint8_t area[]` with everything carved from it, which makes it the most
uniformly CHERI-blind engine of the set. It achieves that by storing every reference
as a compressed offset and rebuilding addresses arithmetically at 93 sites across 60
functions, and `uintptr_t` cannot be made capability-wide on this target. **The
design that hides an allocator from CHERI is the design Capstone's tag model refuses
outright.** Worth a paragraph in the paper; useless for measuring.

## Status vocabulary

Reused from `../paper-bug-inventory.md` so the two files can be read together.

| | meaning |
|---|---|
| **BOTH** | Capstone column and CHERI column measured |
| **CHERI** | CHERI column measured, Capstone column missing |
| **REPRO** | reproducer runs somewhere; no mechanism column yet |
| **TRIAGED** | upstream-verified, not built |
| **BLOCKED** | cannot be built, with the reason given |
