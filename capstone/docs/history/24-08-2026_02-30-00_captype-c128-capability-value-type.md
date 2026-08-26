# captype: a capability is c128, and nothing has to guess any more

Branch `capstone-captype`, on top of `capstone-ptradd`. Five commits.

## What changed

An AS200 pointer stops being `MVT::i128` and becomes `MVT::c128`. i128 stays a
legal type alongside it while `__int128` shares the register class; `drop-i128`
removes that half.

The payoff is not the type, it is what the type makes unnecessary. Three
heuristics are DELETED, not rewritten:

| gone | what it used to guess |
|---|---|
| `isCapstoneIntegerOffset` | "does this value look like an integer offset?" |
| `isCapstoneCapabilityValue` | "does this value look like a capability?" |
| `getCapstoneCapabilityCursor` | the `lcc`-based address read those two fed |

Two more call sites became type questions on the way: which operand of a
cincoffset is the base (the c128 one), and whether an i128 logical can have a
capability operand (it cannot). **The Capstone target is 14 lines SMALLER than
before c128 existed.**

That guessing was not merely imprecise. For `inttoptr(sub(ptrtoint p, n))` it
emitted `cincoffset a0, a1, a0` -- base and offset swapped, with a1 an untagged
scalar, which faults on hardware.

## Results

| gate | result |
|---|---|
| Capstone lit | 59/59, from 10/59 at the start |
| clang Capstone tests | pass |
| RISCV + X86 CodeGen | 6 failures of 7526, all emutls/TLS, ALL PRE-EXISTING (measured: identical with these commits reverted) |
| authority suite | 32/32 |
| nightly core tier | see below |

## Four defects the lit suite could not see

Every one was found by the QEMU runtime suites after lit was green. Three of the
four are the same shape: a generic code path assuming a value is an integer.

1. **Capability comparisons folded through the FLOATING-POINT condition-code
   tables.** `ISD::getSetCCInverse` and friends ask `isInteger()`, and a
   capability is not one, so `setult` became `setoge`. CoreMark's
   `core_list_join` crashed the compiler on it.

2. **Every capability passed to inline asm was silently UNTAGGED.** An `"r"`
   operand is bitcast to the register class's first legal type, and this branch
   had selected every capability-width bitcast as the SCALAR move -- the move
   whose job is to DROP a tag. CoreMark's `CAPSTONE_DELIN(A)` came out as
   `mv a6, a1; delin a6`. A bitcast between two capability-width types is a
   reinterpretation of one register, so it is a plain COPY, which the coalescer
   then removes.

3. **`store(trunc(cap))` folded into a truncating store** whose stored value is
   not an integer.

4. **An unaligned capability load** reached `expandUnalignedLoad`, which refuses
   a non-integer type. Splitting into halves is the RIGHT answer and the oracle
   says so: `tagged_cap_memcpy_misaligned` expects the misaligned copy to lose
   the tag and the later deref to fault. Fixing this also fixed
   `beebs_nettle-des`, which goes through the same memcpy.

## Provenance: the one semantic decision

`(T *)((uintptr_t)p + n)` is an integer round trip, and IR says the result has no
provenance -- on this target, an untagged pointer that faults on first use. Real
C in this project writes exactly that and it used to work.
`recoverCapabilityFromAddressArith` rebuilds it as a cincoffset on the capability
the address came from.

This is NOT the guessing c128 removed. The capability is identified by TYPE, as
the operand of the TRUNCATE that read its address, and there must be exactly one
-- two addresses make a difference, which is not a pointer. Masking is left alone.
The security invariants hold: `forge_inttoptr` and `ptr_int_ptr_roundtrip` both
still tag-fault, because a volatile integer in between breaks the chain.

The real upgrade is a middle-end pass that rewrites uintptr_t round trips into
GEPs while provenance is still visible in the IR, which is where upstream CHERI
does it.

## New instrument

`capstone/tests/scan-tag-stripped-caps.py` reads a disassembly and flags any
`mv rd, rs` whose result is then used where a TAGGED capability is required.
Controls shown in both directions: 8 hits on the CoreMark domain built while
defect (2) was live, 0 on the same domain after. Exit 1 when it fires, so it can
gate a build. This is the failure class lit structurally cannot see.

## Lessons worth keeping

* **A retraction.** The first root cause for defect (2) was `copyPhysReg`'s
  live-source path, whose own comment predicts exactly that failure ("the QEMU
  suites will fail loudly"). It was wrong. Turning that path off with its flag
  left the `mv` in place. A plausible mechanism with a comment vouching for it is
  still a hypothesis; the control run is what settled it.

* **Never rebuild the binary under test while a test run is using it.** Two
  measurements were thrown away to this: 6440 of 7526 "failures" and then 2210
  of 2257, both from a concurrent `ninja` replacing `llc` mid-run. The same
  tests pass at 0 failures when nothing rebuilds. Suspect the instrument when a
  result is surprisingly bad, not only when it is surprisingly clean.

* **Truncating an offset without checking that it fits is a silent wrap.** Made
  once on `capstone-ptradd` and again here: `p + 2^64` became `p + 0` instead of
  being refused. `cap-constants-invalid.ll` caught it both times.

* **`uintptr_t` is 4 bytes on this target while a pointer is 16.** Verified with
  a positive control (`sizeof(uintptr_t)==8` and `==16` both fail; `==4` and
  `sizeof(void*)==16` both hold). Pre-existing, unrelated to this branch, and it
  means `(uintptr_t)p` truncates a 64-bit address to 32 bits. Not touched here.

## Confirmation runs

Two full core-tier runs, plus a standalone re-run of every case that failed in
either. The machine had a high QEMU boot-flake rate that night, so the tally
matters more than any single run's verdict.

| suite | run 1 | run 2 | standalone |
|---|---|---|---|
| authority | FAIL (2 compiler crashes) | FLAKE | **32/32** |
| rv8 | FAIL: dhrystone, primes | FAIL: miniz | all three **PASS** |
| beebs | FAIL: nettle-des, newlib-mod | FAIL: jfdctint | all three **PASS** (nettle-des 4 of 5, 1 infra) |
| revoke-on-free | FLAKE | PASS | |
| linear-uninit-corpus | FLAKE | PASS | |
| intra-domain-mrev | FLAKE | FLAKE | |
| hier-revoke | PASS | FLAKE | |
| the other 8 | PASS | PASS | |

**Every core suite has passed.** The failing set is DIFFERENT between the two
runs, which is what separates a flake from a defect -- and the two that were
NOT flakes (authority's crashes, beebs nettle-des) were both traced to a cause,
fixed, and re-measured against a baseline build of the parent branch.

Two regressions were caught this way and would not have been caught otherwise,
because lit was green for both: CoreMark's inline-asm untagging, and nettle-des's
unaligned capability memcpy. Both were confirmed real by building the PARENT
branch and observing the same case pass there -- a baseline, not an assumption.
