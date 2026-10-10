# Perl 5.36.3: spatial and temporal defects live at the pin

> **Correction, 2026-10-11.** The four-arm account below (2026-10-06) counts 04 (`e4be969235`) and
> 06 (`40727c420c`) as caught on every Capstone arm and on CheriBSD. Those faults were, by every
> sign, a Perl port bug and not the defects: both triggers load `overload.pm`, whose
> `builtin::blessed` call goes through a descriptor pointer kept as an integer (`builtin.c`), which
> loses its capability tag. On 2026-10-11 the same faults appear with no defect involved, on the
> Capstone virtual port (cause 24 in `ck_builtin_func1`) and on CheriBSD purecap (one instruction
> in `Perl_sv_2uv_flags`, revocation on and off), and they go away with port patch 0010. The
> physical 2026-10-06 images are gone, so this rests on the same unpatched upstream code being in
> them, and on their faults' shape (cause 24 on the bounds-only arm too; SIGPROT with revocation
> off too), not on rerunning them. The counts below that include 04 and 06 are therefore not
> readings. The current measurement, on the Sublet platform and on CheriBSD with patch 0010, is
> [`results/2026-10-11-virtual`](results/2026-10-11-virtual/README.md).

**Scope: spatial and temporal memory defects only.** That is this study's subject,
and it is the only thing this corpus admits. A defect that is a null dereference, a
type confusion, an integer overflow with no out-of-bounds consequence, or a crash
from stack exhaustion is not here, however real it is.

Eleven cases. Each is a defect upstream Perl fixed after 5.36.3, whose fix never
reached the 5.36 maintenance branch, and whose trigger is the test upstream added
with that fix.

## Liveness is measured, in both directions

A case is here only because the measurement puts it here:

| | at the pin (5.36.3 + ASan) | on master (`v5.45.3-85-gdb19522155` + ASan) |
|---|---|---|
| reports | **11 of 11** | 0 of 11 |
| silent | 0 of 11 | **11 of 11** |

Same trigger, same harness, same `ASAN_OPTIONS`, two builds. Full table in
[`results/20261006-host-differential`](results/20261006-host-differential/).

**The dates do not settle this, and would get four cases wrong.** Four of the eleven
fixes are *dated before* the `v5.36.3` tag (2023-11-28) and are still absent from it:
they landed on blead and were never backported. So each case records the ancestry that
was actually checked — `git merge-base --is-ancestor <fix> v5.36.3` false,
`--is-ancestor <fix> <blead>` true — rather than a date comparison.

## What the pin's own ASan can and cannot see

This is the finding that motivates the nested-allocator arm, and it is visible in the
host measurement before any capability hardware is involved:

| what the pin reports | cases | where the memory lives |
|---|---:|---|
`heap-use-after-free` | 3 | `malloc`/`realloc`/`calloc` — the **system allocator**, so ASan holds the allocation record |
`SEGV on unknown address`, no allocation context | 5 | a stale `SV*` or a wild pointer; ASan does not know this memory |
Perl's own refcount check | 2 | the SV head arena; ASan silent |
wrong bytes only (`$&` comes back as `b\0`) | 1 | a stale COW `subbeg`; both oracles silent |

**Three of eleven are visible to ASan.** The other eight are in Perl's own arenas, or
are derefs of a pointer that never had an allocation record. A sanitizer that works at
the `malloc` boundary cannot see a lifetime that never crosses it.

## The arms

Four arms, **cumulative**: each step adds exactly one thing, so the difference between
two adjacent arms is that one thing and nothing else.

| arm | protection | catches |
|---|---|---:|
| **`sysalloc-bounds`** | the base an application gets today: per-object heap bounds, `free` only marks | **6** |
| `sysalloc-sublet` | the **system allocator** spatial *and* temporal — the Sublet heap: bounded, every free revokes | **7** |
| `sublet-svheads` | that same image **plus the nested allocator**: the SV head arena through the lifetime adapter | **8** |
| `cheribsd-revocation` | the same release on CheriBSD, revocation as it comes | **7** |

**`sysalloc-sublet` is the floor, and it is the right floor because it equals CheriBSD
case for case.** Not the same count — the same *set*: both report on the same 7, with no
case unique to either. Protecting the system allocator spatially and temporally in a
Capstone domain reproduces CheriBSD's revocation exactly.

| | catches | against the floor |
|---|---:|---|
| `sysalloc-bounds` | 6 | −1 (`10_254b30e378`: bounds cannot see a stale `SvPVX` buffer, revocation can) |
| **`sysalloc-sublet` — the floor** | **7** | — |
| `cheribsd-revocation` | **7** | **identical set** |
| **`sublet-svheads`** | **8** | **+1: `05_17535c984a`, an SV head** |

0 regressions at either step.

**`05_17535c984a` is the argument for the third arm, in one case.** Its `CvXSUBANY` SV
head lives in Perl's own head arena and is never returned to `malloc`. So host ASan is
silent, protecting the system allocator spatially *and* temporally does not help —
`sysalloc-sublet` is clean — and **CheriBSD is clean with revocation both off and on**.
Only Perl's own `Attempt to free unreferenced scalar` and the SV head adapter report it.
The nested allocator is where the lifetime is, and nothing that works at the `malloc`
boundary can see it.

So the whole result is one case wide, and that is the point: once the system allocator
is protected as well as CheriBSD protects it, **the only thing left to gain is the nested
allocator**, and it gains exactly `05_17535c984a`.

Where bounds buy precision rather than reach: the spatial case `08_b7b77ffc1e` faults on
all three Capstone arms as a **bounds** violation (cause 5). An unprotected variant was
measured too and catches 6 as well — it differs only in reporting that case as an
untagged dereference (cause 24). It is in the results bundle as context, not as an arm.

### The cumulative arm needed a change to the port

Before this, no single image could carry both protections. `build.py` refused the
combination outright — `--nested` with `--heap sublet` is a `ValueError`,
*"nested discovery arms use level0 for their separate outer heap"* — and the refusal is
technical, not just methodological: one grant has two consumers, and `regions.c`'s
default wrapper for Perl returns 0 for region index 0. That is harmless for `level0`,
which keeps a static arena, and fatal for a Sublet outer heap, which needs a real region.

`PERLD_HEAP=sublet-svheads` now grants `(2 << HEAP_LOG) + 32 MiB` and compiles
`regions.c` with `EXP_HEAP_AND_POOL`, splitting it: region 0 to the Sublet heap, region 1
to the adapter. This is exactly what `ports/mruby/app/build-mruby-domain.sh` does for
`sublet-gc` — so Perl's third arm is now the same construction as mruby's, which it was
not before.

Checked rather than assumed: the grant reached cmake as `167772160`; the linked
`regions.o` is the splitting branch and not the fallback (it references `abort`, which
only `EXP_HEAP_AND_POOL` does, 2184 bytes against the fallback's 1328); the image hash
differs from every other arm's; and the arm passed both of its controls, which a failed
region split would not.

### What no arm reaches

Of the eleven, ten execute their defect somewhere and **eight are caught**:

- **`11_af11b0c528` is out of reach by construction.** The stale `subbeg` length makes
  `$&` read one byte past the logical string into the NUL terminator, which lies *inside
  the same allocation*. No per-object bounds and no revocation can see an in-bounds read
  of a live object; every arm produces the wrong output and nothing more.
- **`02_d2cddbe1df` is a deliberate carve-out.** Upstream Perl reads a freed head's flags
  on purpose, so the adapter answers `SvIS_FREED` and `SvTYPE` from its sidecar; making
  that path fault stops three upstream test files. Perl's own
  `panic: attempt to copy freed scalar` catches it instead, on both platforms.
- **`03_9e298ab597` is a trigger that does not travel.** No arm reports the defect, on every
  arm and on CheriBSD with revocation off *and* on, but the trigger cannot say whether the stale
  element slot ever holds a stale pointer there: its only check is `pass()`, which
  `harness/shim.pl:8` makes unconditional. (Corrected 2026-10-11; this line used to read
  "its own assertion passes ..., so the stale element slot never holds a stale pointer", which
  that unconditional pass cannot show.) Host ASan reports SEGV, so the defect is real. It is the
  only case outside every denominator, which is therefore 10.

### A library gap that cost a verdict, and the two traps in fixing it

`10_254b30e378` was recorded as `harness_limit` here on 2026-10-06 and that is
**withdrawn**. perl-cross installs almost none of the *target* library — 636 pure-Perl
modules the same release installs natively were absent, `XSLoader.pm`, `DynaLoader.pm`
and `PerlIO/scalar.pm` among them. Every extension is linked statically, but perl still
reaches an XS layer through its `.pm`, so `PerlIO_find_layer("scalar")` could not
`require PerlIO::scalar` and the open fell back **silently**: the domain reported
`LAYERS perlio` where the native reference reports `LAYERS scalar`, writes returned
success, and the backing scalar stayed `undef`.

`build-perl-domain.sh` now stages the library from its own native reference and refuses
to continue if any of those three is still missing. Two things made that step easy to
get wrong, and both were caught by checking its output rather than trusting it:

- the arch-dependent `.pm` stubs live under `lib/<ver>/<archname>/`, so walking the
  version directory alone copies `DynaLoader.pm` and `PerlIO/scalar.pm` somewhere
  `@INC` never looks — the fill reports hundreds of files and still leaves the two that
  matter missing;
- detecting that directory by looking for a `Config.pm` inside it also matches `Net/`,
  because `Net::Config` exists, which flattens `Net/*.pm` into the library root. The
  archname comes from the native perl instead.

### The arms carry their own positive control

A base and a revoking arm that reported the same six would look exactly like a build
whose revocation never fires, so that had to be ruled out before the +1 steps meant
anything. [`results/20261006/heapprobe.c`](results/20261006/heapprobe.c), compiled
with each arm's own SDK and run as a domain, does `malloc`, `free`, then a read at an
argv-derived index whose value is printed so the load cannot be folded away:

| | `sysalloc-bounds` | `sysalloc-sublet` |
|---|---|---|
| read after `free` | `READ-FREED-OK got=Z` — completes | the process ends at the read |
| `realloc` 26→52, 1000→4000 | `moved=1` | `moved=1` |

So revocation is live, `l0_free` really does only mark, and a dangling pointer into a
grown buffer *is* possible here — "realloc grew in place" was considered and refuted.

### CheriBSD: identical to the floor, and the one case it cannot reach

`cheribsd-revocation` reports on **7 of the 11 — the same seven as `sysalloc-sublet`**,
case for case. The one it lacks is `05_17535c984a`.

**Is that head quarantined but not yet swept, or never in the quarantine at all?** The
shadow bitmap answers it in O(1) without running a sweep — one bit per 16-byte granule,
set while quarantined ([`quarantine-probe.c`](results/20261006-cheribsd/quarantine-probe.c),
`LD_PRELOAD`ed around `malloc` and `free`). Under default revocation, for that case:

| | |
|---|---:|
| mallocs | 4072 |
| frees | 2983 |
| of those, quarantined | **2983** — all of them, so the quarantine is fully active |
| mallocs returning memory whose bit was **still set** | **0** |

**Never in the quarantine.** Had the head travelled `free()` → `malloc()`, it would either
have been reissued while quarantined — counted, and the count is 0 — or swept first, which
would have revoked the stale capability and faulted. Neither happened. Perl recycles the
head on `PL_sv_root`, so it never crosses the allocator boundary and there is nothing to
quarantine.

The probe is controlled in both directions: with revocation on a freed block reads 1 and a
live one reads 0; with revocation off a freed block reads 0. So the bit tracks quarantine
membership rather than merely having been freed, and a zero from it is informative.

**An every-free run was tried and is rejected as evidence.** Under
`_RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1` this case does SIGPROT — and so do all eleven,
including `11_af11b0c528`, an in-bounds read of a live object that no bounds and no
revocation may legitimately catch. That configuration cannot distinguish, so its verdicts
are not attributable, and an earlier inference drawn from it is withdrawn.

That is the three-of-eleven ASan result above, repeated in capability hardware: a
defence at the `malloc` boundary cannot see a lifetime that never crosses it, whichever
boundary technology it uses.

### That arm carries its own knob control

Every case ran **twice**, revocation off and on. Without that pair, "7 SIGPROT" would be
indistinguishable from a knob that does nothing:

| | SIGPROT | clean | Perl's own panic |
|---|---:|---:|---:|
| revocation **off** | 5 | 5 | 1 |
| revocation **on** | **7** | 3 | 1 |

`10_254b30e378` and `07_39b4841b25` are clean with it off and SIGPROT with it on, so the
env var is provably live. The five that fault either way are spatial or caught by bounds
alone.

### Two platform constraints, both forced

- **Revocation is selected per process** (`_RUNTIME_REVOCATION_ENABLE=1`), with the
  system-wide `security.cheri.runtime_revocation_default` left at 0. With the system
  default at 1 the guest's own `sshd` runs under revocation, hits the PoisonCap fault and
  dies — even a 2.3 MB `scp` fails with `Connection closed by remote host`, while the
  identical copies succeed with the default at 0. The mruby arm made the same concession
  for the `every_free` knob. **The mruby arm ran under the system default, so the two
  programs' CheriBSD columns are not interchangeable.**
- **The image is the one whose loader accepts this binary.** The stock image's rtld
  refuses the dynamically linked purecap perl before `main` with
  `Traditional TLS not supported`, and this SDK's clang cannot emit the alternative —
  `-mtls-dialect=desc` is an unknown argument. The binary needs exactly one TLS symbol,
  `_ThreadRuneLocale` from FreeBSD libc's ctype inlines.
- A **static** link is accepted by the kernel and then dies with SIGPROT. That is **not a
  Perl defect**: a static purecap probe reads its own `__thread int` initialised to 7 back
  as `1074491920`, then faults at the first libc TLS access. Static TLS setup is broken on
  this platform for any program that touches it. An earlier note in this corpus
  attributing that fault to `perl_construct` as Perl's own is **retracted**.

## One nested allocator of four

`sublet-svheads` protects the SV **heads**. The SV *bodies*, the hash entries and the OP
slabs keep their upstream allocators, so a stale pointer living in those is outside this
arm's reach by construction — the +1 it shows is a floor on what a nested allocator buys
in Perl, not a ceiling. mruby's single GC heap holds essentially all of mruby's object
lifetimes, which is why `sublet-gc` added 6 there; Perl spreads its over four arenas.

## Layout

```
NN_<fix>_<slug>/
  case.json        the schema's fields, including the measured arm oracles
  trigger.pl       upstream's own test, on the shared harness
  PROVENANCE.md    the fix, the unfixed code at the pin, and both host measurements
harness/shim.pl    plan/ok/is/like/... collecting failures instead of aborting
results/           one bundle per measurement day
```

Every trigger's first line is `require "shim.pl";`, resolved through `@INC`, so the
same file runs on the host, in a Capstone domain and on CheriBSD with nothing but
`PERL5LIB` changing.
