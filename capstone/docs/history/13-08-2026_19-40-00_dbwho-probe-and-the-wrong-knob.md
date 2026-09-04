# DBWHO: watching the allocator's caller, and a boot spent on the wrong side of a knob

Date: 2026-08-13. Bitstream `caplifive_12august.bit`. Branch `capstone-bootstrap`.

## What this was after

Boot27 (commit `f6cb42a40b80`) measured `sqlite3DbMallocRawNN` running with a `db` whose cursor
sits 0x3e00 from the connection `sqlite3_open` returned, inside a 64-byte heap block holding SQL
text — valid tag, heap-wide bounds. Every property of the callee is consistent, so the wrong
object was chosen by the CALLER, and no probe had looked there. DBWHO was built to look.

## The instrument

`CAPSTONE_DBWHO_PROBE=1` (build-sqlite-silicon.sh) injects, at the top of the allocator, a
comparison of `db` against the connection. On the first mismatch it reports the return address,
the request size, the contents of the mistaken object, and a ring of breadcrumbs pushed by the
**25 functions that call the allocator directly**, so the caller can be named one hop further up
than a return address reaches. `CAPSTONE_DBFIX_PROBE=1` adds the matched arm: substitute the
connection and see whether the workload completes.

Three controls, each of which caught a real defect in the instrument before it could produce a
reading. They are the reason this note exists at all.

| control | first read | what it meant |
|---|---|---|
| `ok` (calls where `db` WAS the connection) | **0** | `capstone_real_db` was assigned inside the LOOKASIDE probe's `#ifdef` while DBWHO tests it outside. Every comparison was against zero and the `real != 0` gate made the probe do nothing. The report read exactly like "no wrong db was ever seen". |
| `ctlra` (return address on call #1) | **0** | Printing ra's TYPE next to it gave 7 — NOT_CAP. A return address here is a SCALAR, so `lcc` cannot read one through any selector, and any `ra` a probe reports via selector 2 is the guard's zero. Now read with `mv`. |
| `DBFIX selftest` | — | The repair arm is dead code under QEMU (`bad=0`), so a passing build says nothing about whether it substitutes the right object. It derives `c=101ffc3f0` against `real=101ffc3f0`, and on silicon `c=83bfc3f0` against `real=83bfc3f0`. |

Final QEMU state: 924 allocator calls, 920 comparisons, zero mismatches, workload passes. **The
wrong `db` does not reproduce under QEMU.**

## Two harness defects, both of the "ran the wrong thing and reported success" kind

* `run-sqlite-silicon.sh` copied the domain from a hardcoded path while the build script has
  always honoured `OUT_DIR`. Setting `OUT_DIR` built the new domain and **ran the old one**. The
  build log showed the probe being injected; the run log showed a clean pass; they described
  different binaries. Three QEMU runs were read as results, including a "the probe does not fire
  under QEMU" reading that was really "the probe was not in the domain that ran". The runner now
  resolves `OUT_DIR`, prints the hash of what it runs, and errors on a missing artifact.

* `CAPSTONE_DBFIX_PROBE` gated injected code but was never passed to the compiler, so the "fix"
  build compiled to the **same bytes** as the plain one — identical sha256 — and its QEMU pass
  looked like the fix arm working. `bake-sqlite-doms.sh` now refuses to bake two byte-identical
  variants.

## The board result, and the knob I built against the wrong side of

Four domains, one boot, every arm returning, differing only in the flags named:

```
sqoff    baseline                              SQLITE ERROR stage=create rc=11
sqon     + -capstone-guard-cap-granule-copies  SQLITE ERROR stage=create rc=11
sqwhon   + DBWHO                               SQLITE ERROR stage=create rc=11
sqfixn   + DBWHO + substitution                SQLITE ERROR stage=create rc=11
```

rc=11 is SQLITE_CORRUPT; the message is "malformed database schema". Three readings follow:

1. **The COMPILER-side S-06 guard changes nothing here.** Expecting otherwise was reasonable — a
   corrupt schema is what a lost high half looks like — but the guard-on and guard-off builds are
   indistinguishable at this failure. What carries the schema text is the LIBRARY memcpy.
2. **DBWHO and DBFIX are non-perturbing**: the no-probe baseline fails identically.
3. **DBWHO never fired**, so no wrong `db` occurs before this point. The 64-byte block seen as
   `db` on boot27 is **downstream** of an earlier corruption. That measurement was correct and
   was placed too early in the causal chain.

The governing knob was already in the tree, with its outcomes recorded in
`build-sqlite-silicon.sh`: `SQLITE_LDC_HIGH_HALF_FIXUP` defaults to **0**, and with it OFF the
schema text is corrupt and SQLite bails at CREATE with rc=11 — exactly the four lines above. With
it ON the schema is repaired, SQLite runs deeper into CREATE than it ever has on silicon, and
takes mcause 25. The default is OFF only because an error return is easier to work with than a
wedge. So this boot reproduced the documented baseline four times and could not have reached the
fault.

**Consequence for the next boot:** with the fixup ON the domain WEDGES, and a wedge never
returns, so the host never writes the payload and any DBWHO report dies with it. The report needs
an arm that both repairs the schema and returns — fixup ON plus the lookaside fallback — now
QEMU-validated green.

## Board-run notes

`k800` entry-stalled at `SQ: F/share2` in slot 1 on **three separate firmware hashes**, so the
documented REDRAW remedy did not apply and the stall was not a per-image draw. A domain that
RETURNS A RESULT is its own proof that the boot worked — a control is what you need to interpret
a FAILURE — so moving the control out of slot 1 turned three void boots into two useful ones.
`sqlite_host.c` calls this the "SHA5 stall lottery" and already carries a runtime probe selector
for the same reason.

## Tooling added

`bake-sqlite-doms.sh` builds several variants into one firmware image and proves each is present
by hashing the bytes **inside `rootfs.cpio`** rather than trusting a filename: `A=linux-rebuild`
before `A=opensbi-rebuild`, and a refusal to bake byte-identical variants. Negative-tested: MATCH
on a staged domain, STALE on a perturbed hash, NOT IN IMAGE on an absent name.
