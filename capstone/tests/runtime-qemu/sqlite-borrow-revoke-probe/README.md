# SQLite column-borrow / revoke probe (Stage-2 "after", row 3)

The Capstone "after" for `benchmarks/sqlite/cve-repros/row3_diesel_colname_cached`
(diesel **RUSTSEC-2021-0037**): a `sqlite3_column_*` pointer cached across
`sqlite3_step`, dereferenced once the row it named has advanced — a use-after-free
that is silent on a normal machine and caught by AddressSanitizer in the Stage-1
"before".

## Mapping to the Capstone borrow/revoke model (the working #70 path)

| SQLite | Probe |
|--------|-------|
| SQLite engine owning the row buffer | engine = lender (`*_guest.c`, `.user`) |
| binding that reads `column_text` and caches it | host = borrower (`*.smode.c`, `.smode`) |
| `sqlite3_column_text()` hand-out | `shared_region_annotated(PERM_IN, REV_BORROWED)` |
| `sqlite3_step()` advances the row | `revoke_region()` |
| cached pointer dereferenced after step | round-2 read of the cached capability |

`PERM_IN` (read-only) is used because the column is engine-produced data the host
only reads (E→H); `REV_BORROWED` makes the hand-out revocable.

## Result (validated 2026-07-06) — safe-fail

```
column buffer borrowed to host
round 1 (read before step) retval = 0xc01a0dedc01a0ded   # borrow live: real column read
host read column OK before step
step revoked the column borrow
entering round 2 (use-after-free read)
round 2 returned 0x000000000fa017ed                      # cached read faulted
use-after-free read TRAPPED (domain faulted, ret=0xfa017ed)
```

After the revoke, the host's cached capability reloads **untagged**, so the
round-2 read faults; the monitor cleanly terminates the domain
(`fault_return_from_domain`) and returns the sentinel `0x0FA017ED` to the engine
instead of a stale column value. The use-after-free is converted into a
deterministic trap while the pre-step read stays zero-copy.

## Build / run

```
./build-sqlite-borrow-revoke-probe.sh   # buildroot gcc; one .user + one .smode
./run-sqlite-borrow-revoke-probe.sh     # one guest boot; asserts the trap markers
```

This is the validated template for the other borrow/UAF/UAC rows (1–10, 16); each
differs only in the lifecycle point that triggers the revoke (`finalize`/`close`/
hook-unregister instead of `step`).
