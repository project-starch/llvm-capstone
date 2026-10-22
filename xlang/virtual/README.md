# xlang shims on the virtual Capstone profile: the virtual-malloc arm (vp2-malloc on the board)

`run-virtual.py` runs `xlang/repro`'s and `xlang/lua-cdp`'s C shims on the virtual profile in one boot. It writes one
verdicts-v1 bundle per corpus, and `capstone/bug-corpora/tools/verdicts.py` judges every row. The runner's docstring is
the contract.

## The project lead's decisions (relayed by the runtime lane, 2026-10-22)

"capi и xlang как рекомендовано" ("capi and xlang as recommended"). For xlang:

- **(a) Attribution:** the generic malloc controls (`bounds-malloc`, `uaf-malloc`, in the same boot) plus a per-row
  declared fault site. There is no test-only no-revoke runtime: the classic column's `XLANG_NO_REVOKE` control has no
  virtual analogue.
- **(b) The mruby shim rows are dropped:** the real-interpreter arms cover mruby. All 11 rows whose shim is an
  `mruby_*` shim (`xlang/capstone/rows.tsv` keys 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15) stay in the bundle as
  NO-READING:out-of-denominator, so the corpus total stays visible.
  - The relayed decision said "the 12 mruby shim rows". The table has 11. The repro corpus.json's `program` field
    ("12 of 15 rows are mruby or an mruby gem") carries the same count. Its four other rows are rlua (2), libpulse-binding and
    secp256k1, none of them mruby.
  - What is dropped is every mruby row, which is the decision's stated reason.

So 17 rows are scored: repro's 4 (rlua_userdata_uaf, rlua_escaped_handle_uar, libpulse_iterator_uaf,
secp256k1_preallocated_uaf) and lua-cdp's 13.

## How a row is read

- **The build:** the shim unchanged, `capstone-cc -O0 -g` on the virtual SDK, with its mock. It is linked with
  `virtual-main.c` through `-Wl,--wrap=main`.
- **Reached:** `XLANG BEGIN`, which the wrapper prints at main's entry. A shim prints nothing before its stale access,
  so this marker is entry, not the access.
- **Completed:** exit 0. A shim whose stale access completed prints `MOCK <row> use-after-free-survived` and returns 0.
- **Attributed:** the fault's function is one of the row's `fault_sites` in `sites.tsv`.
  - Each site is host ASan's frame #0 for the shim as built here (`cc -O0 -g -fsanitize=address`, shim plus mock), run
    2026-10-22 before this arm's first run.
  - `memcpy` is added where ASan's reported source line is an explicit `memcpy()` call: rows 1, 3 and 7. ASan names
    the caller there, while the virtual profile's `memcpy` is a separate runtime function (a sized global symbol in
    the image) that the same copy faults inside.
  - The stack use-after-return row (repro 2) has no site. ASan reports a null SEGV there, because stack-use-after-return
    detection is off, so the report is not the defect. That row can read MISSED or unattributed, but never CAUGHT.

## Pre-registration (2026-10-22, before the first run): qkit7-batch2, the batch SDK t3-freeze-nb-nomovc-cw-r60

Checked before this was written:
- all 28 rows enumerate (17 kept, 11 dropped), every kept row has a `sites.tsv` entry, and all 17 build;
- the runtime's `memcpy` is a sized global function, so the judge's symbolizer names it `memcpy`. (sqlite/capi-repros'
  row 11 read unattributed on exactly this point: a zero-size label inside a static function.)

**Predicted** (the exploratory run of these shims, capstone/tests/corpus-cell-probes/README.md, built the same way
less the wrapper):

| corpus | rows | verdict | reading |
|---|---|---|---|
| repro | 1 rlua_userdata_uaf, 3 libpulse_iterator_uaf, 7 secp256k1_preallocated_uaf | CAUGHT | cause 25 in `memcpy` |
| repro | 2 rlua_escaped_handle_uar | MISSED | completes, exit 0 (no allocator involved) |
| repro | 11 mruby rows | NO-READING:out-of-denominator | dropped, decision (b) |
| lua-cdp | lua-openssl-141 | CAUGHT | cause 25 in `openssl_cipher_ctx_free` |
| lua-cdp | the other 12 | CAUGHT | cause 25 in `main` |

- Controls: bounds-malloc fault, uaf-malloc fault.
- Tallies: repro CAUGHT 3, MISSED 1, out-of-denominator 11; lua-cdp CAUGHT 13.
- A row reading otherwise is reported as read, not re-sited.

**For the board (vp2-malloc):** the same invocation with `--board-kit <kit>` in place of `--virtual-kit`. Its
prediction is the virtual run's reading, row for row.
