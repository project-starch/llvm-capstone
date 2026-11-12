# 2026-11-12: all 11 cases on both Capstone arms, on the one-operand CREVOKE and CRENEW

The first reading of cases 10 and 11 on the one-operand lifetime instructions, taken with cases 01-09
in the same boot so the two new rows can be read against the recorded ones. It is also the first
reading of `virtual-malloc` on those instructions: `../2026-11-11-virtual` re-measured the pools arm only.
`verdict_bundles` is unchanged, and this run sits beside the bundles it names.

## Platform

The kit of `../2026-11-11-virtual` and of `cpython/allocator-boundary-repros/results/2026-11-12-seed0`,
equal by sha256 in every platform field of `inputs.json`: QEMU `2d5a7107890e`, module `e47bca62addf`
at its default of one context, launcher `0f8e0f903725`, kernel `48eb725a40e4`, firmware `5b5c047a0030`,
compiler `0bbb1f88bc90`; pinned mode, `x-capstone-test-csrs=true,x-capstone-exact-bounds=true`. The
fixture is that run's cluster (stable tree `799df33864da`, tree `0353a5b9770f`).

The two servers were built from this branch at c8a7344bac59, `ports/common/application/build-virtual.sh
postgres` with `CAPSTONE_SDK_CORPUS=1` and `CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin`, patch 0003 at
sha256 `29b94c5a026e` on the pools arm. Both link test_decoding (case 11 decodes through it), so
neither hash is that of an earlier bundle:

| | sha256 |
|---|---|
| image, `virtual-malloc` (`PGSU_NESTED=none`) | `15e304ece399045555c76f4a98ba78ef96631d345fbdf0da79a78252c9df3e83` |
| image, `virtual-pg-pools` (`PGSU_NESTED=sublet`, `CAPSTONE_VIRTUAL_LIFETIMES=1`) | `7f5a8aadf15fdc2138585299c5f8ea28f02a5a7bbb7fc258b929a918c1ec03b2` |

One boot for each arm (`shared/run-arm.py --virtual-kit`), 27 of 27 sessions on `virtual-malloc` and
29 of 29 on `virtual-pg-pools`. Every control did what `tools/arms.json` declares: `bounds-malloc` and
`uaf-malloc` faulted on both arms, `bounds-palloc` and `uaf-palloc` on `virtual-pg-pools`. The two
boots ran side by side, with two CPython boots beside them for their first ten minutes, and took 52
and 73 minutes of wall time; no step timed out.

## Results

| arm | CAUGHT | MISSED | NO-READING |
|---|---|---|---|
| `virtual-malloc` | 02, 03, 08 | 01, 04, 05, 06, 07, 09, 10, 11 | none |
| `virtual-pg-pools` | 01, 02, 03, 04, 06, 07, 08, 10, 11 | 05 | 09 (unattributed) |

| case | `virtual-malloc` | `virtual-pg-pools` |
|---|---|---|
| 10 tuplestore double free | MISSED: reached, the one expected error, completed | CAUGHT: cause 25 in `SubletRelease`, by control (`control.sql` completed on the same image) |
| 11 reorderbuffer stale read | MISSED: reached, completed | CAUGHT: cause 25 in `ReorderBufferProcessTXN`, a declared fault site |

**Cases 10 and 11 read as on 2026-10-29, on both arms.** Compared field by field with
`../2026-10-29-virtual` (the two-operand instructions, dev's QEMU pin of that day), all 11 rows of each
arm are equal in verdict, cause, fault function and attribution route.

**Cases 01-09 read as the declared bundles.** `virtual-pg-pools` equals
`../2026-11-11-virtual/virtual-pg-pools` in all nine rows, and `virtual-malloc` equals
`../2026-10-18-review/virtual-malloc` in all nine rows, by the same four fields. Only offsets inside
the functions and the load bases differ.

## What this run does not settle

- `verdict_bundles` still names `2026-10-18-review` for `virtual-malloc` and `2026-11-11-virtual` for
  `virtual-pg-pools`, both nine-case bundles, so `case.json` of 10 and 11 carries these readings in its
  status line only. Naming this run instead would derive all 11 rows from one boot for each arm.
- `cheribsd-revocation` has not run cases 10 and 11.
- One boot for each arm. No repeat was taken.
