# Spatial corpus status

Generated numbers come from `spatial-allocator-corpus.csv`; run
`python3 verify-spatial-corpus.py --self-test` before trusting any of them.

| | rows | certain | measured in the domain |
|---|---|---|---|
| spatial | 31 | 22 | 3 |

## Measured

| id | issue | stock | domain |
|---|---|---|---|
| `MPY-S01` | #19314 | SIGSEGV | **fault, cause 7** — bounds `0x60000` = the whole 384 KiB heap |
| `MPY-S05` | #15271 | not run | **untrapped** — eight writes past a 4096-byte allocation completed |
| `MPY-S31` | #19129 | not run | **untrapped** — the alloca fallback runs; the port's own stack check ends the row |

The first trapped defect in either corpus, and the bounds width is why it is not
the good news it looks like: the write was stopped at the far edge of the REGION,
not at the end of the object it overran, so the entire heap was overwritten on
the way there. Detail in `cases/MPY-S01_sequence-repeat-size-overflow/RESULT.txt`.

## Why the rest are not measured, and what each would need

`MPY-S05` was reached by REVERTING its fix on the pinned tree rather than building
a parent. That is available only when a fix is one hunk with no second hardening
elsewhere, and it was checked for `S02` and `S03` too. Both failed, and not for
want of effort:

- `S02` (#13041, zero-length `int.to_bytes`): `py/binary.c`'s `mp_binary_set_int`
  now clamps `val_sz` to `dest_sz`, so a zero-length destination writes nothing
  even with the overflow check reverted. The 2023 defect needs the 2023 tree.
- `S03` (#13007, float read as int in `slice.indices`): under
  `MICROPY_LONGINT_IMPL_NONE` `mp_obj_int_get_checked` is `MP_OBJ_SMALL_INT_VALUE`,
  a shift of the tagged word with no memory read at all. The out-of-bounds read
  cannot occur in this configuration however the source is arranged.

Both were established by reading the source before spending a build.

| reason | count | rows |
|---|---|---|
| parent build needed, otherwise reachable | 1 | `S04` |
| not reproducible in this port's configuration | 2 | `S02` `S03` |
| needs a VFS + block device | 5 | `S07` `S08` `S09` `S16` `S17` |
| needs `.mpy` loading (`MICROPY_PERSISTENT_CODE_LOAD` is 0) | 4 | `S10` `S11` `S12` `S13` |
| needs modules absent here (ssl, machine) | 2 | `S14` `S15` |
| needs `uctypes`, which faults architecturally here | 1 | `S06` |
| needs `mpz` (`MICROPY_LONGINT_IMPL` is NONE) | 1 | `S18` |
| already fixed at the pin, no trigger form found | 1 | `S19` |
| not applicable to a 32-bit-`int` target | 1 | `S20` |
| 2014 tree, parent build not worth the archaeology | 1 | `S21` |

`S04` is the one cheap row left: it needs only the `re` module, which this domain
already carries, and a parent build of the 2023 tree.

## The scope split, which is the point of this corpus

| scope | certain rows | prediction | why |
|---|---|---|---|
| `gc-heap` | 16 | not trapped, or trapped only at the region boundary | the block is one offset into the single heap array |
| `static-global` | 4 | trapped | `-capstone-gp-captable` carves each global separately |
| `stack` | 1 | trapped | the domain stack carries its own capability |

The last two are the corpus's built-in positive controls: if they came back
untrapped, the instrument would be broken rather than the target. **Neither has
been demonstrated yet, and `MPY-S31` shows it is harder than it looked.** That row
reaches the defect — the alloca fallback runs untrapped — but MicroPython's own
`mp_cstack_check` ends recursion at depth 8, guarding the C stack at 393 KB while
the stack capability's bound sits near 800 KB. The port stops the descent before
the hardware can. Testing the `trapped` half therefore needs either a
`static-global` row (all four still blocked on a VFS or absent modules) or a build
with the port's stack guard relaxed, which changes the configuration under test.

## Not counted

Nine rows are `is_spatial` `uncertain` or `no` and are kept visible with the
reason rather than deleted. `S22`–`S26` are reports that describe two different
outcomes, or fix index arithmetic without establishing an out-of-bounds access;
`S27`–`S30` matched the search on the word "overflow" and are a padding bug, two
feature requests and a feature PR.
