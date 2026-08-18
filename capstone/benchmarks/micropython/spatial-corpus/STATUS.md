# Spatial corpus status

Generated numbers come from `spatial-allocator-corpus.csv`; run
`python3 verify-spatial-corpus.py --self-test` before trusting any of them.

| | rows | certain | measured in the domain |
|---|---|---|---|
| spatial | 30 | 21 | 1 |

## Measured

| id | issue | stock | domain |
|---|---|---|---|
| `MPY-S01` | #19314 | SIGSEGV | **fault, cause 7** — bounds `0x60000` = the whole 384 KiB heap |

The first trapped defect in either corpus, and the bounds width is why it is not
the good news it looks like: the write was stopped at the far edge of the REGION,
not at the end of the object it overran, so the entire heap was overwritten on
the way there. Detail in `cases/MPY-S01_sequence-repeat-size-overflow/RESULT.txt`.

## Why the rest are not measured, and what each would need

| reason | count | rows |
|---|---|---|
| parent build needed, otherwise reachable | 4 | `S02` `S03` `S04` `S05` |
| needs a VFS + block device | 5 | `S07` `S08` `S09` `S16` `S17` |
| needs `.mpy` loading (`MICROPY_PERSISTENT_CODE_LOAD` is 0) | 4 | `S10` `S11` `S12` `S13` |
| needs modules absent here (ssl, machine) | 2 | `S14` `S15` |
| needs `uctypes`, which faults architecturally here | 1 | `S06` |
| needs `mpz` (`MICROPY_LONGINT_IMPL` is NONE) | 1 | `S18` |
| already fixed at the pin, no trigger form found | 1 | `S19` |
| not applicable to a 32-bit-`int` target | 1 | `S20` |
| 2014 tree, parent build not worth the archaeology | 1 | `S21` |

The four `parent-build` rows are the cheap ones and are all pure `py/` or a module
this domain already carries: `S02` is three lines of Python, `S04` needs only the
`re` module, `S05` needs no module at all.

## The scope split, which is the point of this corpus

| scope | certain rows | prediction | why |
|---|---|---|---|
| `gc-heap` | 16 | not trapped, or trapped only at the region boundary | the block is one offset into the single heap array |
| `static-global` | 4 | trapped | `-capstone-gp-captable` carves each global separately |
| `stack` | 1 | trapped | the domain stack carries its own capability |

The last two are the corpus's built-in positive controls: if they came back
untrapped, the instrument would be broken rather than the target. Neither is
measured yet — every row in both is blocked on a VFS, `.mpy` loading, or a module
this domain does not have.

## Not counted

Nine rows are `is_spatial` `uncertain` or `no` and are kept visible with the
reason rather than deleted. `S22`–`S26` are reports that describe two different
outcomes, or fix index arithmetic without establishing an out-of-bounds access;
`S27`–`S30` matched the search on the word "overflow" and are a padding bug, two
feature requests and a feature PR.
