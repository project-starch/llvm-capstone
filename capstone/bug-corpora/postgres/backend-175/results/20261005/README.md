# Results, 2026-10-05 -- the two Capstone arms

The 19 SQL cases run as a real `postgres --single` backend on the persistent
Capstone application VM, one fresh 16-byte-MAXALIGN cluster per attempt.
Summaries only; raw backend output is not committed (SCHEMA.md rule 6).

| arm | detected | differential | reached-recording | silent | not runnable | run | of |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spatial` | 2 | 4 | 1 | 9 | 3 | 16 | 16 |
| `sublet` | **4** | 3 | 1 | 8 | 3 | 16 | 16 |
| `cheribsd-revocation` (2026-10-04) | 3 | - | 1 | 10 | 2 not reached + 2 not run | 15 | 17 |

`not runnable` are cases 12, 14 and 18, each of which says in its own
`harness_limit` why: two need a postmaster, one needs a transaction block that
`postgres --single` does not provide. They are outside the denominator rather
than counted as silence.

## What changed between the arms

Exactly two cases, and both move the same way:

| case | `spatial` | `sublet` |
|---|---|---|
| `01_12a6206864a0_to_char_tz_overflow` | silent | **detected**, `cause 7` store-side bounds violation |
| `15_3d160401b65e_oidvector_oob_read` | differential (`errors 0/2`) | **detected**, `cause 5` load-side bounds violation |

Case 15 is the cleaner of the two. Its axis-1 class is `spatial/oob-read`,
assigned from the upstream report long before this run, and `cause 5` is
`RISCV_EXCP_LOAD_ACCESS_FAULT` -- a load-side bounds violation. The
classification and the hardware agree without either having been derived from
the other. On `spatial` the same defect was visible only as a wrong error
count; on `sublet` the access itself is refused.

## What this comparison does and does not license

**Licensed**: these two builds differ in the five Sublet memory-context
patches (`memorychunk-sublet-metadata-indices`, `allocset-sublet-context-
revocation`, `slab`, `generation` and `bump` lifetimes) and in nothing a case
can choose. Same source archive, same compiler, same musl tarball, same QEMU,
same cluster fixture, same VM, one run apart.

**Not licensed**: "the Sublet patch alone caused this". `build-domain.sh:81`
gives the sublet build a 64 MiB `CAPSTONE_APPLICATION_GRANT_BYTES` region that
the spatial build does not have at all, because the mechanism needs a region
to sublet from. So the two arms differ by the patch *and* by the presence of
that region. The direction is the safe one -- the protected arm has strictly
more memory, so a fault there cannot be attributed to starvation -- but
separating the two would need a third arm with the grant region and without
the patch, which was not built.

**Neither build logs its own memory configuration.** The arena and grant sizes
above are read from the build script, which is version-controlled and
identical in both trees, not from a record of either run. That is weaker than
a measurement and is stated here rather than left to be assumed.

## Reproducing

    ARM=spatial  python3 ~/arms/postgres/capstone/run-pg-appvm.py
    ARM=sublet   python3 ~/arms/postgres/capstone/run-pg-appvm.py

The runner records the image's sha256 beside each result, so the arm is
established by the binary that ran and not by the label passed in. It refuses
to score a case whose backend prompt never appeared, and refuses to score at
all if no case produced output -- a parse failure is not a measurement of zero.
