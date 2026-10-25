# The quantum's zero budget: result lines for every value, and S-26's RTL half

*RTL lane, 2026-10-19. The auditor on the S-26 candidate confirmed the mechanism from source but could not find
this arm's source or results in any ref it searched, and read the result only as prose. These are the lines.*

**The source.** `capstone-ariane`, branch `virtual-profile-2-c`, commit **`9b272ff2d`**, file
`verif/tests/custom/capstone/va-satpstale.S`, with every build registered in `verif/tests/testlist_sup.yaml` at
the same commit. It is a submodule of this repository, which is why a search of `dev` alone does not find it. The
arm's own header carries the pre-registration for each build and the result beside it.

**The builds.** All carry `-DFENCE_BEFORE -DRESUME_PC`; the quantum arms add `-DQUANTUM_ESC` and a `-DQUANTUM_N`
value, except `bqb` which uses `-DBIG_QUANTUM` (0x100000) and `q1k-nf` which omits `-DFENCE_BEFORE`.

**Reading positions.** `-br`: the caller's resume marker, then the context's probe value, the escape kind, the
cause, the caller's post-escape read, the pass marker. The `QUANTUM_ESC` builds insert the loop counter and a
second zero slot after the marker, so their order is: marker, counter, probe value, kind, cause, caller's read,
pass marker. All values as printed, hexadecimal.

| build | quantum | retired | harness | readings |
|---|---|---|---|---|
| `va-satpstale-b` | 0x100000 | 28984 | SUCCESS | 0xaaaa 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-br` | 0x100000 | 28988 | SUCCESS | 0xaaaa 0x7a 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-q0` | **0** | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-bq` | 40 | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-bq60` | 60 | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-bq100` | 100 | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-bq400` | 400 | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-bq1k` | 1,000 | 28987 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-q1k-nf` | 1,000, no fence | 28986 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-q1500` | 1,500 | 28989 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 **0x1** 0x0 0xaaaa 0x55 |
| `va-satpstale-q2000` | 2,000 | 29030 | SUCCESS | 0xaaaa 0x7a **0xa** 0xbbbb **0x1** 0x0 0xbbbb 0x55 |
| `va-satpstale-q2500` | 2,500 | 29032 | SUCCESS | 0xaaaa 0x7a 0xa 0xbbbb **0x3** 0xb 0xbbbb 0x55 |
| `va-satpstale-q3000` | 3,000 | 29032 | SUCCESS | 0xaaaa 0x7a 0xa 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-q3500` | 3,500 | 29032 | SUCCESS | 0xaaaa 0x7a 0xa 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-bq4k` | 4,000 | 29032 | SUCCESS | 0xaaaa 0x7a 0xa 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-bqb` | 0x100000 | 29031 | SUCCESS | 0xaaaa 0x7a 0xa 0xbbbb 0x3 0xb 0xbbbb 0x55 |
| `va-satpstale-bf4` | page-fault kind, options 3 | 28994 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 0x2 **0x18** 0xaaaa 0x55 |
| `va-satpstale-bf4c` | page-fault kind, options 1 | 28994 | SUCCESS | 0xaaaa 0x7a 0x0 0x0 0x2 **0x18** 0xaaaa 0x55 |

Every arm ran with `--sv_seed 1` and `+define+S12_MEM_DELAY=12+SUP_TRACE`, zero exceptions in the retirement
trace, on `bf3db3175`, which is tree A's parent and the RTL the vp2 bitstream was built from.

## What the lines say

- **A zero budget retires nothing.** `q0`: counter 0, kind 1, cause 0, and the caller's marker present. The
  context executed no instruction; the retired instruction after the CSRUNV in the trace is the caller's own
  marker store at CSRUNV+4.
- **Every budget below the startup cost is bit-identical to zero.** 0, 40, 60, 100, 400, 1,000 and 1,500 all give
  the same readings, and all but `q1500` and `q1k-nf` the same retired count of 28987 (those two differ by the
  fence and the loop's own instruction count, not by behaviour).
- **The startup cost is between 1,500 and 2,000 cycles**, charged to the context's own budget. At 2,000 the loop
  completes, counter 10, and the quantum still expires before the context's ecall, kind 1. At 2,500 and above the
  context escapes by its own ecall, kind 3 cause 11.
- **Kind 1 with cause 0 is the quantum's signature**, not a generic interrupt: the quantum's cause is
  0x8000000000000010 and register id 71 masks it to zero in the frame (`csr_regfile.sv:497`, tval at `:499`).
- **The cold page-table walk is not the cost:** `q1k-nf`, the same 1,000-cycle budget without the pre-entry fence,
  retires nothing too.

## Scope

This establishes the **RTL half** only: a context entering with a zero budget retires nothing and escapes kind 1,
with the caller resuming at CSRUNV+4. Whether that is S-26's cause depends on the software half, that nothing
writes CSR 0x7C3 before the first virtual context, which is the compiler lane's reading and is being audited.
Noted from the runtime lane: on the board firmware the 0x7C3 writer is the **process-ABI STEP path through
`supervised_invoke`**, not a classic domain call, so the workaround's mechanism needs stating in those terms.

The page-fault kind is **not read**: its base was an integer, so the capability check refused the load with cause
24 (0x18) before translation was attempted, and both options arms therefore read kind 2 correctly, since 24 is not
in the kind table's re-kinded set. It needs a capability minted over the unmapped range.
