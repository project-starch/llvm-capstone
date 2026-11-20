# PoisonCap and Sublet on the 19 temporal SQLite cases of the paper's Table 2 (2026-11-18)

`../2026-10-26-poisoncap-heap256/` compared the two systems on the temporal set as it stood on
2026-10-26: 17 cases. The corpus was revised on 2026-10-29, and the paper's Table 2 lists 19 SQLite
cases:

    Table 2, 19 cases:        02 03 05 06 07 10 11 12 13 14 15 16 17 19 20 21 22 23 24
    2026-10-26, 17 cases:     00 03 05 06 07 10 12 14 15 16 17 19 20 21 22 23 24

- Cases 02, 11 and 13 are in the 19 and had no reading on PoisonCap with their present triggers.
  Case 02's trigger was corrected on 2026-10-26 and those of 11 and 13 on 2026-10-29 (`case.json`
  of each). The 2026-10-26 runs built all three from the earlier sources; its verdict files mark
  11 and 13 "no qualified trigger" and show 02 with the old trigger's NULL dereference.
- Case 00 leaves the count. The corpus classes it `spatial`, and both systems reject it by bounds.

This record runs 02, 11 and 13 on both systems in the configuration of `driver-la` and
`v6-sublet-block` of 2026-10-26 and tabulates all 19. The other 16 cases were not re-run: both
platforms and both build chains are the ones of 2026-10-26 (next section).

## Result, 19 cases

| | rejected at site | in quarantine only | neither |
|---|---|---|---|
| PoisonCap, published artifact, driver configuration | **14** | 4: 11, 15, 16, 17 | 1: 12 |
| Sublet, block port, same configuration | **19** | none | none |

Verdict rule as before: a case counts as rejected when the process faults in one of the case's
`fault_sites`. "In quarantine only": PoisonCap does not reject the case, and the stale object's
block was freed and still in PoisonCap's quarantine at the stale access. Every run was made twice
and the two runs agree in every case. `tabulate.out` holds the table of all 19 cases
(`scripts/tabulate.py`, which reads this record and `../2026-10-26-poisoncap-heap256/`).

Numbers for the paper: `\qeSecurityPoisoncapCases` 19, `\qeSecurityPoisoncapRejected` 14,
`\qeSecurityPoisoncapQuarantineOnly` 4, `\qeSecurityPoisoncapModelMisses` 4.

## The three cases

| case | PoisonCap (`c19-driver-la`, `-rep2`) | Sublet (`c19-sublet-block`, `-rep2`) |
|---|---|---|
| 02 | fault at site `sqlite3BtreePager`, SIGPROT (34), si_code 3, poison lines on the console | fault at site `sqlite3BtreePager`, cause 25 |
| 11 | no fault and no return within 300 s, no poison line | fault at site `memcmp`, cause 25 |
| 13 | fault at site `sqlite3Fts5PoslistNext64`, SIGPROT (34), si_code 3, poison lines on the console | fault at site `sqlite3Fts5PoslistNext64`, cause 25 |

**Case 02.** The review of 2026-10-24 noted a fault at its site on both systems without a temporal
signature, a tag fault on a NULL `pPage1`. That reading belonged to the trigger before its
correction (si_code 2 in `sqlite3BtreeUpdateMeta`, still visible in
`../2026-10-26-poisoncap-heap256/verdicts/driver-la/verdicts.md`). With the corrected trigger the
case has a temporal signature on both systems: PoisonCap reports si_code 3 with the poison
exception for a 128-byte block (`poison base 506a2a80`), in `sqlite3BtreePager`, the function host
ASan names for the stale read of the freed 72-byte `Btree`; Sublet reports cause 25, access through
a revoked capability, in the same function.

**Case 13.** PoisonCap reports the poison exception for the 64-byte poslist buffer
(`poison base 506bbfc0`) in the lookahead reader; Sublet reports cause 25 in the same function.

**Case 11 on PoisonCap.** The process neither faults nor returns: 300 s in both scored runs and
1,200 s in `c19-long-la`. The diagnostic amalgamation (`c19-diag11-la`, hooks FREE, FLUSH and the
new FTS3CMP of `make-diag-amalgamation.py`) shows the sequence:

1. Before `optimize()`, `fts3SegReaderCmp` compares the pending term at `0x506ac550` with the
   committed segment's term: `memcmp` over 3 bytes returns 0.
2. `optimize()` frees the pending-terms hash. PoisonCap's `memsys5Free` poisons and quarantines
   the term's 16-byte block (`DIAG FREE base=0x506ac550 size=16 q=602`).
3. `matchinfo()` restarts the readers, and `fts3SegReaderCmp` compares the same two terms again
   through the stale capability, whose tag is still set. `memcmp` completes and returns -31: the
   read returns the bytes the free left there. No fault and no poison line.
4. The process then stays in the row walk of `fts3EvalGatherStats` (two samples in
   `runs/c19-long-la/stack-samples.txt`: user time only, no system call, program counter in
   `fts3EvalNextRow` under `fts3EvalGatherStats`). `case.c` describes this walk as one that does
   not end on a system that does not stop the read.

The block was in the quarantine at the stale read: the run frees 736 blocks, fewer than the 4,096
entries that flush the quarantine, and prints no flush. So case 11 counts as in quarantine only.

The cause is the address mask of their QEMU model, the symptom recorded for cases 16 and 17
(`../2026-10-26-poisoncap-heap256/README.md` and the bit-4 controls of the kit branch
`diag-small-poison`): the stale term is a 16-byte block whose address `0x506ac550` has bit 4 set.
The cross-check used for 16 and 17 holds here too. With minimum request 64 and everything else
unchanged (`c19-min64-la`, not scored), PoisonCap faults at the site: SIGPROT, si_code 3, in libc
`memcmp` at offset 0x1b095c, the offset of cases 16 and 17 at that minimum, with the poison
exception for a 64-byte block.

## Same platforms and builds as 2026-10-26

The 2026-10-26 record holds hashes for the Sublet platform only, so the PoisonCap side is checked
through a build.

- **Sublet platform.** The six hashes of `platform-v6-heap256` equal
  `../2026-10-26-poisoncap-heap256/sublet-platform.sha256` (`sublet-platform.sha256` here). CPU
  string as recorded there (`runs/c19-sublet-block*/cpu.txt`).
- **PoisonCap artifact.** QEMU, kernel, disk image, compiler and published amalgamation carry
  modification times of 2026-10-24, before the 2026-10-26 runs. Case 03, whose source did not
  change, built today to the byte-identical binary of `builds/driver-la` of 2026-10-26, and it
  faults at the same program counter (`build-equality.sha256`). The `uaf_read` control faults at
  the program counter of 2026-10-26 with the same poison lines. `poisoncap-platform.sha256` records
  today's hashes for later checks. `build.json` equals that of `driver-la` in compiler, target,
  amalgamation and heap configuration.
- **Sublet build chain.** Two inputs of `sdk-v6` had been deleted from the host since 2026-10-26:
  the compiler `~/spec-kit/toolchain/build` and the libc archive under `~/v6gate`. The compiler is
  rebuilt from the recorded source archive with the recorded recipe (`spec-kit/kit/recipes/
  toolchain.sh`), and its sha256 equals the one in `toolchain-record.txt` of 2026-10-20. Of the
  libc archives left on the host, `~/sdkself/sdk-branch/libc-capstone.a` reproduces the
  2026-10-26 build: with it, case 03 links to the byte-identical image of `builds/v6-sublet-block`,
  and the SQLite engine object is byte-identical too (`build-equality.sha256`). `sdk-v6-restored`
  is `sdk-v6` with its `libc` entry pointing at that archive.

So step 2 of the plan needed no re-run of the 16 earlier cases.

## Configuration

| run | system | heap | minimum request | scored |
|---|---|---|---|---|
| `c19-driver-la`, `-rep2` | PoisonCap, published artifact, their memsys5 | 256 MiB from malloc | 16 | yes |
| `c19-sublet-block`, `-rep2` | Sublet, block port, platform v6 | 256 MiB from malloc | 16 | yes |
| `c19-long-la` | PoisonCap, case 11, timeout 1,200 s | 256 MiB from malloc | 16 | no |
| `c19-diag11-la` | PoisonCap, diagnostic amalgamation, case 11, timeout 600 s | 256 MiB from malloc | 16 | no |
| `c19-min64-la` | PoisonCap, case 11 | 256 MiB from malloc | 64 | no |

Lookaside on, C-library revocation off (`libc_revoke_enabled=0` in every gate), guest booted
single-user with `security.cheri.runtime_revocation_default=0`, as on 2026-10-26. The control gate
of `run-cases.py` passes in every PoisonCap run (`runs/*/run.json`). The Sublet runs carry the two
standing controls and the four realloc and capacity controls of the block port, with the readings
of 2026-10-26 (`same_address=1 kept=1`, cause 25 in `control_read`). The cases print
`heap_bytes=268435456` on both systems.

The cases are built from `capstone/bug-corpora/sqlite/engine-repros/` of `dev` at `fe1d2dad2ebe`.
The kit (`kit-log.txt`) is the heap256 kit of 2026-10-26 at its head `4789284`, with
`capstone/ports/sqlite` and `capstone/bug-corpora/sqlite` replaced by those of `fe1d2dad2ebe`.
Runtime headers and the staged runner stay those of the heap256 kit, which match the prebuilt SDK
and platform.

## Notes on the runs

- A first attempt started three PoisonCap guests at once (a scored run, the long run and a
  diagnostic run). All three failed before or at the control gate (payload not unpacked,
  revocation default not disabled, ssh banner timeout) and are void. Every run in this record had
  the PoisonCap guest to itself.
- `scripts/c19-poisoncap.sh` was stopped after the second scored run. Its last two steps did not
  run; `c19-diag11.sh`, `c19-long.sh` and `c19-min64.sh` replaced them.
- An earlier session of the same day ran the three cases once from the corpus at `ea9e89bea`
  (runs `c3-driver-la`, `-rep2` on the host). Its PoisonCap readings equal the ones here. Its
  Sublet build failed on the deleted compiler. It is not part of this record.
- During `c19-long-la` the process was sampled with `gcore` twice. The run is not scored.

## Limits

QEMU only, on both sides. Function-level sites. Case 11's stale read is observed on the diagnostic
amalgamation. The scored build shows the same outcome, no fault and no return, and the address of
its stale read was not recorded. Case 15's place in "in quarantine only" rests on `../2026-10-24-poisoncap/EVIDENCE.md`,
and that of 16 and 17 on the diagnostic runs of 2026-10-26.

## Files

`runs/<run>/`: `results.jsonl`, `build.json` of the build it ran, and `run.json` (PoisonCap) or
`cpu.txt` (Sublet). `verdicts/<run>/`: PoisonCap's scoring (`corpus/score-cases.py`). `scripts/`:
the drivers as run on the host and `tabulate.py`. `tabulate.out`: all 19 cases.
`sublet-platform.sha256`, `poisoncap-platform.sha256`, `build-equality.sha256`: the hashes named
above. `kit-log.txt`: the kit's commits.
