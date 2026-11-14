# gp's cursor is __global_pointer$ in every context: directed test

Linker gp relaxation (llvm-capstone compiler-gp-relax: `-capstone-mark-gp-relaxable`, lld `--relax-gp`) rebases a
global access on gp itself (`ld rd, x - __global_pointer$(gp)`). Such code is correct only if gp's cursor is
`__global_pointer$` in every context, so the transport lock's identity no longer lives in it (delegate.c, "THE
OWNER'S IDENTITY": single-context mode, then service identities after the first create).

**What it checks** (`gpcur.c`). Each check reads gp's cursor with `movc` and compares it with
`__global_pointer$`'s cursor. That cursor is formed pc-relative, so it does not depend on gp's own. Each access
check writes small globals by name and reads them back through their address. In the gp arm the by-name access is
gp-relative; the address path is not.

| check | where |
|---|---|
| main_cursor, main_access | at main |
| longjmp_cursor | after a longjmp out of a nested call |
| handler_cursor, nested_cursor, handler_access | in a handler on the alternate signal stack, and in one nested in it (SA_NODEFER) |
| handler_create_refused | single mode: a context create inside a handler fails with EDEADLK, specifically; pthread_create fails; the mode stays 0 (contract test 3) |
| preempt_cursor | across 4096 delegated calls, 256 fresh-page faults and a 2e8-iteration spin |
| single_mode | identity mode 0 with no context created (contract test 1) |
| service_mode | -DGP_MULTI: mode 1 after the first create |
| child_cursor, child_access | -DGP_MULTI: in the second context |
| contention | -DGP_MULTI: two live contexts write 2048 records each, concurrently, through the transport; every record arrives whole and once (contract test 2) |

Signal delivery runs inside the delegated round (delegate.c dl_settle), so a handler always runs with its context
holding the wire. That is what makes check 3 deterministic: `raise()` delivers before its own call releases.

**Controls.**
- `-DGP_MUTANT` moves gp's cursor to the image base at main's start, where a skipped set point would leave it.
  - gp arm: every cursor check fails, and the relaxed stores land elsewhere.
  - control arm (no relaxation): the cursor checks fail and the access checks pass, because without relaxation no
    access reads gp's cursor.
- The gp arm's binary must contain gp-based accesses, and the control's none (objdump count, gate RESULTS).
- Contract test 1 (single-context mode, no service exit for the identity) is checked by `single_mode`. It is not
  measured as an exit count on the native route: no second context ever exists there, so service mode never applies
  past the first create attempt. That attempt still switches the mode (`service_mode` in the multi build).

**Where each check applies.** The native route, where the SPEC images run, has exactly one context per task:
- its kernel refuses caught handlers (`capstone_native_signal_action`, class 2: EOPNOTSUPP);
- it refuses further contexts (the create service fails, and musl reports EAGAIN).
So there the handler and context checks cannot apply. The program reports each such group as `GP_NA` with that
exact reason, the judge requires exactly those reasons, and any other setup failure is a `GP_SETUP_FAIL`, a failure.
Those checks apply where handlers and contexts exist (the virtual route). This SDK lineage (application ABI 2)
does not run on the virtual kit at hand (kit-prof's capstone-vexec faults its first instructions, cause 24, with
native IO on or off), so they are not measured here.

**How it runs.** The native route, as the SPEC images do: built with the arm SDK's capstone-cc and the arms' flags,
staged in a copy of the native mcf stage with its binary replaced, run on the native twin
(`perf/faultesc/run.sh`); stdout is `results/inp.out`. The gate script and its judge are
`c1-gate/gt2/gate-gp/gate-gp4.sh` and `judge-gp4.py` (pass 5), local to the build host.

## Results (2026-11-14, native route, natio twin kernel)

The builds:
- arms GPC (control: llvm-c10 81193d42edb0, flags off) and GPR2 (`-capstone-mark-gp-relaxable
  -capstone-small-data-limit=16 -global-merge-min-data-size=17`, lld `--relax-gp`);
- both on the gp SDK (this branch at ea3ab7249002) and this branch's link script.

| build | GPC | GPR2 |
|---|---|---|
| gpcur | 5/5 PASS; handler checks n/a (EOPNOTSUPP) | 5/5 PASS, the same n/a; 49 gp-based accesses (GPC 0) |
| gpcur -DGP_MULTI | 5/5 PASS (service_mode included); context checks n/a (EAGAIN) | the same; 47 gp-based accesses |
| gpcur -DGP_MUTANT | cursor checks FAIL, access checks PASS | the relaxed stores land elsewhere: SIGSEGV, no GP_DONE |

The hosted contract reads the same 45 check lines (37 PASS) on GPC, GPR2 and FRCDKMS.

Withdrawn on the way:
- An escape check that opened /dev/null read FAIL on both arms. The native grant needs `dom_null` for /dev/null,
  and this kernel predates it. The check now writes a file in the run directory, and an environment failure is a
  setup failure.
- An early build took main's globals' addresses in main itself. The shared base was not relaxable, so the test read
  no gp-based access in main.
