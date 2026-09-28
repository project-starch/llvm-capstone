# R-43 — R-35's revocation cache refused LIVE capabilities once ~256 revocation ids were live

**What it is.** R-35's fix (`capstone-ariane 4ad0df694`) lets an M-mode load/store through only when the
capability's exact 30-bit revnode id is resident in a 4-way × 64-set cache and marked live, and it
**denied on a miss**. That is safe (it never allows a revoked capability), but once a program holds more
than about 256 live revocation ids, entries are evicted and a perfectly LIVE capability is refused with
cause 25. Long-lived capabilities (a domain's own globals) are the first casualties.

**Siblings, so a reader with a neighbouring symptom is redirected:**
- **R-35** (`../R35-revoked-reference-retains-authority/`) is the defect whose fix caused this.
- **R-45** (`docs/ref/ISSUES.md`) is the revocation ORDERING window, found while testing this fix and closed
  in the same bitstream: a load/store issued right after REVOKE/DROP could be checked before the revocation
  took effect.
- **R-44** is the same optimistic adopt at the CPMP (S/U mode). It is not touched here.

## The problem, in one picture

```
   access via capability C (id c)
          |
          v
   +-------------------------------+        4 ways x 64 sets; set = id[5:0]
   | revocation cache (R-35 fix)   |        filled by passive taps on the rev-node's own node traffic
   |   hit, live  -> ALLOW         |
   |   hit, dead  -> DENY 25       |
   |   MISS       -> DENY 25   <---+---- R-43: more than 4 live ids in C's set evict C's entry,
   +-------------------------------+           so a LIVE capability is refused
```

On silicon (`caplifive_r42_6cbdaeeb4.bit`) this killed both R1 harness runs on their first invocation, and
the live128 / live512 sweeps, always on a long-lived globals capability that had been allowed a few
instructions earlier.

## The fix: on a miss, ASK the source of truth

```
   access via C, cache MISS
          |
          v
   LSU holds the access (the load/store unit only samples it in IDLE / SEND_TAG, so it really waits)
          |
          |  probe_req(index of C)          -- a new, lowest-priority, NO-RESPONSE rev-node endpoint
          v
   rev-node unit reads node[index]          -- the only place that knows the truth
          |
          |  the cache's EXISTING read tap sees that read and reports {generation, index, valid}
          v
   LSU resolves:  same 30-bit id AND valid  -> ALLOW (one-entry ALLOW record)
                  anything else              -> DENY 25 (one-shot DEAD record)
                  no answer within the bound -> DENY 25 (fail closed; a stuck rev-node cannot hang the core)
```

- **The ALLOW record** keeps an access that was allowed when accepted, allowed through the rest of its
  transaction, even if its cache entry is evicted meanwhile. Any invalidation of that index clears it.
- **The stall is built only from the lookup and registered state,** never from the revocation broadcast.
  The broadcast would close a combinational loop through the load path.

## R-45, closed in the same bitstream

```
   REVOKE r            ... walk runs in the rev-node ...          REVOKE commits (after the walk)
   ld x, 0(C)   <-- checked HERE, before C's node is written dead ---^
                    so it was ALLOWED                          R-45 fix: flush younger instructions at
                                                               REVOKE/DROP commit; the ld re-executes
                                                               after the revocation and is DENIED
```

## Evidence

- **RTL simulation:** `results/sim-query-on-miss.result-lines.txt`. It covers every arm and every
  deliberately broken build, each with its trace signature.
- **Synthesis:** `results/synth-*.result-lines.txt` (once run).
- **Board:** `results/board-*.result-lines.txt` (once run).

The pre-registered acceptance is in `docs/plans/r43-query-on-miss.md`.

## Reproduce

In a capstone-ariane worktree on branch `r43-query-on-miss`:

```bash
# cva6-build-rv container (docs/ONBOARDING.md); on apollo pin it: --cpuset-cpus=0-7,32-39
python3 cva6.py --testlist=../tests/testlist_r43.yaml --test r43-evict-live \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --isscomp_opts="+define+S12_MEM_DELAY=12+R43_TRACE" --sv_seed 1 --iss_timeout 7200 \
  --issrun_opts=+time_out=20000000
```

Read results from the CAPPRINT registers in the retirement trace, and from the `R43 ...` trace lines in
the `.log.iss`. `cva6.py`'s return code and its "SUCCESS" line carry no information here.
