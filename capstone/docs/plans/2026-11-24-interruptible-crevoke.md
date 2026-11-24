# Interruptible CREVOKE (programme Line B step 4(b)): specification

Status: draft for the rtl lane, 2026-11-24. Nothing built. Revises the semantics of
`capstone/docs/design/sublet-reference-model.md` (32dde56a6b42) §4.3 and closes the gap that the model's §9 records:
"The model defines no interleaving of a long CREVOKE with another hart, a device or an interrupt." The submitted paper
calls CREVOKE neither atomic nor interruptible (paper `main`, grep for interrupt and atomic). Section 5's wording, if any
is wanted, is the lead's.

## Why

A CREVOKE runs to its end, and nothing interrupts it.
- Measured on P1's RTL: 720,892 cycles at 65,532 nodes, at memory delay 0.
- On silicon F, a 100,000-node subtree takes 2.82 M cycles. PostgreSQL's largest context reset, 34,509 nodes, takes
  about 39 ms.
- The Sublet unit of B5b-1 (capstone_sublet_life.sv, b5b1-step3 8446ce50c) walks the run twice, a check pass and a
  commit pass, and has no interrupt input (:33: "A request, once accepted, runs to its end").

So the worst interrupt latency the lifetime unit adds is one whole revoke, and it grows with the subtree.

## What is retracted on the way

The first direction, sent to the rtl lane the same day, released the run from its FRONT in chunks. It is withdrawn.
- The run is in depth-first preorder (§3.4). A prefix of it can end inside a subtree.
- Example: run c1(d+1) g1(d+2) g2(d+2) c2(d+1). A chunk that releases c1 and g1 leaves g2 live with its parent dead.
  No sequence of legal instructions reaches that state.
- If the context never resumes, for example when the task is killed in the handler, the orphan stays. A later CDERIVE
  from r inserts new(d+1) directly after r, and g2 then reads as new's child.

## The rule

**Release a suffix of the run, from its end backward.**
- In preorder every descendant follows its ancestor, so any suffix of the run is a union of complete subtrees.
- After any number of chunks, the table equals the table after legal CREVOKEs of those subtrees' roots, in some order.
- Every intermediate state is therefore one that legal instructions can reach. That holds whether or not the context
  ever resumes.

## Semantics

Names: o is the operand, r its record, d = r.depth. The run is the records after r with depth > d (§3.4). `after` is
the first record past the run, or none.

```
CREVOKE rs1                      (replaces §4.3; the checks before the walk are unchanged)
o = x[rs1]; !o.tag: 24; table invalid: 29; !live(o.id): 25; r = record of o.id
locate the run's last record t        (the scan below; a saved position may resume it)
repeat:
    take up to K records ending at t, walking previous links: each must have depth > d,
        and a record with depth 0, a link of 0, or a walk that reaches a record with
        depth <= d other than r: 29, before any release in this chunk
    release them in reverse list order (3.4); let p = the record before the chunk
    splice: p.next = after; if after: after.prev = p
    t = p
    if t == r: break
    if an interrupt is pending and enabled: end the instruction without retiring it
        (pc stays on the CREVOKE), and take the interrupt
r.prev.next = after; if after: after.prev = r.prev; release r
```

- **Restart.** A re-executed CREVOKE starts again from its checks. It finds the remaining run, which is shorter, and
  continues. Restart is idempotent.
- **What a handler may do in between.**
  - A CDERIVE from o or from a run member inserts inside the run, before `after`, so the revoke releases that record.
    That is correct: the record is o's descendant.
  - A CREVOKE inside the run shortens the run.
  - A CREVOKE of o, or of an ancestor of o, makes the restarted instruction raise 25. That is the same as the handler
    having run before the CREVOKE began.
- **Forward progress.** The interrupt is looked at only after a chunk commits, never before the first chunk. Every
  execution therefore releases at least one chunk, and the total work is bounded by the records that exist.
- **Corruption.** A free record has depth 0 and previous 0; a retired one is {2^32-1, zeros}; a run member always has
  depth > d >= 0. So a free or retired record met in the run is 29, and so is a 0 link. Within one execution, visits
  still count against next, as today.
- **The one weakening against today.** A 29 found in chunk j leaves chunks 1 to j-1 released and r live. Today the check
  pass makes a 29 all-or-nothing. That state is still a legal one, by the suffix rule above.

## Finding the end on restart: the saved position

Locating t from r means scanning the run forward, which is O(remaining) reads. If every execution starts there, one
execution's latency is O(n) again. Progress still holds, by the first-chunk rule, but the latency bound does not.

The scan itself must also be interruptible. It is read-only, so an interrupt during it loses nothing. But without a
saved position, each restart scans again from r.

**The unit keeps one saved position.** Its fields:
- the operand id;
- the root (`srevroot`);
- a phase, scan or release;
- a record id with its generation: in the scan phase the last record scanned, in the release phase t, the run's
  current last record;
- in the scan phase, the visit count.

It is set when the unit stops the instruction for an interrupt.

**Why t, not `after`.** The saved record is t, the run's last record, rather than `after`. A run can reach the end
of the list, and then `after` is none and has no id whose liveness could be checked. t always exists until the run is
empty: it is a run member, or r itself.

**Use on re-issue.** The saved position is used only if all of these hold:
- the operand id and the root match;
- the saved record is live under its full id;
- in the release phase, the record after t is none or has depth <= d.

A CDERIVE from t inserts after t, and the unit handles it by walking forward from t while depth > d before it walks
back. Otherwise the CREVOKE starts fresh from its checks, which is correct and only slower.

**Stale cases**, all handled by starting fresh:
- the handler revoked the saved record, or its slot was reused (the generation differs);
- `srevroot` changed (the key);
- another CREVOKE was stopped in between and replaced the position (single entry).
  This includes the handler's own CREVOKE answering partial.

**Architectural or not.** Two options:
- **A: microarchitectural (recommended first).** No new CSR and no kernel change. The latency bound holds while one
  stopped CREVOKE resumes before another one stops. A context switch that stops a second CREVOKE costs the first one a
  rescan.
- **B: architectural, vstart-like.** A CSR holds the position, and the kernel saves and restores it on a switch. The
  bound then holds across switches, at the cost of an ABI field and kernel work. It is the kernel's to specify if A
  proves insufficient.

A is chosen first because it changes nothing a program or kernel can see. Every restart is correct without the saved
position.

## Latency target and K

- **Target:** with the saved position valid, the latency CREVOKE adds is at most 5,000 cycles, about 200 µs at 25 MHz.
- **K:** the largest chunk whose check and commit fit that target. The rtl lane derives it from 2a's measured cycles per
  record.
- **The bound:** latency stays at most K records' work, plus one splice, plus one forward step past t.
- **K is a localparam.** A wrong K costs a rebuild, not a redesign. The rtl lane's P1 figure, about 11 cycles per
  node at delay 0, suggests K in the low hundreds. A value from simulation is only a first estimate, because simulation
  memory delay is not silicon cost.
- **On silicon.** The first bitstream that carries 4(b) pre-registers a reading of the worst-case latency on silicon.

## Hardware shape (rtl lane's reading, 2026-11-24)

- One registered "interrupt pending and enabled" input, with the same predicate commit uses, debug request included. It
  is sampled only after a chunk commits.
- A new unit status, partial. ex_stage turns it into a replay of the same pc, through the refetch path the R-43 marker
  already uses. The interrupt is then taken at the normal instruction boundary with epc = the CREVOKE.
- No new exception cause.
- With no interrupt pending, the walk runs to the end at today's cost, plus one splice per chunk.
- The saved position adds about 3 ids, a count and 2 bits of state to the unit.

## Reference model and tests

- QEMU's walk_run (capstone-qemu c1985bb529, `cap_rev_tree.c:258-291`, `cap_rev_kill :232-246`) becomes the chunked
  backward loop. A test property forces a stop after every chunk, since QEMU takes no interrupt inside a helper. It runs
  the same instruction to completion over repeated re-executions.
- Tests in the model's §8 style:
  1. the chain and fan shapes at N = 1, K, K+1 and 65,532, stopped after every chunk, give the same final table as one
     uninterrupted revoke;
  2. a handler CDERIVE from o, from a run member and from t between chunks: every new record is released at the end;
  3. a handler CREVOKE of o, and of an ancestor of o, between chunks: the restart raises 25;
  4. a free record spliced into the run: 29, with the chunks before it released and the table valid;
  5. a kill between chunks (no restart): the table passes §3.6 validation, and every live record's ancestors are live;
  6. the saved position stale (revoked t, reused slot, changed srevroot): the restart completes from a fresh scan.
- RTL: the same tests as directed `.S`, each with an interrupt arm that stops after chunk 1, chunk j and the last chunk.
  Plus a latency arm that reads the cycles from interrupt assertion to trap entry at N = 65,532.

## Open

- K and the measured per-record cost on 2a.
- Option B only if option A's bound proves insufficient on a real workload. The case to measure is a revoke stopped
  repeatedly while another context switches in and stops its own revoke.
