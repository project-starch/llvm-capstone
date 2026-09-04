# S-12 Explained: How a Stalled Store Hands the Wrong Operand to the Next Instruction

> **STATUS: CONSISTENT WITH FIXED, not proven. Variant A is synthesised and flashed; the SQLite
> domain that trapped now completes, 4 draws of 4, against an old arm that trapped 3 of 4.
> Fisher p = 0.071; against this folder's own ~54% per-draw rate, P(4 clean) = 0.045 — a bound
> this folder has ALREADY ruled insufficient for a "cure" claim (00-README.md:1278). Two more
> clean draws would take it to 0.0095 and settle it. See §9.4.** The flash was a
> deliberate lead override: `run.tcl`'s criterion remains unmet on negative slack.** The engineering detail, the variant comparison and the
> criterion live in `agent-handoff/plans/s12-fix-synthesis-request.md`. Measurements and
> provenance are in `00-README.md`.
>
> Last updated 2026-09-04 (synthesis results, §9).

S-12 wedges the SQLite domain with `mcause 25` (UNEXPECTED_OPERAND) and `tval = 0` at a
`cincoffsetimm`. It is silicon-only — QEMU never reproduces it, because QEMU models neither a
store buffer nor a scoreboard. This document explains why it happens.

```text
                    THE WHOLE STORY, AT A GLANCE

   §1     a capability store's scoreboard destination is its OWN
          source register.  stc a4 -> rd = a4.  (Deliberate.)
                              │
                              ▼
   §2     the store buffer fills, so that store STALLS at commit:
          we_gpr stays HIGH, commit_ack goes LOW
                              │
                              ▼
   §3     the WAW guard reads "a write is happening" and releases.
          The write IS happening.  The RETIREMENT is not.
                              │
                              ▼
   §4     so a younger `ldc a4` issues while the stalled store's
          entry is still LIVE and still a forwarding candidate
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        the LOAD lands                  the CONSUMER is fed
        correctly, register             the STALLED STORE's
        file is CORRECT                 value instead
                              │
                              ▼
   §5     that value is create_cnull() = {cursor 0, cap_type 0}
          → FLU raises UNEXPECTED_OPERAND, tval = 0
                              │
                              ▼
   §6     why it looks sporadic (~54%/draw) and why a fence cured it
                              │
                              ▼
   §7     why it lands on ONE address in a 1.6 MB program
                              │
                              ▼
   §8     the two fixes, and the one question that separates them
```

---

## 1. A Capability Store Writes Back to Its Own Source

An ordinary RISC-V store has no destination register. A **capability** store does, and it is the
register it stores *from*:

```text
        PLAIN STORE                    CAPABILITY STORE
        sd  a4, 0(a5)                  stc a4, 0(a5)

   ┌──────────────────────┐       ┌──────────────────────┐
   │ rs1 = a5  (address)  │       │ rs1 = a5  (address)  │
   │ rs2 = a4  (data)     │       │ rs2 = a4  (data)     │
   │ rd  = x0  ← NOTHING  │       │ rd  = a4  ← ITSELF   │
   └──────────────────────┘       └──────────────────────┘
                                             │
                                             ▼
                              `decoder.sv`:  rd := instr.rtype.rs2
```

**This is deliberate and must not be "fixed".** A capability store of a *linear* capability has to
destroy the source register — that is what linearity means, the authority moved to memory and must
not remain in the register. `capstone_dyn_unit.anvil` clears `rs2` to `cnull` for the
linear family, and the write-back path that does it is why `rd` is set at all.

The consequence, though, is that `stc a4, 0(a5)` is a genuine **producer of a4** in the
scoreboard — indistinguishable, to the hazard logic, from any other instruction that writes a4.

---

## 2. The Stall: `we_gpr` High, `commit_ack` Low

When the store buffer is full, the commit stage cannot hand it another store. It stalls — but it
stalls *asymmetrically*:

```text
   `commit_stage.sv`, one combinational block:

        we_gpr_o[0]    = 1'b1;      ← asserted EARLY, for the writeback
        ...
        if (commit_lsu_ready_i)  commit_lsu_o   = 1'b1;
        else                     commit_ack_o[0] = 1'b0;   ← ONLY the ack is cleared

   ┌──────────────────────────────────────────────────────────┐
   │  during a store-buffer-full stall, EVERY CYCLE:          │
   │                                                          │
   │      we_gpr_o[0]    = 1     "I am writing a4"            │
   │      waddr_o[0]     = a4                                 │
   │      commit_ack_o[0]= 0     "I am NOT retiring"          │
   └──────────────────────────────────────────────────────────┘
```

`we_gpr_o[0]` is never retracted. It stays high for the whole stall, which may be many cycles.

---

## 3. The Guard Reads the Wrong Half of That

The issue stage refuses to issue an instruction whose destination is already claimed by a live
producer — a WAW hazard. It has two ways to clear that refusal:

```text
   `issue_read_operands.sv`, the WAW block:

   CLAUSE 1   if (rd_clobber_gpr[rd] == NONE)          ← "nobody is claiming it"
                  stall_waw = 0;

   CLAUSE 2   if (we_gpr_i[c] && waddr_i[c] == rd)     ← "a commit port is writing
                  stall_waw = 0;                          it THIS CYCLE"
```

Clause 2's own comment says it checks that the register *"will be written in this cycle by the
commit stage"*.

**That premise is TRUE.** `we_pack[i] = we_gpr_i[i]` drives the register file with no dependence
on the acknowledgement. The write really does happen, every cycle of the stall.

**The INFERENCE is what fails.** Clause 2 exists to say *"the old producer is finishing, so a new
claimant may proceed."* With the ack withheld, the producer is **not** finishing:

```text
   WRITE  ──────────────► happens.        (we_gpr = 1)
   RETIRE ──────────────► does NOT.       (commit_ack = 0)

   and RETIREMENT is what ends a producer's role as a forwarding source:

        `scoreboard.sv`   "we've got an acknowledge from commit"
             if (commit_ack_i[i]) begin
                 mem_n[...].issued    = 1'b0;    ─┐
                 mem_n[...].cancelled = 1'b0;     ├── the bits candidacy is built from
                 mem_n[...].sbe.valid = 1'b0;    ─┘
             end

        `still_issued[i] = mem_q[i].issued & ~mem_q[i].cancelled`
```

So clause 2 releases the guard on the strength of a write, while the entry it belongs to is still
live, still valid, and still a candidate to forward from.

---

## 4. The Younger Load Issues Into the Gap

```text
   PROGRAM                     SCOREBOARD STATE                 GUARD

   movc a4, zero               ── a4 := {cursor 0, NOT_CAP}
        │
        ▼
   stc  a4, 0(a5)              ── entry S: rd = a4              store buffer FULL
        │                         issued = 1, valid = 1         we_gpr=1, ack=0
        │                         (written back, NOT retired)          │
        ▼                                                             ▼
   ldc  a4, 0(a0)              ── entry L: rd = a4        ←── CLAUSE 2 RELEASES IT
        │                         issued = 1, valid = 0        (a4 "is being written")
        │                         (issued, NOT produced)
        ▼
   cincoffsetimm a4, a4, 0xb0  ── needs a4 NOW
        │
        ▼
   forwarding candidacy = still_issued & sbe.valid
        │
        ├── entry S:  issued=1, valid=1   →  CANDIDATE        ✗ the stalled store
        └── entry L:  issued=1, valid=0   →  not a candidate  ✓ the real producer
                                                                 hasn't produced yet
        │
        ▼
   the consumer is handed entry S's value
```

Two live entries claim `a4` at once — the condition the WAW guard exists to prevent — and the only
one *eligible to forward* is the wrong one.

---

## 5. Why the Fault Is `mcause 25` With `tval = 0`

Entry S is a capability store whose source was `movc a4, zero`. Its write-back value is
`create_cnull()`:

```text
        create_cnull()  =  ┌─────────────┬──────────────┐
                           │  cursor  0  │ cap_type  0  │
                           └─────────────┴──────────────┘
                                  │              │
                                  │              └── 0 = NOT_CAP
                                  └── this is what `tval` reports

        cincoffsetimm needs a real capability in rs1.
        It receives {0, NOT_CAP}.
                                  │
                                  ▼
                  FLU raises UNEXPECTED_OPERAND  → mcause 25
                  with the ingested operand      → tval = 0
```

**This is the detail that misled the investigation for weeks.** `tval = 0` was read as *"the load
returned a null"*, which pointed at a memory-forwarding defect. But `tval` reports the operand
**as the execution unit ingested it**, not what the load returned — and those differ *precisely*
when the operand is forwarded.

The register file is **correct** at the fault. Measured twice, on both instruments:

```text
   in simulation, printed at all 254 traps:
        Reg[14]: Cursor 0000000080004000 | Revnode 2 | Type 2 | Perm 7
                 ^^^^^^^^^^^^^^^^^^^^^^^ a real capability — the load LANDED

   on silicon, the in-domain handler's report word:
        obs = 0xE643D221
              ^ marker 0xE = "a4's cursor is NON-ZERO"
        (0xF would have meant the load returned the null. It did not.)
```

The load lands. Only the consumer is misfed.

---

## 6. Why It Looks Sporadic, and Why a Fence Cured It

The mechanism needs the store buffer to be **full at that instant**. That is a property of the
dynamic memory behaviour, not of the code:

```text
   store buffer NOT full        store buffer FULL
   ────────────────────         ─────────────────
   stc retires immediately      stc stalls at commit
   ack fires                    we_gpr=1 / ack=0 persists
   guard releases correctly     guard releases WRONGLY
   ldc issues after S retires   ldc issues while S is live
        │                            │
        ▼                            ▼
     no fault                   S-12 fault
```

That explains three things that were confusing on the board:

| observation | why |
|---|---|
| ~54% per draw, not deterministic | the buffer must be full at that instant |
| `fence rw,rw` in the window cured it (0 wedges / 7, against a same-layout control wedging 4/4) | a fence **drains** the store buffer, so the store never stalls |
| QEMU never reproduced it | QEMU models neither a store buffer nor a scoreboard |

It also explains why the defect went unreproduced in simulation for so long. The directed test
that creates the shape existed for a day and reported **zero** — because the testbench default is
**zero memory latency**, where the store buffer cannot fill. The same ELF:

```text
        S12_MEM_DELAY = 0    →    0 traps      ← the test could not create its own trigger
        S12_MEM_DELAY = 40   →  254 traps
```

---

## 7. Why It Lands on One Address in a 1.6 MB Program

The instruction shape is common. The *exploitable* shape is not:

```text
   counted over the built SQLite domain, 331,808 instructions:

   ┌────────────────────────────────────────────────┬────────┬──────────┐
   │ stc rX  followed by  ldc rX   (the bare alias) │  2832  │ 12.8% of │
   │   → harmless: nothing type-checks the operand  │        │  stores  │
   ├────────────────────────────────────────────────┼────────┼──────────┤
   │ stc rX / ldc rX / capability consumer reads rX │    68  │  0.02%   │
   │   → the exploitable triple                     │        │          │
   ├────────────────────────────────────────────────┼────────┼──────────┤
   │ ...of which, on a4                             │     1  │ ← S-12   │
   └────────────────────────────────────────────────┴────────┴──────────┘
```

Without an immediate capability consumer, the stale operand is never ingested by anything that
checks it, so the escape is invisible. The defect is mechanically broad and observably narrow.

---

## 8. The Two Fixes, and the Question That Chooses Between Them

Both make the guard stop releasing on a write that is not a retirement. Both are validated in
simulation on the bitstream's own base; **neither has been synthesised.**

```text
   VARIANT A — s12-fix-for-synthesis        VARIANT B — s12-fix-variant-b
   require the acknowledgement               delete clause 2 entirely

   if (we_gpr_i[c] && commit_ack_i[c]        (clause 2 removed; clause 1 alone,
       && waddr_i[c] == rd)                   which already keys on still_issued
       stall_waw = 0;                         — i.e. candidacy itself)

   + a port and a pass-down                  one file, no port
   83/85 rows cycle-identical to base        31/85 rows slower, up to +8.95%
   imports a deep signal into the            adds no signal at all
     issue cone
```

Both: reproducer 255 → 1, delay sweep 0/4, in-regime suite 72/13/3 with identical non-passing
sets, UNOPTFLAT signal set unchanged.

**The question that decides it is not performance and not routability.** These bitstreams fail
timing badly — WNS −16.400 ns, 102,769 failing endpoints — and are usable anyway for one reason:

```text
   EVERY failing endpoint originates from  dom_switcher/cur_idx_q_reg
                              │
                              ▼
   that register toggles ONLY during a domain switch, frontend flushed
                              │
                              ▼
   so every failing path is INERT while a domain body executes
                              │
                              ▼
   THAT is why WNS -16.400 is tolerable and why board results mean anything

   Variant A adds a term to a stall driving issue_ack_o.  If that produces
   failing endpoints originating in the ISSUE cone, they are NOT inert —
   and the bitstream would still build, route and boot, with the reason its
   measurements can be believed silently gone.
```

So the discriminator is the **timing census**, not WNS. The reading is pre-registered in the
synthesis request. Variant B cannot import a new originating register, because it adds no signal.

---

## 9. What Synthesis Actually Said

Both arms built. Arm 1 is variant A on the resident base **with** the debug instrumentation, i.e.
it differs from what is on the board by the fix alone.

```text
                        WNS          failing endpoints    "found timing loop"
   base 80843404c      -16.400            102,769                100
   arm 1 (A + instr)   -15.311            101,782                100
   arm 2 (A, no instr) -13.491             99,879                100
```

**The fix IMPROVED timing** — 1.089 ns better than the resident image, with 987 fewer failing
endpoints. Consistent with gating the escape on `commit_ack` *removing* a release term rather
than adding one. Both arms routed legally, zero unroutable nets, no `LUTLP-1` DRC. The loop-count
concern did not materialise: 100 on both arms, identical to base.

**The census — the reading the whole decision hung on:**

```text
   ARM 1, verified as a census before anything was read from it
   (by-startpoint and by-endpoint both 101,784 vs arm 1's own 101,782):

        101,573   dom_switcher/_thread_0_event_reg_87_q_reg[0]
            209   dom_switcher/_init_0_reg
              2   DDR  (outside the CPU clock; base shows the same)
   ──────────────────────────────────────────────────────────────
              0   issue_read_operands   ← ABSENT, not hidden behind a larger cone
```

The zero is admissible because the control fired: `dom_switcher` returned 101,782 in the *same
query* that returned 0 for `issue_read_operands`. **By the pre-registered criterion, this is
"ship A".**

### 9.1 The finding that outlives this build

Arm 2 — the same fix with the debug tree tied off — has the mirror-image census: **99,879 of
99,879 launching from `issue_read_operands`, with `dom_switcher` at zero.**

The mechanism is that the debug mux consumes dom-switch state extensively (all five
`dom_switch_*_log_q` registers), so a large share of the base's failing paths were paths *into
the mux*. Remove the mux and they cease to exist, exposing whatever was second — which is issue
logic.

**Stated precisely, because a looser version of this was published first and had to be pulled
back.** Which comparisons are single-variable decides what may be claimed:

```text
   arm1 vs arm2   both carry the FIX, differ ONLY by the tie-off      ONE variable   ✓
   base vs arm1   differ by the fix only                              ONE variable   ✓
   base vs arm2   differ by fix AND instrumentation                   TWO variables  ✗
```

**ESTABLISHED** (from the arm1/arm2 pair, which is genuinely single-variable): *on the S-12-fixed
design*, removing the instrumentation moves the cone entirely — `dom_switcher` 101,782 → 0,
`issue_read_operands` 0 → 99,879.

**NOT ESTABLISHED:** that the same happens on the **unfixed** design. That is exactly the gap
reading (b) in §9.2 occupies — under (b) the issue-cone paths are the *fix's*, the mux merely
masked them in arm 1, and the historical builds' inertness was **genuine** rather than an
artifact. Nothing measured excludes (b).

The reason no history settles it: **`6f8345fdb` is the only instrument-free build that exists on
this project.** All thirteen other archived builds carry the debug tree. So there is no evidence
whatsoever about the instrument-free behaviour of any *pre-fix* commit.

What survives, and it is still the consequential part:

> The inertness argument has only ever been validated on **instrumented** configurations. It
> should not be inherited by an instrument-free build without being remade.

What does **not** survive is the causal version — "the inertness of every timing-failing bitstream
in this investigation was an artifact of the instrumentation". That generalises from one build
which also carries a fix, and it resolves §9.2's open question by assertion. It was published in
an earlier revision of this file and is retracted here.

None of this endangers arm 1, which carries the instrumentation and whose failing cone is
demonstrably inert.

### 9.2 The control that is missing

Arm 2 was compared against a base that HAS instrumentation. So arm 2's issue-cone census cannot
yet be attributed:

```text
   arm 2  = fix + no instrumentation  →  99,879 endpoints from issue_read_operands
   base   = no fix + instrumentation  →  102,769 endpoints from dom_switcher

   MISSING:  no fix + NO instrumentation
             └── if it also shows ~100k from issue_read_operands, the fix is NOT the cause
                 and this is a pre-existing property of the design
             └── if it does not, the fix is the cause and variant B becomes the candidate
                 for any instrument-free build
```

Until that arm exists, "the fix is safe in an instrument-free configuration" is unproven in both
directions. It does not block flashing arm 1, which is instrumented.

### 9.2b The licence names one register; the census names another

The inertness argument has always been stated about **`cur_idx_q_reg`** — "it toggles only during
a domain switch with the frontend flushed, so every failing path is inert while a domain body
runs." Arm 1's census does not name that register:

```text
   101,573  dom_switcher/_thread_0_event_reg_87_q_reg[0]     ← 99.8% of the failing endpoints
       209  dom_switcher/_init_0_reg
```

`_thread_0_event_reg_87_q` is not `cur_idx`. It does not appear in the Anvil source at all — it is
a **compiler-generated event-join register** in the switcher's thread machinery:

```text
   EVENTS0[87].event_current = (EVENTS0[86] & q[0]) | (EVENTS0[80] & q[1])
                             | (EVENTS0[86] & EVENTS0[80])
   _thread_0_event_reg_87_n  = q ^ {EVENTS0[86], EVENTS0[80]} ^ {EVENTS0[87], EVENTS0[87]}
```

a rendezvous flag recording which of two predecessor events has fired. It changes only when those
events fire, so its activity is **event-driven**, and the events come from counters
(`_thread_0_event_counter_{86,80}_1_q`) on a thread that loops
(`EVENTS0[0] = _init_0 || EVENTS0[102]`).

**RESOLVED 2026-09-04, by measurement, in the direction that keeps the licence.** A trace of a
domain-body run, checking the only three events that can change the register:

```text
   run length 1057 timestamps
      EVENTS0[80]   transitions = 1     (settles to 0 at t=0, nothing after)
      EVENTS0[86]   transitions = 1     (same)
      EVENTS0[87]   transitions = 1     (same)
```

A register whose every driving term is constant cannot toggle. The event thread does **not**
advance during body execution — it idles. So arm 1's 101,573 endpoints are inert *on their own
evidence*, not by inheritance from the `cur_idx` argument. The remaining 209 from `_init_0_reg`
are inert by construction: set at reset, cleared once when `EVENTS0[4]` first fires, static
thereafter.

**The limit, which matters for our workload specifically.** That trace was taken on `capldc`,
where the switcher stays idle throughout. It establishes inertness for *body execution with no
switch in progress*. It does **not** characterise the thread across a switch, nor show it
returning to the static state after one.

That is not academic here: **the S-12 workload is a SQLite domain entered via `capenter`**, so its
body executes *after* a switch. If a completed switch could leave the thread in a state where the
events subsequently toggle, this trace would not see it. Closing that needs a trace of a workload
that performs a `capenter` or `domcall` and then keeps executing — worth having before anyone
relies on this for a flash.

**Why this was worth asking at all.** The criterion said *"dom_switcher-originating"*; what it
MEANT was *"originating from something inert during body execution"*. Those coincided only because
every previous build happened to launch from `cur_idx`. Arm 1 separated them. The general lesson
is the same one this investigation kept relearning, one level up: **the check fired, and whether
firing separates the hypotheses is a different question.** The criterion should name the
*property*, with `dom_switcher` membership as evidence toward it rather than as the test itself.

### 9.3 Why the instrument-free question is not academic

The debug instrumentation appears to be **stale**, and it is expensive. Across the entire S-12
board campaign every mux reading was weak, void, or faulted:

```text
   sw=208  0xb8  "unmatched (WEAK)"   ldc0_valid(=UNKNOWN SEMANTICS for this bitstream)
   sw=208  0xfe  "INSTRUMENT FAULT: ldc0_src == 3 is not a defined source. Readout is wrong"
   sw=204        "displacement VOID ... cross-aperture contamination. It is NOT data."
   rev-node head "VOID ... NO consumption datum for this rep"
```

Not one usable datum, and the decoder's own text says "UNKNOWN SEMANTICS for this bitstream" — it
no longer matches what the silicon exposes. `HALT_MUX_READS` defaults to `0`, so the reads are off
by default in any case.

**Every S-12 verdict came from software instead:** the in-domain trap handler packs `mcause`,
`mepc` and `a4`'s zero/non-zero into the domain's return word (`SQ: obs=`). That is what produced
`obs=0xE643D221`. It needs no bitstream support at all.

**The tree is cheap in area and expensive in timing**, which is only a paradox until you see
where the endpoints launch from:

```text
   post-placement        arm 1      arm 2     delta
   Slice LUTs          169,694    168,944      +750     0.37% of the part
   Slice Registers      93,257     92,618      +639     0.16%
   BRAM / DSP               54/27     54/27       0

   101,573 endpoints launch from ONE BIT with enormous fanout.
   The tree does not CONTAIN those paths — it gives them somewhere
   distant and timing-poor to END. Remove it and they do not move,
   they cease to exist, and the next-worst cone is exposed.
```

Meanwhile the instrumentation costs, on the synthesis numbers:

```text
   base (instrumented)     -16.400
   arm 1 (fix + instr)     -15.311     the fix buys        1.089 ns
   arm 2 (fix, no instr)   -13.491     dropping it buys a further 1.820 ns
```

**Dropping the debug tree is worth more than the fix**, and removes ~100k failing endpoints from
the switcher. So the missing control in §9.2 decides something larger than S-12: whether this
project can ship a permanently simpler, ~2.9 ns better bitstream.

* base+tie-off shows ~100k from `issue_read_operands` → pre-existing; **arm 2** is the better
  vehicle and the debug tree can go.
* base+tie-off is clean there → the fix created them; **variant B** becomes the instrument-free
  candidate and arm 1 remains the safe instrumented option.

---

### 9.4 Flashed and verified on silicon — 2026-09-04

`caplifive_s12fix_5097eb166.bit` (arm 1, sha256 `7a97ccd0…62999b0`) was installed in the console
store, hash-verified on both sides of the transfer, and flashed. The console independently reports
it resident via its own `flash_state.nv_bitstream_name`.

The same domain image that produced the trap — `sha 29289cdeeac9`, byte-identical, same test,
same control — was then re-run. Only the bitstream changed (the two run scripts differ by exactly
one line, `FPGA_BITSTREAM`).

**Two limits on that, both found by audit rather than by us.** The pre-registered criterion said
"per-draw image sha verified IN-BOOT"; it was not. The `verifying … sha256=` line is a host-side
preflight print that never reaches the board. What does hold in its place is the driver's
STALE-FIRMWARE check, which decompresses the initramfs inside `fw_payload.bin` and searches for
the actual `.dom` bytes — a real check with a real failure path, but host-side inference rather
than a board report. And the control (`sqslt.dom`) passes on BOTH bitstreams, so it is a
**boot-health control, not a fault-positive one**: nothing in the post-fix set was required to
fail, so a run that silently never reached the trigger would look identical to a fix.

> **CORRECTED 2026-09-04 after audit. The table first published here was WRONG, and so was the
> lesson drawn from it.** Both are replaced below; the original is described rather than kept,
> because it misfiled every row of the old arm.

Read from each run's OWN region (everything before `POST images/upload` is the console replaying
the previous boot):

```text
   draw   OLD bitstream s10fix_80843404c     NEW bitstream s12fix_5097eb166
    1     E643D221  X/fail  rc=1             5117600D  completed=1  rc=0
    2     E643D221  X/fail  rc=1             5117600D  completed=1  rc=0
    3     E643D221  X/fail  rc=1             5117600D  completed=1  rc=0
    4     5117600D  completed=1  rc=0        5117600D  completed=1  rc=0
                    ^^^^^^^^ the old arm PASSED here
   old arm: 3 of 4 trapped.   new arm: 0 of 4.
```

**What the first version got wrong, and why it matters more than the numbers.** It published the
old arm as 1-of-4 trapping with draw 4 as the trap. In fact draws 1-3 trapped and draw 4 passed —
every row was misfiled, by a single line in the run script:

```bash
grep -aoE "obs=[0-9]+" /tmp/capstone/$ARM-$i.log | tail -1     # WRONG: whole log, silent fallback
```

* Draws 1-3 reported `obs=40465`, which is **not UART output at all** — it is the driver's own help
  text explaining what that sentinel would mean.
* Draw 4's current run emitted no `obs=` at all, so `tail -1` fell back to the **replayed previous
  boot** and reported that draw's trap word.

**So the "reading discipline" lesson published here was INVERTED.** It said the whole-log grep was
right and the scoped extractor wrong. The opposite is true: the scoped capture was correct — its
`SQ: self=` values bind it to the current boot — and the whole-log grep read the replay. The rule
that survives is the ordinary one, stated correctly: **scope every read to the region after
`POST images/upload`, and treat an absent marker as an ERROR, never as a fallback to `tail -1`.**

`0x5117600D` is **`SQLITE_HC_SLT_RAN`** (`capstone/benchmarks/sqlite/sqlite_hostcall.h:67`) — a
FIXED CONSTANT meaning "the runner reached its normal exit", not a computed result. That is the
right reason to trust it, and it explains what would otherwise look suspicious: the same word
appears for the control, for the arm, and for the old bitstream's one passing draw. Calling it an
"ordinary result value" was weaker and slightly wrong. The
`SLT-SUMMARY` is identical to what QEMU emulation produces for this test, so silicon now agrees
with the reference.

**Reading discipline for anyone re-running this.** Two traps caught on the way:

* `obs=` came back EMPTY on the fixed build. That is missing data, not success — the verdict came
  from `ENT2` plus `rc` plus the presence of `SLT-SUMMARY`, all read in the run's own region.
* The SCOPED per-draw capture for the old trapping run showed `completed=1, rc=0`, contradicting
  the trap. It was displaying an earlier run. Here the whole-log grep was correct and the scoped
  extractor was not — the reverse of the usual failure, and it nearly caused a correct result to
  be retracted. Check both, and reconcile them against `rc` before believing either.

**The strongest single observation, which the original write-up did not use.** `s12stress.test`
completed **128 records / 128 prepares** on the fixed bitstream, against `dd2_join`'s 2 — far more
exposure to the trigger, passing, with `query_pass=120 query_fail=0` identical to native x86. It
cannot be scored, because no old-bitstream `s12stress` arm exists, but it is better evidence than
the arm that was scored.

---

## 10. Shortest Possible Explanation

A capability store must write back to its own source register, so it is a real producer of that
register. When the store buffer fills, that store keeps asserting "I am writing this register"
while no longer asserting "I am retiring". The hazard guard reads the first half and lets a
younger load to the same register issue — but the stalled store's entry is still the only one
*eligible to forward from*, because the load has not produced yet. The consumer therefore receives
the store's value, a null capability, and faults. The load itself was fine; only the consumer was
misfed.

---

## 11. What This Does NOT Cover

* **Why the board rate is ~54% while simulation is deterministic.** Under a fixed memory latency
  the mechanism fires 254/254 on a 737-cycle period. The board's variability is unexplained and is
  not claimed either way.
* **The other 67 exploitable triples.** Whether they are reachable, and whether any has ever
  fired, is unknown.
* ~~**Synthesizability.**~~ **SETTLED 2026-09-04:** arm 1 builds, routes legally, improves WNS by
  1.089 ns and adds no failing endpoints from the modified module. See §9.
* **Whether the fix is safe WITHOUT the debug instrumentation.** Unproven in both directions until
  a no-fix / no-instrumentation arm exists — see §9.2.
* **A general WAW hole that predates this defect.** Under variant A the repaired clause still
  releases on the acknowledgement of the *oldest* clobberer while a younger in-flight entry claims
  the same `rd`. That hole exists in the base too; A narrows it to the S-12 shape rather than
  closing it. (Clause 1 is not involved — it never looks at the acknowledgement.)
