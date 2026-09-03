# S-12 Explained: How a Stalled Store Hands the Wrong Operand to the Next Instruction

> **STATUS: root cause confirmed in RTL simulation AND on silicon. Two candidate fixes exist,
> both validated in simulation; NEITHER has been synthesised.** The engineering detail, the
> variant comparison and the pre-registered criterion that chooses between them live in
> `agent-handoff/plans/s12-fix-synthesis-request.md` — this document deliberately does not carry
> them. Measurements and provenance are in `00-README.md`.
>
> Last updated 2026-09-04.

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

## 9. Shortest Possible Explanation

A capability store must write back to its own source register, so it is a real producer of that
register. When the store buffer fills, that store keeps asserting "I am writing this register"
while no longer asserting "I am retiring". The hazard guard reads the first half and lets a
younger load to the same register issue — but the stalled store's entry is still the only one
*eligible to forward from*, because the load has not produced yet. The consumer therefore receives
the store's value, a null capability, and faults. The load itself was fine; only the consumer was
misfed.

---

## 10. What This Does NOT Cover

* **Why the board rate is ~54% while simulation is deterministic.** Under a fixed memory latency
  the mechanism fires 254/254 on a 737-cycle period. The board's variability is unexplained and is
  not claimed either way.
* **The other 67 exploitable triples.** Whether they are reachable, and whether any has ever
  fired, is unknown.
* **Synthesizability.** Simulation says nothing about it, and the fix touches a module inside a
  standing combinational-loop cone. Only `synth_design` settles it.
* **A general WAW hole that predates this defect.** Under variant A the repaired clause still
  releases on the acknowledgement of the *oldest* clobberer while a younger in-flight entry claims
  the same `rd`. That hole exists in the base too; A narrows it to the S-12 shape rather than
  closing it. (Clause 1 is not involved — it never looks at the acknowledgement.)
