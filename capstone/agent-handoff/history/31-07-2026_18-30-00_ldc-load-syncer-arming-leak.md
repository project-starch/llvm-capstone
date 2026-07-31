# 31-07-2026 — RTL defect: LDC's NOT_CAP arm leaves the load syncer armed

## The defect, verified by direct quote

`capstone_dyn_unit.anvil` arms the load syncer **before any operand check**:

```
302:    send cap_load_ri.init(msg.trans_id) >>
303:    if(rs1.metadata.cap_type==cap_type_t::NOT_CAP){
305:        dprint "Unexpected operand in LDC %d" (msg.rs1);
306:        call raise_exception(msg.trans_id,ex_code::UNEXPECTED_OPERAND)
307:    }
```

Every **other** LDC error arm tears the arming back down with
`call abort_accumulation_load(...)` — bad cap type, insufficient permission, out of bounds,
misaligned, revoked. The `NOT_CAP` arm does not.

`STC`, which is otherwise a mirror image, **does** disarm on the same condition:

```
367:    if(rs1.metadata.cap_type == cap_type_t::NOT_CAP){
369:        call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)>>
370:        call abort_accumulation_store(data.trans_id,ex_code::UNEXPECTED_OPERAND)
371:    }
```

Two mirror-image functions differing on exactly one line is a missing line, not a design
choice.

## Why a stale arming is fatal

`capstone_dyn_unit.anvil:521-522` — the syncer's state is
`reg cap_trans_id : logic[3]` and `reg req_set : logic[1]`. **Three bits: eight values.**

The divert, `:554-565`:

```
554:  if(*req_set == 1'd1 && msg.trans_id == *cap_trans_id){
555:      try cap_msg = recv dyn_ep.req {
556:          call check_load_data(msg,cap_msg) >> set req_set := 1'd0
558:      } else {
560:          set lsu_reg := msg;
561:          set lsu_msg_set := 1'd1
562:      }
563:  } else {
565:      send lsu_ep.normal_res(msg) >>
```

`lsu_ep.normal_res` at `:565` is the **only** path to writeback
(`ex_stage.sv:933  assign load_valid_o = forward_normal_load_valid;`). A load result whose
`trans_id` happens to match a *stale* `cap_trans_id` takes `:560-561` instead: it is filed
into `lsu_reg` to await a `dyn_ep.req` that will never come, and is never forwarded. The
instruction never retires. The store syncer is line-for-line identical.

## Why this matches the measured wedge and the killed candidates do not

Measured on the wedged core (LED mux, `switches` values verified against `cva6.sv:1090-1215`,
`debug_byte_sel = 0b111` confirmed in the capture):

```
sw=224   privM=1  flu_ready=1  dyn_ready=1  lsu_ready=1  ex_commit.valid=0
sw=225   stall_issue=1, all other status bits 0
sw=249   rev head=217   sw=250 overflow=0
COMMIT pc = 0x81f3c71c  ->  image VA 0x14c71c
```

* **stall_issue=1 with every unit READY** — the consumer of the lost writeback is RAW-blocked
  in `issue_read_operands.sv`, while no unit is busy, because the dyn unit is idle at its loop
  head: the LDC *finished* (via an error arm).
* **All three dyn wait flags 0** — `waiting_for_load_syncer` is set only inside
  `abort_accumulation_load`, which is precisely the call that was not made.
* **No capability fault in the trap latch** — see the trigger discussion below.
* **QEMU has no syncer at all**, so the divergence is free and needs no other explanation.

## The trigger: NOT established, and one candidate is ruled out

The `NOT_CAP` arm *does* raise cause 24, which is "nontrivial" and would have overwritten the
latched cause-9 in `recent_nontrivial_mcause_log_q`. It did not. So **a program-level LDC on
an untagged base is not our trigger** — and independently, QEMU is not permissive about that
either (`capstone-qemu/target/riscv/op_helper.c:1051-1055` raises on `!rs1_v->tag`), so it
would have failed under emulation too, and it does not.

The candidate that fits the measurement is a **pipeline flush landing between `:302`'s
`init` and the `req`/`res` pair at `:343-345`**, abandoning the transaction with *no* abort
and *no* exception. That needs no illegal instruction, and the window is wide because
`get_node_query_validity` at `:331` is a full round trip to the rev-node unit.

Suggestive but not proof: the last committed instruction is `bnez` at image VA `0x14c71c`,
the loop back-edge of `strlen`. A mispredicted branch is exactly what produces a flush.

Not settleable from the tree: whether Anvil resets procedure registers on `loop
dyn_ep.flush` (`:520`). `find capstone/caplifive-system -iname "*dyn_unit*"` returns only the
`.anvil` — no generated Verilog — so `req_set`'s behaviour across a flush cannot be read here.

## What is needed to settle this (RTL-side)

1. The one-line asymmetry above (`:306` versus `:369-370`) — it is a defect on its own merits
   whether or not it is our wedge. Registered as R-14 in ref/ISSUES.md.
2. The question that cannot be answered from the sources in this repo: **does a pipeline
   flush reset `req_set` / `cap_trans_id` in the load and store syncers?** If not, any
   abandoned capability access arms an 8-value comparator that will silently swallow a later
   unrelated load.
3. The measurement above, which shows a core stalled at issue with every functional unit
   ready and nothing committing.

## Status

The defect is **verified**; its role in the SQLite wedge is **not**. Do not report it as the
root cause until the trigger is established. The independent bisection —
`sqlite_stage20/21/22`, which unbundle the four variables of the 20-line reproducer — is the
line of evidence that should confirm or refute it, and it does not depend on this hypothesis
being right.

---

## Cross-references

* Registry entry: **R-14** in `ref/ISSUES.md` — the observable failure and its four variants.
* Reproducer bundle: `/tmp/capstone/R14-strline-struct-repro.tar.gz` — four ready-to-run
  `.dom` files (A wedges, B returns a wrong value, C and D are correct), the source, the run
  recipe, and the wedged-core register dump.
* **Board-validated workaround:** variant C. Building the array in a loop from a static table
  instead of straight-line passes on silicon. That is a software fix and needs no RTL change,
  so it should be tried before this hypothesis is pursued further.
