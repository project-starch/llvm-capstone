# Paper claims checked against Line B's final core (b42c8c644), and the stage-2 decoder side (2026-11-26)

*RTL lane. Asked by the runtime lane after the lead's re-read of the paper (ebc1700a). Board-free: every reading here is
the RTL source at b42c8c644 or a simulation of it. The paper itself is not touched.*

Tests live on capstone-ariane `b5b2-hangmon-c`, whose core is b42c8c644's plus SUP_TRACE-only monitors and comments.
- Pre-registered at d2efe50d8; first cuts read there.
- Second cuts pre-registered at a4305cb8d; their results recorded at 9ed07e2db.
- Runs in s2pre-wt `.sup/paper1` and `.sup/paper2`: dirty=no, memory delay 12, each model's verFiles carrying
  +R43_TRACE.

## The three claims

| Paper | Claim | On b42c8c644 | Evidence |
|---|---|---|---|
| 5.2 | "a separate record TLB of N entries supplies the record page" | **present, N = 16** | source (below) |
| 5.3 Release | "a slot that has used up its generations retires" | **holds**, at the core and against the reference | b5b2-genlim, sublet-life-diff |
| 5.3 Access | "discards pending fills that predate a record update and replays younger accesses that checked the old generation" | **holds on the instance reached**, warm and cold | b5b2-relrace-warm/-cold/-live |

### 5.2: the record TLB

- `ex_stage.sv:2218` `localparam int unsigned PwTlbN = 16`, inside `if (CVA6Cfg.CAPSTONE_EXT)` (:986), with its Sublet
  parts under `ifdef CAPSTONE_SUBLET_FMT`.
- It is fully associative. The tag is slot[30:8], one record page of 256 records. It holds the page address [55:12],
  and replacement is round-robin (`pw_tlb_rr_q`, 4 bits).
- It is looked up from flops only (:2289-2302). It is filled after a directory walk unless the namespace flushes in
  that cycle (:2394-2399), and flushed whole on every namespace flush (:2402-2405).
- Flat tables bypass it (`pw_flat`, :2385).

### 5.3 Release: retirement at generation 2^32 - 1

The rule is in `capstone_sublet_life.sv`: `GEN_LIM = 2^32 - 1` (:115-116). A release whose next generation reaches it
(:626-633) does three things:
- it writes the retired word {GEN_RET, 0};
- on a paged table it adds one to the header's retired count;
- it pushes nothing.

**Before today** the only test of the limit was sublet-life-diff's `hi-gen` case (317af40e8). That case reaches the
limit by CRENEW, the out-of-place path. Nothing released a record at 2^32 - 2 by CREVOKE.

**b5b2-genlim**, core level, has a control one generation lower. The free head holds generation G; CDERIVE pops it and
CREVOKE releases it. S reads the record and the header straight after the release, and a second CDERIVE follows.
Both arms read exactly as pre-registered:

| arm | G | id1 | record w0 / w1 after the release | q1 | retired | id2 |
|---|---|---|---|---|---|---|
| default | 2^32 - 2 | 0x7fffffff00000064 | 0x0 / 0xffffffff00000000 | 0x0 | 0x1 | 0x100 (next) |
| GL_CTL | 2^32 - 3 | 0x7ffffffe80000064 | 0x0 / 0xfffffffe00000000 | 0x100000064 | 0x0 | 0x7fffffff00000064 (reissued) |

The first cut's GL_CTL was VOID at three positions by my prediction error: M read the table after the second CDERIVE
had re-popped record 100. Its id2 already showed the push. The second cut reads before that CDERIVE.

**Unit level, sublet-life-diff.py.** New cases: `kill-hi-gen` and `paged-kill-hi-gen`, each with a `-ctl` one generation
lower.
- Built here: the reference harness from capstone-qemu da49602d76 by build-harness.sh, and tb_sublet_life with the
  host's Verilator 5.020, not the project's 5.008.
- 84 of 86 cases equal the reference. The 5 recorded divergences behave as recorded, and every answer kind is produced.
- In the two new limit cases the release itself equals the reference: record dump, retired count, no push, and the
  next CDERIVE.
- They differ in ONE answer: a liveness query for {2^32 - 1, slot} after the slot retired. The reference answers dead
  (`cap_rev_table_sublet_live_word`, cap_rev_table.c:460-463: `!sublet_retired(gen) && gen == id_gen`). The RTL's
  query answers live: `S_Q_W`, capstone_sublet_life.sv:398, `rlive_q <= !rsp_bad_q && (r_gen(rsp_w1_q) == id_gen)`,
  with no retired-generation term.
- The LSU's probe path excludes the retired word (ex_stage.sv:2628-2629), so this shape cannot allow an access.
- The divergence is wider, and is filed as **ISSUES R-84** (claim-audited).
  - The query reads the record alone, so it also answers live for a slot at or above `next`, and for any slot of a
    table whose header fails validation. Both are measured on scratch streams; the reference answers dead.
  - Its only sender under Sublet is LCC selector 0, which grants no authority.
  - No instruction issues an id with generation 2^32 - 1 or at or above next. Such an id needs a forged tag (S-30).
  - Whether the ACCESS path shares the last two shapes is not tested, and R-84 names the arm that would settle it.

### 5.3 Access: a release in flight against a younger access (the record path's analogue of R-74)

**The mechanism in RTL** (load_store_unit.sv):
- A pending probe resolves on any read of its node (:1833-1836), so a read made before the release can fill ALLOW.
- The release's write invalidates the ALLOW record and the cache entry (`rvc_ok_kill`, :1415-1429). An invalidation in
  the resolving cycle beats the set (`rvc_set_kill`, :1838-1841). `rvc_inv_now` denies an access in its cycle
  (:1471-1475).
- The CREVOKE's commit flushes and refetches everything younger (commit_stage.sv:951-954).

**Before today** no Line B test put a release in flight against an access that checked the old generation.
b5b2-crevint's CI_YLDC issues an LDC during a CREVOKE walk, but through a live child.

**b5b2-relrace-warm / -cold / -live** are b5b2-crevint with an independent plain load right after CREVOKE o, x31 = 0x5e
before it. The walk ends o and its 200 children.

**First cut VOID** (through x25, the CREVOKE's own rs1). The vectors read as predicted, but the window was not reached:
the load's only verdict follows o's release write.
- Every capability op but LDC, the raw ops and a non-UNINIT STC claims its rs1 while issued (`gpr_clobber_vld`,
  issue_read_operands.sv:648-656), so a younger reader of the CREVOKE's operand waits for its writeback.
- So **a younger access through the revoked register itself cannot race the release, by construction.**
- The race goes through another register holding a capability the same CREVOKE ends.

**Second cut:** the load goes through o's last child (slot 0xcf), released at cycle ~115,300 of a walk from ~101,200.

| arm | reading | window |
|---|---|---|
| warm | 0x19 ... 0x5e ... 0x2 | first verdict `hit=1 live=1 exc=0`, before the child's release write: it **checked the old generation**; re-executed after the commit flush `live=0 exc=1` (25); x31 still 0x5e, never committed |
| cold | 0x19 ... 0x5e ... 0x2 | a miss (`replay=1`) whose probe resolved `-> ALLOW` **before the release write**: a fill that predates the update; invalidated by the release; the re-execution denied (25) |
| live (control) | 0x0 ... 0x0 ... 0x1 | the same load through the root completes: the sentinel is replaced, so the detector sees a completed load |

The full vectors are in the test header. N = 1 race instance per arm, at one timing: not a proof for every timing.

## Stage 2: the decoder side (for the compiler lane's spec)

- **Zcb is decoded.** The config package sets `CVA6ConfigZcbExtEn = 1` (:77) and Zcmp 0. compressed_decoder.sv:152-153
  decodes `OpcodeC0Zcb` under `CVA6Cfg.RVZCB`. Q0 100 [12:10] = 100..111 stay free.
- **The mode bit's home (agreed with the compiler lane, its spec compiler/stage2-trap-spec cec9f6b8cea9).**
  - A separate flop `crvc_q` is loaded from bit 0 of the srevroot write (csr_regfile.sv:2076, which already flushes)
    and read back in bit 0 (:1104).
  - It is effective only in protected U: `crvc_o = crvc_q && cap_data_ctx[0]` (:3013-3014). The protected-U predicate
    and the table address are unchanged.
  - The other srevroot write, at csr_regfile.sv:2146, has no driver on Line B:
    - CSRUNV is illegal at decode under Sublet (decoder.sv:1492-1500).
    - Every switcher arm for id 67 is gated on `is_runv` (capstone_dom_switcher.anvil:556-574, :324, :510).
    - The non-CSRUNV full walk ends at id 66 (:575-576).
    - The spec still clears crvc_q there, as a defence.
- **Size and gate.**
  - The change is confined to compressed_decoder.sv (one instance, id_stage.sv:167). Each form is a fixed bit
    permutation into an existing 32-bit Capstone encoding.
  - The decoder's output feeds `id_stage_i.genblk2[0].decoder_i.instruction_o`, on the standing UNOPTFLAT list. So the
    **synthesis-only gate comes first**.
  - pc + 4 sites re-read: frontend.sv:449; commit_stage.sv:953's flush_commit list has none of the stage-2 shapes.
