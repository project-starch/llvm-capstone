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
- This is ISSUES R-84 (A), claim-audited, and row D11 of the B5b-1 format plan's divergence table. The other two shapes first filed there, a slot at or above `next` and a header that fails validation, are that table's D1 and D4, recorded and ruled on 2026-11-22. I re-derived them (corrected in R-84).
- Superseded detail, kept for the record:
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

## Six more claims of sections 1, 5 and 6, checked against Line B (2026-11-28)

*RTL lane, item 2.8 of docs/plans/2026-11-28-line-b-completion.md. Every reading below is a source line at b42c8c644
(capstone-ariane), or in the kernel that the campaign rows booted. That kernel is `<hdd>/capstone/trapk/fpga/linux-6.1`,
the source of the 165.03 G row's payload 539b2ef41ee4d444..., whose .config equals trapk/fpga-v1's except the initramfs
path. Kernel files are cited by path and the first 16 hex digits of their sha256. The paper is the submitted ebc1700a.
It is not touched here; the wording is the lead's.*

| Paper (ebc1700a) | Claim | On Line B | Evidence |
|---|---|---|---|
| 5, Lifetime record (:47-55) | 16-byte record: generation 32 bits, the other three fields 31 each, 3 bits reserved | **holds** | capstone_sublet_life.sv :9-10, :130-147 |
| 5, Allocate (:107-113) | a batch is 16 record pages (4,096 records) by default, and T starts with one batch | **holds** (P1 started with four pages) | kernel sublet-table.c :54, :578 |
| 5, Release (:124-128) | CREVOKE ends the object and its descendants "in one pass over their records" | **does not hold as worded:** a forward scan, then a backward check and release in chunks of 64 records, interruptible | capstone_sublet_life.sv :34-51, :126 |
| 5, Release (:130-133) | it fails on a stale capability, "checked before it changes any record" | **holds** for the stale case; across chunks, see below | capstone_sublet_life.sv :478-481 against the kill path's writes |
| 6, Address Space Layouts (:44-55) | a CDERIVE below R per region (code, data, stack, each mmap); munmap checks the reference live, CREVOKEs it, and only then removes the pages | **holds**, with munmap's revoke only when the unmap removes the block's last pages | kernel sublet-table.c :504-545, :730-737, :813-826; sublet-blocks.c :35-75, :96-123, :244; mm/mmap.c :2406, :2782-2783 |
| 1 (:96-97) | "Programs that use capabilities run in a dedicated mode with user-space privileges" | **a configuration of U, not a separate mode:** protected U is U with srevroot != 0, chosen per process by the kernel | csr_regfile.sv :3012-3015 (cap_data_ctx_o[0]); capmode tied to 0 (:348-349) |

**The record layout.**
- capstone_sublet_life.sv :9-10 states the packing: word 0 = {next[29:0] at 63:34, depth[30:0] at 33:3, zero[2:0]};
  word 1 = {generation[31:0] at 63:32, previous[30:0] at 31:1, next[30] at 0}.
- The accessors at :130-147 (r_next, r_depth, r_prev, r_gen; mk_w0, mk_w1) read and write exactly that.
- So: a 32-bit generation, three 31-bit fields (previous, next, depth) and 3 zero bits, 128 in all, as the paper
  says. The reserved bits sit at word 0's low end, and next straddles the two words.

**The initial table size.**
- The kernel creates each context's table with node_initial_pages record pages (sublet-table.c :578,
  `sublet_table_grow(c->table, 1, node_initial_pages, node_initial_pages, &pages)`).
- node_initial_pages defaults to CV_TABLE_BATCH_PAGES (:54) = CAP_REV_SUBLET_BATCH_PAGES = 16 (cap_rev_table_abi.h
  :116, f952b49103024d65). A page holds 256 records (sublet-format.h :80, e37613d6d4c06a58), so one batch is 4,096
  records.
- Growth adds node_batch_pages, also 16 (sublet-table.c :54, :110).
- Both are module parameters. Neither the board's bootargs (caplifive-b5b2.dts :10: "earlycon console=ttyS0,57600
  rdinit=/init sublet.dom_binfmt=1") nor CONFIG_CMDLINE sets them, so the booted kernel uses the defaults.
- sublet-table.c is e94037ad98b917db.

**CREVOKE is not one pass on Line B** (capstone_sublet_life.sv :34-51, the interruptible CREVOKE of B5b-2 4(b)).
- A forward scan finds the run's last record t.
- The run is then released from its end backward, in chunks of at most KCHUNK = 64 records (:126). Each chunk is
  checked walking back from t (live, deeper than the operand, a prev that names it as next) before any of it is
  released, then walked again to release it (states S_KB_* check, S_KR_* release, S_KS_* splice).
- After a chunk, a pending interrupt ends the instruction with status PARTIAL, and the CREVOKE re-executes from its
  checks on the shorter run. A saved position may let the re-execution skip the forward scan (:44-51).
- So every record of the run is read at least twice (the scan and the chunk's check), and its release is a third
  visit. A long CREVOKE may span several executions.
- The paper's "in one pass over their records" describes the reference's model (cap_rev_tree.c walk_run's check and
  commit passes are two as well), not this hardware.
- CRENEW keeps the forward commit pass in list order in one execution (:43).

**The stale check precedes every write.**
- The kill path resolves the operand first (S_RS_R, S_RS_W, :473-489). A slot outside [min, next), an unreachable
  record, a generation other than the id's, or the retired generation answers status 1 (cause 25) and ends the
  operation.
- Every write of a CREVOKE comes later, from states reached only after S_RS_W: S_K_M_WR, S_KR_WR, the S_KS_* splice,
  S_K_SP_*, S_K_NODE*, and the header write-back S_H_Q* (the t_write sites at :628-810). The header states S_O_H* before
  S_RS only read.
- So a stale operand changes no record, as the paper says.
- What the sentence does not cover: a check that fails in a LATER chunk (a corrupted descendant) finds earlier
  chunks already released. "Every check of an operation precedes its first write" (:34) holds per chunk, not per
  instruction, since 4(b).

**Regions and munmap** (kernel).
- Each region's block is derived from R by root_derive, a CDERIVE (CV_DERIVE, funct7 0x51; sublet-table.c :174,
  :504-545). The loader grants the program's regions (sublet_grant_slot, :724-741), and mmap makes a block per mapping
  (sublet-blocks.c :244).
- munmap of a protected mapping goes through sublet_munmap (mm/mmap.c :2782-2783, ef8ee180f4fa0d69):
  - it resolves the handle to its block and inspects the reference's record (block_request, sublet-blocks.c :35-75,
    b3822f1594e23abc);
  - sublet_unmap_commit runs "immediately before detaching the VMAs" (:96-123; mm/mmap.c :2406);
  - it calls sublet_ref_revoke, which is an LDC of the kernel's slot followed by CREVOKE (CV_KILL, funct7 0x52;
    sublet-table.c :813-826), when the unmap removes the block's LAST pages (pages_last, :65-69).
- So the revoke precedes the pages' removal, as the paper says. An unmap of part of a region does not end the
  region's lifetime; the block stays live with fewer pages. The paper's "to free a whole region" is the case that
  revokes.
- The revoked block is dropped afterwards, so its address can never be reached through it again (request_finish,
  :124-133).

**"A dedicated mode".**
- On Line B, a protected program runs in ordinary U mode. The processor treats U as protected exactly when srevroot
  != 0 (csr_regfile.sv :3014, cap_data_ctx_o[0]; :161-162 names it "protected U (U, srevroot != 0)"), and the kernel
  sets srevroot per process.
- The capability mode of the earlier lines (CAPENTER, capmode) is gone under the Sublet format: capmode is tied to 0
  (:348-349) and CAPENTER is illegal at decode.
- So "a dedicated mode with user-space privileges" reads, on Line B, as "user mode with the process's lifetime table
  installed". Unprotected programs run in the same U mode with srevroot = 0. Whether to keep the word "mode" is the
  lead's call.
