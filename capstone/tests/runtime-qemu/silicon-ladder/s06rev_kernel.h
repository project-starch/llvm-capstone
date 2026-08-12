#ifndef S06REV_KERNEL_H
#define S06REV_KERNEL_H
/* MINIMAL REPRO for the SECOND fault -- the `mcause 25 INVALID_CAPABILITY` wedge that appears
 * whenever an S-06 fix repairs the data and SQLite runs deeper.
 *
 * ROOT-CAUSE HYPOTHESIS, derived from the RTL, which this rung is built to test:
 *
 *   Revocation nodes live in MEMORY, at CAP_REVNODE_MEM_BASE = 0xBFF0_0000
 *   (ariane_pkg.sv:590), and a validity query READS one back:
 *   capstone_rev_node.anvil:36-42 `get_rev_node` issues mem_ch.read_req and returns data.valid,
 *   which ex_stage.sv:1030 reconstructs from {rev_mem_rd_res_i.data_ruser[29:0], data_rdata}.
 *   So `valid` arrives through `ruser`.
 *
 *   That region is CACHEABLE -- CachedRegionAddrBase 0x8000_0000 + length 0x4000_0000 covers
 *   [0x8000_0000, 0xC000_0000) and the pool is [0xBFF0_0000, 0xC000_0000)
 *   (capstone_cv64a6_imafdc_sv39_config_pkg.sv:142-144).
 *
 *   But it is NEVER SHADOW-TAGGED. wt_axi_adapter.sv:139-145 gates `needs_tag` on
 *   in_data_region = [MEMORY_BASE, MEMORY_TOP) = [0x8000_0000, 0xBC2D_2D2D), which EXCLUDES the
 *   pool -- deliberately, per the comment and elaboration assert at :987-992, "Shadow tag writes
 *   must never reach the revnode region".
 *
 *   Therefore: a rev-node write sets cap_tag_q in L1 (ex_stage.sv:1044 forces a constant into
 *   data_wuser so |user is non-zero), but writes NO tag byte to DRAM. The cache is write-through,
 *   so if that line is later EVICTED and REFILLED, wt_dcache_mem.sv restores the tag from the
 *   shadow byte -- which was never written. The line comes back UNTAGGED, `ruser` is force-zeroed,
 *   the node reads valid = 0, and the next ldc/stc through that capability raises mcause 25.
 *
 * That shape matches every observation: it needs eviction pressure, so it appears only at scale
 * and only once a fix adds store traffic; it never reproduces in Verilator or on a hot 10 KB rung;
 * and rev_node_head at the wedges (606, 418, 249) is far below the 65536-entry pool, so it is not
 * exhaustion.
 *
 * THE EXPERIMENT. No capability needs to be MINTED: every ldc/stc already performs a validity
 * query on the revocation node of its ADDRESS capability (capstone_dyn_unit.anvil:337 for LDC,
 * :404 for STC). So the domain's own data capability suffices, and the first round trip caches the
 * node's line. Then evict by streaming through more than the 32 KB D-cache, and use it again. Both
 * arms are identical except for the eviction, which is the single variable.
 *
 * An earlier version of this rung called MREV to mint a node explicitly. That was wrong and QEMU
 * caught it: MREV requires a LINEAR operand (capstone_dyn_unit.anvil:81 raises
 * UNEXPECTED_CAP_TYPE otherwise) and an ordinary data pointer is NONLIN, so the rung faulted
 * before reaching the experiment.
 *
 *   arm HOT   round trip, round trip again              -> must return 1
 *   arm EVICT round trip, stream 64 KB, round trip again -> 1 if the node survived
 *                                                          NO RETURN (wedge) if the hypothesis holds
 *
 * The HOT arm is the control and it runs FIRST: if it does not return, the rung is broken rather
 * than the hypothesis confirmed, and the run carries no verdict. A wedge is itself the result
 * here, so this rung is expected to be the LAST domain in its boot.
 *
 *   returns 11 -> both arms survived: the hypothesis is WRONG, rev-node lines keep their tags
 *   returns 10 -> the hot arm worked and the evicted arm reported failure without faulting
 *   no return  -> the evicted arm wedged: hypothesis CONFIRMED (check k800 in the same boot)
 */

#define S06REV_STREAM_BYTES (64u * 1024u)   /* > the 32 KB D-cache, so a pass evicts everything */

__attribute__((aligned(16))) static unsigned char s06rev_stream[S06REV_STREAM_BYTES];
__attribute__((aligned(16))) static unsigned char s06rev_slot[64];

#define S06_LDC(out, addr) \
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(out) : "r"(addr))
#define S06_STC(val, addr) \
  __asm__ volatile(".insn s 0x5b, 0x4, %0, 0(%1)" :: "r"(val), "r"(addr) : "memory")

/* Touch every line of a 64 KB buffer. Volatile so it cannot be optimised away -- without that the
   whole eviction step vanishes and the EVICT arm silently becomes the HOT arm. */
static void s06rev_evict(void)
{
  volatile unsigned long *p = (volatile unsigned long *)s06rev_stream;
  unsigned i;
  for (i = 0; i < S06REV_STREAM_BYTES / 8u; i += 2u) p[i] = (unsigned long)i;
}

/* Round-trip a capability through memory. The ldc and the stc each query the validity of the
   revocation node belonging to the ADDRESS capability, which is the operation under test. */
static unsigned s06rev_use(int evict)
{
  void *back;
  void *cap = (void *)s06rev_slot;

  S06_STC(cap, s06rev_slot);        /* first round trip: caches the node's line */
  S06_LDC(back, s06rev_slot);

  if (evict) s06rev_evict();        /* THE ONLY DIFFERENCE BETWEEN THE TWO ARMS */

  S06_STC(back, s06rev_slot);       /* second round trip: the validity query that may see valid=0 */
  S06_LDC(back, s06rev_slot);
  return 1;
}

static unsigned s06rev_compute(void)
{
  unsigned hot = s06rev_use(0);     /* CONTROL first: if this does not return, the rung is broken */
  unsigned ev  = s06rev_use(1);     /* the probe: expected to wedge if the hypothesis holds */
  return hot * 10u + ev;
}
#endif /* S06REV_KERNEL_H */
