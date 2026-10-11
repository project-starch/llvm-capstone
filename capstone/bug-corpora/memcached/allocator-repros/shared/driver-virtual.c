/* main() for the VIRTUAL Capstone process build with the port's Sublet authority (MCP_SUBLET): the
 * corpus's `virtual-nested-pools` arm. The ledger is the one every arm shares (src/shared/leases.c);
 * underneath it the Sublet authority, over a payload the virtual heap LENDS as one linear capability
 * (capstone_borrow_aligned_block, aligned to MCP_GRAIN), so a chunk or object freed in mode 1 is
 * revoked in place.
 *
 * driver.c is compiled here unchanged, so every other build of it stays byte-identical. Only main()
 * is replaced: the mode is an argument (`fixed|buggy N 0|1`) and the payload is lent, not taken
 * from aligned_alloc. The result lines are driver.c's, so one observer reads every arm. */
#define main mc_driver_hosted_main
#include "driver.c"
#undef main
#include "borrow-aligned-block.h"

int main(int argc, char **argv) {
  if (argc != 4 || (strcmp(argv[1], "fixed") && strcmp(argv[1], "buggy")) ||
      (strcmp(argv[3], "0") && strcmp(argv[3], "1"))) {
    fprintf(stderr, "CONTROL-FAILED usage: fixed|buggy N 0|1\n");
    return 75;
  }
  int fixed = !strcmp(argv[1], "fixed");
  if (atoi(argv[2]) != mc_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            mc_case_number, argv[2]);
    return 75;
  }
  unsigned mode = (unsigned)(argv[3][0] - '0');
  void *metadata = aligned_alloc(4096, MCP_META_BYTES);
  capstone_cap_slot lent, head, tail;
  if (!metadata ||
      !capstone_borrow_aligned_block(MCP_PAYLOAD_BYTES, MCP_GRAIN, &lent, &head, &tail)) {
    fprintf(stderr, "CONTROL-FAILED the virtual heap lent no %lu-byte payload\n",
            (unsigned long)MCP_PAYLOAD_BYTES);
    return 75;
  }
  mcp_meta_init(metadata);
  mcp_payload_init(capstone_cap_load(&lent));
  mcp_set_mode(mode);
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  printf("case=%d arm=%s mode=%u\n", mc_case_number, fixed ? "fixed" : "buggy", mode);
  struct mc_outcome o = {0};
  mc_case_body(fixed, &o);
  struct mcp_header stats = {0};
  mcp_stats(&stats);
  printf("unit_reissued=%d accessed_through_stale=%d damage=%d "
         "chunk_reuses=%llu object_reuses=%llu\n",
         o.unit_reissued, o.accessed_through_stale, o.damage,
         (unsigned long long)stats.chunk_reuses,
         (unsigned long long)stats.object_reuses);
  int defect = !fixed && o.accessed_through_stale && o.damage;
  int held_up = fixed && !o.accessed_through_stale && !o.damage;
  printf("VERDICT %s%s\n",
         defect ? "DEFECT-REPRODUCED " : held_up ? "FIXED " : "INCONCLUSIVE",
         defect ? o.defect_text : held_up ? o.fixed_text : "");
  return !fixed ? !defect : !held_up;
}
