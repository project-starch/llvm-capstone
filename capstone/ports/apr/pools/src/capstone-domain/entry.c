/* The domain entry: four shared regions arrive first -- report, metadata,
 * trace, payload -- then one call runs the program built through the seam.
 * The shape is the pymalloc port's; only the names and the result word differ,
 * so the guest loader and the corpus runner can tell the two apart. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
static struct aprp_header *report;
static const struct aprp_header *trace;
static void *metadata;
static unsigned shares;
static unsigned long *domain_result;
unsigned char aprp_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void aprp_fail(unsigned code) {
  if (report)
    report->status = code;
  if (domain_result)
    *domain_result = code;
  __asm__ volatile("1: auipc t0, %%pcrel_hi(aprp_exit_frame)\n"
                   "addi t0, t0, %%pcrel_lo(1b)\n"
                   ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
                   ".insn i 0x5b, 0x3, sp, 0(t0)\n"
                   ".insn i 0x5b, 0x3, ra, 16(t0)\nret\n" ::
                       : "memory");
  __builtin_unreachable();
}
void aprp_entry(unsigned long *result, unsigned func) {
  if (func == 1) {
    switch (shares++) {
    case 0:
      report = (void *)result;
      break;
    case 1:
      metadata = result;
      break;
    case 2:
      trace = (void *)result;
      break;
    case 3:
      aprp_payload_init(result);
      break;
    default:
      aprp_fail(401);
    }
    return;
  }
  domain_result = result;
  if (shares != 4)
    aprp_fail(402);
  aprp_meta_init(metadata);
  aprp_set_mode(report->mode);
  if (apr_pool_initialize() != APR_SUCCESS)
    aprp_fail(403);
  aprp_replay(trace, report);
  apr_pool_terminate();
  *result = 42046;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(aprp_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj aprp_entry\n");
