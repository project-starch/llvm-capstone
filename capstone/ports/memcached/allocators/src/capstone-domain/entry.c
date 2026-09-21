/* The domain entry: four shared regions arrive first -- report, metadata,
 * trace, payload -- then one call runs the program built through the seam.
 * The shape is the pymalloc and APR ports'; only the names and the result word
 * differ, so the guest loader and the corpus runner can tell the ports apart.
 * slabs_init runs with upstream's defaults and no preallocation: pages are
 * asked for one at a time, which is where the adapter meets them. */
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
static struct mcp_header *report;
static const struct mcp_header *trace;
static void *metadata;
static unsigned shares;
static unsigned long *domain_result;
unsigned char mcp_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void mcp_fail(unsigned code) {
  if (report)
    report->status = code;
  if (domain_result)
    *domain_result = code;
  __asm__ volatile("1: auipc t0, %%pcrel_hi(mcp_exit_frame)\n"
                   "addi t0, t0, %%pcrel_lo(1b)\n"
                   ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
                   ".insn i 0x5b, 0x3, sp, 0(t0)\n"
                   ".insn i 0x5b, 0x3, ra, 16(t0)\nret\n" ::
                       : "memory");
  __builtin_unreachable();
}
void mcp_entry(unsigned long *result, unsigned func) {
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
      mcp_payload_init(result);
      break;
    default:
      mcp_fail(401);
    }
    return;
  }
  domain_result = result;
  if (shares != 4)
    mcp_fail(402);
  mcp_meta_init(metadata);
  mcp_set_mode(report->mode);
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  mcp_replay(trace, report);
  *result = 42047;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(mcp_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj mcp_entry\n");
