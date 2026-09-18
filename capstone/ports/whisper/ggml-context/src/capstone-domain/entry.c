#include "port.h"
static struct wg_header *report;
static const struct wg_header *trace;
static void *metadata;
static unsigned shares;
static unsigned long *domain_result;
unsigned char wg_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void wg_fail(unsigned code) {
  if (report)
    report->status = code;
  if (domain_result)
    *domain_result = code;
  __asm__ volatile("1: auipc t0, %%pcrel_hi(wg_exit_frame)\n"
                   "addi t0, t0, %%pcrel_lo(1b)\n"
                   ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
                   ".insn i 0x5b, 0x3, sp, 0(t0)\n"
                   ".insn i 0x5b, 0x3, ra, 16(t0)\nret\n" ::
                       : "memory");
  __builtin_unreachable();
}
void wg_entry(unsigned long *result, unsigned func) {
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
      wg_init_backing(metadata, result, report->mode);
      break;
    default:
      wg_fail(401);
    }
    return;
  }
  domain_result = result;
  if (shares != 4)
    wg_fail(402);
  wg_replay(trace, report);
  *result = 42050;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(wg_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj wg_entry\n");
