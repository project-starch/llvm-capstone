#include "port.h"
static struct wm_header *report;
static const struct wm_header *trace;
static void *metadata;
static unsigned shares;
static unsigned long *domain_result;
unsigned char wm_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void wm_fail(unsigned code) {
  if (report)
    report->status = code;
  if (domain_result)
    *domain_result = code;
  __asm__ volatile("1: auipc t0, %%pcrel_hi(wm_exit_frame)\n"
                   "addi t0, t0, %%pcrel_lo(1b)\n"
                   ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
                   ".insn i 0x5b, 0x3, sp, 0(t0)\n"
                   ".insn i 0x5b, 0x3, ra, 16(t0)\nret\n" ::
                       : "memory");
  __builtin_unreachable();
}
void wm_entry(unsigned long *result, unsigned func) {
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
      wm_init_backing(metadata, result, report->mode);
      break;
    default:
      wm_fail(401);
    }
    return;
  }
  domain_result = result;
  if (shares != 4)
    wm_fail(402);
  wm_replay(trace, report);
  *result = 42060;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(wm_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj wm_entry\n");
