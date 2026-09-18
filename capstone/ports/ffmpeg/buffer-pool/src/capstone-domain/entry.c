#include "replay-engine.h"

static void *metadata;
static struct ff2_header *report;
static const struct ff2_header *trace;
static unsigned shares;
static unsigned long *domain_result;
unsigned char ff2_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void ff2_fail(unsigned code) {
  if (report)
    report->status = code;
  if (domain_result)
    *domain_result = code;
  __asm__ volatile("1: auipc t0, %%pcrel_hi(ff2_exit_frame)\n"
                   "addi t0, t0, %%pcrel_lo(1b)\n"
                   ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
                   ".insn i 0x5b, 0x3, sp, 0(t0)\n"
                   ".insn i 0x5b, 0x3, ra, 16(t0)\n"
                   "ret\n" ::
                       : "memory");
  __builtin_unreachable();
}
void ff2_entry(unsigned long *result, unsigned func) {
  if (func == 1) {
    switch (shares++) {
    case 0:
      report = (struct ff2_header *)result;
      break;
    case 1:
      metadata = result;
      break;
    case 2:
      trace = (const struct ff2_header *)result;
      break;
    case 3:
      ff2_payload_init(result, FF2_PAYLOAD_BYTES);
      break;
    default:
      ff2_fail(218);
    }
    return;
  }
  domain_result = result;
  if (shares != 4)
    ff2_fail(219);
  ff2_replay_run(trace, report, metadata);
  *result = 42044;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(ff2_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj ff2_entry\n");
