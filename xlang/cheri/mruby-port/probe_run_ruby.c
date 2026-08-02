/* Stage 7: init now completes with MRB_USE_METHOD_T_STRUCT. Can the VM
 * actually RUN Ruby? Uses the public API only (mrb_open + mrb_load_string),
 * bypassing the CLI driver, which is where the remaining SIGBUS must live. */
#include <mruby.h>
#include <mruby/compile.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const char *why(int sig, int c) {
  if (sig == SIGBUS) return c == 1 ? "BUS_ADRALN (misaligned capability)" : "BUS_other";
  switch (c) { case 1: return "CHERI_BOUNDS"; case 2: return "CHERI_TAG";
               case 5: return "CHERI_PERM"; default: return "other"; }
}
static void on_fault(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  printf("MRBFAULT sig=%d si_code=%d %s si_addr=%#p\n",
         sig, si->si_code, why(sig, si->si_code), si->si_addr);
  fflush(stdout);
  _exit(70);
}

int main(void) {
  struct sigaction sa; memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_fault; sa.sa_flags = SA_SIGINFO;
  sigaction(SIGPROT, &sa, NULL);
  sigaction(SIGBUS, &sa, NULL);

  printf("MRBPROBE A_start\n"); fflush(stdout);
  mrb_state *mrb = mrb_open();
  printf("MRBPROBE B_open_full=%s\n", mrb ? "ok" : "NULL"); fflush(stdout);
  if (!mrb) return 1;

  mrb_load_string(mrb, "1 + 1");
  printf("MRBPROBE C_arith_ok\n"); fflush(stdout);

  mrb_load_string(mrb, "puts 'HELLO_FROM_PURECAP_MRUBY'");
  printf("MRBPROBE D_puts_ok\n"); fflush(stdout);

  mrb_load_string(mrb, "a = [1,2,3].map { |x| x * 2 }; puts a.inspect");
  printf("MRBPROBE E_blocks_ok\n"); fflush(stdout);

  FILE *f = fopen("row10.rb", "r");
  if (f) { mrb_load_file(mrb, f); fclose(f);
           printf("MRBPROBE F_trigger_ran\n"); fflush(stdout); }
  else   { printf("MRBPROBE F_no_trigger_file\n"); fflush(stdout); }

  mrb_close(mrb);
  printf("MRBPROBE G_all_ok\n"); fflush(stdout);
  return 0;
}
