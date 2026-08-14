/* The first program in a Capstone domain whose I/O goes through musl.
 *
 * Nothing here is Capstone-aware: it includes <unistd.h> and calls write().
 * That call goes musl write() -> __syscall1 -> __capstone_hostcall (our
 * arch/capstone64 layer) -> __capstone_yield -> host -> real write(2).
 *
 * The entry point is capstone_main() rather than main() because
 * runtime/hostcall.c owns domain_main: the first two domain entries carry the
 * shared regions and only the third runs the program.
 *
 * TWO writes, of DIFFERENT lengths, deliberately. One write proves the outward
 * path only. The second can only be reached by returning from the first, so it
 * is what shows a syscall RETURNED rather than merely fired, and the differing
 * length means a duplicated round cannot be mistaken for a second one.
 */
#include <string.h>
#include <unistd.h>

#define MUSL_HELLO_MSG1 "musl-hello: write #1 through musl in a domain\n"
#define MUSL_HELLO_MSG2 "musl-hello: write #2, so write() RETURNED\n"

/* PADDING, SIZED FOR STACK HEADROOM -- and NOT for the reason first written here.
 *
 * The first comment claimed padding was needed because "the monitor splits the
 * code capability at a fixed 0x1000 offset". That is RETRACTED: reading
 * create_domain (sbi_capstone.c:291) and the module (capstone.c:83) gives the
 * actual arithmetic, and it does not say that.
 *
 *   module:   dom_tot = code_len + 1536;  pages = ceil(dom_tot / 4096)
 *             tot_size = 4096 * 2^ceil(log2(pages))        <-- power of two
 *   monitor:  [base, base+code_len)                        = dom_code
 *             [base+code_len, +1536)                       = dom_seal (context)
 *             [base+code_len+1536, base+tot_size)          = dom_data
 *
 * By that arithmetic a 1232-byte image is fine (dom_data = 1328), so the padding
 * did not fix the first failure -- that run was `__CAPSTONE_INFRA_FLAKE__
 * phase=boot-login`, a slow boot, and I mistook it for the split fragility.
 * ISSUES.md's 416-byte helper_cssplit case is also not explained by this
 * arithmetic, so its cause remains open; nothing here should be read as closing
 * it.
 *
 * What the padding IS good for: crossing a page boundary so the power-of-two
 * rounding hands back a whole extra page as dom_data. Unpadded, code_len ~2320
 * gives tot_size 4096 and dom_data 240 bytes. 48 words of padding pushes
 * code_len past 2560, tot_size becomes 8192, and dom_data becomes ~3950 -- which
 * matters because the yield frame alone is 256 bytes. Bigger padding makes
 * dom_data SMALLER again, which is the opposite of useful.
 */
#ifndef MUSL_HELLO_PAD_WORDS
#define MUSL_HELLO_PAD_WORDS 48
#endif
__attribute__((used, retain)) volatile const unsigned long __pad[MUSL_HELLO_PAD_WORDS] = {1};

/* The proven path: hostcall.c's own WRITE_STDOUT, no musl above it. */
extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);

#define SAY(s) __capstone_hc_write(1, (s), sizeof(s) - 1)

/* A LADDER, because the first attempt at this program died with
 * `helper_cscincoffset: Assertion rs1_v->tag failed` and produced NO output at
 * all -- one bit of information, and not even about which layer. Each rung
 * prints through the layer below it BEFORE exercising the layer above, so the
 * last line on the console names the rung that faulted.
 *
 * Rung 1 also settles a question the yield probe could not: the probe has its
 * own domain_main, so it never exercised hostcall.c's globals or its round
 * function.
 */
int capstone_main(void) {
  ssize_t first, second;

  if (__pad[0] != 1)
    return 1;

  SAY("S1: hostcall.c direct write ok\n");

  /* Rung 2: musl. write() -> __syscall1 -> __capstone_hostcall -> yield. */
  first = write(1, MUSL_HELLO_MSG1, strlen(MUSL_HELLO_MSG1));
  SAY("S2: musl write() RETURNED\n");
  if (first != (ssize_t)strlen(MUSL_HELLO_MSG1))
    return 2;

  second = write(1, MUSL_HELLO_MSG2, strlen(MUSL_HELLO_MSG2));
  if (second != (ssize_t)strlen(MUSL_HELLO_MSG2))
    return 3;

  return 0;
}
