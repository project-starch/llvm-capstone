/* main() for the lifetime controls, on both virtual arms: `ctl-NN N VARIANT`.
 *
 *   virtual-nested-pools (MC_CAPSTONE_SUBLET, patch 0006): slabs.c takes its pages from virtual
 *     mallocng and derives and revokes every chunk itself; no ledger. As shared/driver-virtual.c.
 *   virtual-malloc (MCP_STOCK_MALLOC, the release build): the stock ledger in mode 0, each page one
 *     virtual-mallocng object, no revocation anywhere. As shared/driver.c.
 *
 * stdout is unbuffered: a fault ends the process, and the ACCESS-BEGIN line of the access that
 * faulted must already be on the console. Exit 0 and CTL-COMPLETED only when every access and every
 * check of the control ran; 75 when a check refused the run (CONTROL-FAILED, never a verdict). */
#include "controls.h"
#include <stdlib.h>

_Noreturn void mcp_fail(unsigned code) {
  printf("CONTROL-FAILED %u\n", code);
  exit(75);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc != 3) {
    printf("CONTROL-FAILED usage: ctl-NN N VARIANT\n");
    return 75;
  }
  if (atoi(argv[1]) != ctl_number) {
    printf("CONTROL-FAILED program is control %d, run asked for %s\n", ctl_number, argv[1]);
    return 75;
  }
#ifdef MC_CAPSTONE_SUBLET
  const char *arm = "virtual-nested-pools";
#else
  const char *arm = "virtual-malloc";
  void *metadata = aligned_alloc(4096, MCP_META_BYTES);
  void *payload = aligned_alloc(4096, MCP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    mcp_fail(702);
  mcp_meta_init(metadata);
  mcp_payload_init(payload);
  mcp_set_mode(0);
#endif
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  printf("CTL control=%d variant=%s arm=%s\n", ctl_number, argv[2], arm);
  ctl_body(argv[2]);
  printf("CTL-COMPLETED control=%d variant=%s arm=%s\n", ctl_number, argv[2], arm);
  return 0;
}
