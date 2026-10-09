/* FULL CONFIGURATION, tshark: wmem's chunk port (ports/wireshark/wmem/src/allocators/sublet/chunks.c
 * over ports/wireshark/app/src/tsapp-wmem-chunks.c, as TSAPP_HEAP=chunks builds it) is linked into a
 * plain case and brought up before main(): a block is opened LINEAR from the Sublet heap, its first
 * chunk split, issued, written and retired -- the port's per-chunk revoke. The block stays open for
 * the whole case, so the case runs with the port live, as inside tshark on the chunks arm.
 *
 * The case's objects come straight from g_malloc/malloc, not from wmem, so this measures that the
 * port changes nothing for a direct-allocation bug. */
#include <stdio.h>
#include <stdlib.h>

#include "chunks.h"

static struct wm_block_auth full_config_block;
static struct wm_chunk_auth full_config_chunk, full_config_rest;

__attribute__((constructor(65535))) static void full_config_wireshark(void) {
  const size_t hdr = 32, size = (size_t)1 << 16, len = 256;
  wm_chunks_init(1); /* the protected mode, as the chunks arm's own constructor sets it */
  wm_block_open(&full_config_block, size, hdr);
  wm_chunk_adopt(&full_config_block, &full_config_chunk);
  size_t at = full_config_block.base + hdr;
  wm_chunk_split(&full_config_chunk, at + len, &full_config_rest);
  wm_chunk_issue(&full_config_block, &full_config_chunk, at, len);
  unsigned char *p = wm_chunk_bytes(&full_config_chunk, at, len);
  p[0] = 1;
  unsigned long out = capstone_cap_type(&full_config_chunk.slot);
  wm_chunk_retire(&full_config_chunk); /* one revoke: the chunk's every alias dies */
  /* The give leaves the region LINEAR in the slot again; while the chunk was out the slot held
   * its handle. (sublet.h's counters are per translation unit, so this file cannot count
   * chunks.c's revoke; the slot's type is the evidence.) */
  unsigned long back = capstone_cap_type(&full_config_chunk.slot);
  printf("FULLCONFIG wireshark wmem=chunk-port live block=%zu chunk=%zu slot_out=%lu slot_back=%lu\n",
         size, len, out, back);
  fflush(stdout);
  if (back != CAPSTONE_CAP_LINEAR || out == CAPSTONE_CAP_LINEAR) {
    printf("FULLCONFIG-FAILED wireshark: the retire did not give the chunk back\n");
    fflush(stdout);
    exit(75);
  }
}
