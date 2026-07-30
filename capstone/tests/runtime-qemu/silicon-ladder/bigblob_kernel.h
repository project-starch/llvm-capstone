#ifndef BIGBLOB_KERNEL_H
#define BIGBLOB_KERNEL_H
/* SQLite's INITIALIZER BLOB SIZE, and nothing else -- the last create-time input
 * no rung covers.
 *
 * WHAT THE MONITOR ACTUALLY DOES AT CREATE TIME. create_domain
 * (caplifive-system .../capstone-sbi/sbi_capstone.c:290-505) is parameterised by
 * exactly four numbers: base_addr, code_size, tot_size and gpoff. Everything
 * size-dependent in it derives from those:
 *
 *   repr_gran / split_size / data_off   from tot_size and code_size
 *   the blob COPY loop  (:493-499)      `for (ci = 0; ci < (code_size-gpoff)>>3; ci++)
 *                                         dom_data[ci] = dom_code[gpoff_c + ci];`
 *
 * The existing 2 MiB rungs pin three of the four:
 *
 *   rung      globals   gpoff      tot_size   blob (code_size-gpoff)   silicon
 *   bigwin    1         0x140000   2 MiB      88 B                     PASS
 *   bigmany   64        0x140000   2 MiB      ~1.6 KB                  PASS
 *   SQLite    1059      0x140000   2 MiB      78,760 B                 WEDGES
 *
 * So base/tot_size/gpoff/repr_gran are all covered and all pass. The one create-time
 * quantity still separating SQLite from a passing rung is the BLOB: 78,760 bytes,
 * i.e. 9,845 iterations of an M-mode word copy, against bigwin's 11. This rung sets
 * that number and holds everything else at bigwin's values.
 *
 * WHY BLOB SIZE IS A CREDIBLE CAUSE OF A POST-RETURN WEDGE. The observed failure is
 * not a fault inside create_dom -- the ioctl RETURNS and the machine dies during the
 * next syscall, 16 characters into the host's next write (commit c9630d6f). An M-mode
 * routine that runs ~3 orders of magnitude longer than any rung's is the one create-time
 * property that can leave the system in a state that only bites on the next trap: it is
 * long enough for an M-mode-visible interrupt to arrive mid-copy, and the monitor's
 * handler `while(1)`s on anything it does not recognise. The 2026-07-19 instance of
 * exactly this signature -- "the next userspace write emits 16 bytes then the board is
 * dead" -- was an M-mode trap the monitor could not service, not a domain fault.
 *
 * WHAT EACH OUTCOME MEANS.
 *   WEDGES  -> reproduced in a rung that rebuilds in seconds. The cause is create-time
 *              and blob-size-driven; SQLite's own code, its 1,059 records and its glibc
 *              host are all exonerated, and the bisection continues on this rung
 *              (halve/double the array) instead of on 35-minute SQLite iterations.
 *   PASSES  -> blob size is not it either. Every create_domain input is then covered by
 *              a passing rung, which moves the suspect off the monitor's create path and
 *              onto what is left: the 1,059-record entry glue (runs AFTER create, so it
 *              cannot explain a wedge before entry) or the SQLite HOST binary -- which,
 *              unlike every ladder controller, is linked against glibc. ladder_perf_ctl.c
 *              is freestanding by construction "(the board rejects glibc's hard-float
 *              fsd)" (ladder_perf_ctl.c:22-25), so no rung has ever tested a glibc host.
 *
 * SHAPE. One 78,656-byte initialized array (the copy path, blob_off >= 0) plus one
 * zero-init array (blob_off == -1, and the kernel stores into it, so the carved storage
 * is proven writable). Two records -- deliberately NOT 1,059: record count is bigmany's
 * variable and it already passes.
 *
 * The payload is a repeating 16-word unit so the .dom gzips to a few KB and the UART
 * transfer stays as fast as bigwin's. The hash mixes the INDEX into every word, so a
 * truncated, zeroed or displaced copy changes the checksum even though the source bytes
 * repeat; a displacement that happens to be a whole multiple of 64 bytes is the one
 * corruption this payload cannot see.
 */

/* 16 distinct words = 64 B, the repeating unit. */
#define BB_U16 \
  0x3B9ACA07u, 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, \
  0x27D4EB2Fu, 0x165667B1u, 0xD3A2646Cu, 0xFD7046C5u, \
  0xB55A4F09u, 0x9E3779B1u, 0x7FEB352Du, 0x846CA68Bu, \
  0x2545F491u, 0xBF58476Du, 0x94D049BBu, 0xA24BAED4u

/* Object-like, NOT function-like. A function-like BB_R2(x) does not work here:
   argument prescan expands BB_U16 before substituting it, so the nested call sees
   16 arguments instead of 1 and the preprocessor errors out. */
#define BB_R1    BB_U16
#define BB_R2    BB_R1, BB_R1
#define BB_R4    BB_R2, BB_R2
#define BB_R8    BB_R4, BB_R4
#define BB_R16   BB_R8, BB_R8
#define BB_R32   BB_R16, BB_R16
#define BB_R64   BB_R32, BB_R32
#define BB_R128  BB_R64, BB_R64
#define BB_R256  BB_R128, BB_R128
#define BB_R512  BB_R256, BB_R256
#define BB_R1024 BB_R512, BB_R512

/* 1229 units * 16 words * 4 B = 78,656 B, so that with the 80-byte init descriptor
   and the cap-table the loaded blob lands within a few dozen bytes of SQLite's
   78,760. Verify after building, do not assume:
     readelf -l <dom>            LOAD MemSiz - 0x140000 == blob size
   1229 = 1024 + 128 + 64 + 8 + 4 + 1. */
#define BB_N 19664

static unsigned bb_blob[BB_N] = {
  BB_R1024,
  BB_R128,
  BB_R64,
  BB_R8,
  BB_R4,
  BB_R1
};

/* Zero-init: the blob_off == -1 descriptor path, and the store below proves the
   carved storage is writable rather than merely readable. */
static unsigned bb_sink[8];

static unsigned bigblob_compute(void) {
  unsigned h = 2166136261u;
  for (unsigned i = 0; i < 8; i++) bb_sink[i] = i * 2654435761u + 17u;
  for (unsigned i = 0; i < BB_N; i++) {
    h ^= bb_blob[i] + i * 2654435761u;
    h *= 16777619u;
  }
  for (unsigned i = 0; i < 8; i++) { h ^= bb_sink[i]; h *= 16777619u; }
  return h;
}
#endif
