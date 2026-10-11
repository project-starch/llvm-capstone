#include "corpus.h"
#include <stdint.h>

/* VVCSH ends with entry_point_start_ctu (ps.h:265), and a VVCSH is itself a
 * member of the slice context -- every consumer reaches it as `&lc->sc->sh`
 * (ctu.c:532 and eight other sites). So a write past the END of VVCSH's last
 * member lands on the ENCLOSING struct's next member, and the whole thing is
 * ONE allocation. That is why this is a sub-object crossing and not a crossing
 * of the allocation bound: the array is last in its own struct but not last in
 * the allocation.
 *
 * VVC_MAX_ENTRY_POINTS is VVC_MAX_TILE_COLUMNS * 135 (vvc.h:153), which is a
 * few thousand. It is REDUCED here, because the defect is that j is never
 * bounded at all -- the array's real length changes how many iterations it
 * takes to leave, not whether it leaves. The reduction is stated rather than
 * silent, and the offsets are asserted below. */
#define ENTRY_POINTS 64 /* reduced from VVC_MAX_TILE_COLUMNS * 135 */
#define TAIL_WORDS 8

struct vvc_slice_context {
  struct {
    uint8_t cu_qp_delta_subdiv;
    uint8_t cu_chroma_qp_offset_subdiv;
    uint32_t entry_point_start_ctu[ENTRY_POINTS]; /* last member of VVCSH */
  } sh;
  uint32_t tail[TAIL_WORDS]; /* the enclosing struct's next member */
};

__attribute__((noinline, used)) static void
write_probe(volatile uint32_t *p, uint32_t v) {
  *p = v;
}

FF2_CASE(6) {
  /* Case 6 -- VVC slice-header entry points, fix a809a784ec. SUB-OBJECT: an
   * UNBOUNDED four-byte write walking off the end of one struct member into the
   * next, inside a single allocation.
   *
   * sh_entry_points at the pin (ps.c:1469-1483) is:
   *
   *     for (int i = 1, j = 0; i < sh->num_ctus_in_curr_slice; i++) {
   *         ...
   *         if (<a tile or entropy-sync boundary>) {
   *             sh->entry_point_start_ctu[j++] = i;
   *         }
   *     }
   *
   * j counts boundaries and is compared against nothing. A slice whose every
   * CTU starts a new tile row makes j advance on every iteration, so the writes
   * run off the array and keep going for as long as the CTU count lasts. The
   * fix returns AVERROR_INVALIDDATA once j reaches VVC_MAX_ENTRY_POINTS, and
   * changes the function's return type to carry that out -- which is why the
   * diff touches sh_derive too.
   *
   * This case is a MAGNITUDE case, and that is deliberate: the write crosses
   * the member bound on its first step past the array and keeps crossing. It is
   * the one row in this corpus where the crossing could in principle be made to
   * leave the whole allocation, and the reduction keeps it inside so that the
   * sub-object claim is what is measured. How far it ran is reported.
   *
   * NOTHING WE HAVE CAN CATCH THIS: every write stays inside one allocation. */
  struct vvc_slice_context *sc = av_refstruct_allocz(sizeof *sc);
  CHECK(sc, 641);
  /* The claim: one past the array IS the enclosing struct's next member. */
  CHECK((char *)&sc->sh.entry_point_start_ctu[ENTRY_POINTS] == (char *)&sc->tail[0],
        642);

  for (unsigned i = 0; i < TAIL_WORDS; i++)
    sc->tail[i] = 0xA5A5A5A5u;

  /* Every CTU starts a boundary, so j advances every iteration. The CTU count
   * is chosen to overrun the array by exactly TAIL_WORDS, so the overrun stays
   * inside the allocation and the damage is bounded and checkable. */
  const unsigned num_ctus = ENTRY_POINTS + TAIL_WORDS;
  unsigned j = 0, past = 0;
  int stopped_early = 0;

  for (unsigned i = 0; i < num_ctus; i++) {
    if (fixed && j >= ENTRY_POINTS) { /* the fix's bound */
      stopped_early = 1;
      break;
    }
    if (j >= ENTRY_POINTS)
      past++;
    write_probe(&sc->sh.entry_point_start_ctu[j], i);
    j++;
  }

  unsigned clobbered = 0;
  for (unsigned i = 0; i < TAIL_WORDS; i++)
    if (sc->tail[i] != 0xA5A5A5A5u)
      clobbered++;

  printf("cap=%zu entries=%u past=%u clobbered=%u\n",
         sizeof sc->sh.entry_point_start_ctu, j, past, clobbered);

  FF2_VERDICT(!fixed && past == TAIL_WORDS && clobbered == TAIL_WORDS,
              fixed && stopped_early && clobbered == 0,
              "j was bounded by nothing, so the writes walked 8 words off the end of "
              "entry_point_start_ctu and overwrote the next member, inside one allocation",
              "the fix returns AVERROR_INVALIDDATA once j reaches the array's length");
  av_refstruct_unref(&sc);
  return !fixed ? !(clobbered == TAIL_WORDS) : !!clobbered;
}
