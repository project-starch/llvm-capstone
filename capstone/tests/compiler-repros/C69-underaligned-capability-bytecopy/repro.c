/* C-32-adjacent? No: an UNDER-ALIGNED capability access. Reduction of libavfilter's FF_FIELD_AT. */
struct cfg { void *formats, *samplerates, *channel_layouts; };
struct link { int pad; struct cfg incfg, outcfg; };
struct merger { unsigned offset; };
#define FF_FIELD_AT(type, off, obj) (*(type *)((char *)&(obj) + (off)))

void *field_at(struct link *l, const struct merger *m) {
  return FF_FIELD_AT(void *, m->offset, l->incfg);      /* runtime offset -> align 1 */
}
void *field_direct(struct link *l) {
  return l->incfg.channel_layouts;                      /* constant offset -> naturally aligned */
}
void field_store(struct link *l, const struct merger *m, void *v) {
  FF_FIELD_AT(void *, m->offset, l->incfg) = v;         /* the STORE half */
}
