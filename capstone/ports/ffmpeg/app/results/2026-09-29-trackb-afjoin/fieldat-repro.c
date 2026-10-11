/* Reduction of libavfilter's FF_FIELD_AT(void *, m->offset, link->incfg) on capstone64. */
struct cfg { void *formats, *samplerates, *channel_layouts; };
struct link { int pad; struct cfg incfg, outcfg; };
struct merger { unsigned offset; };
#define FF_FIELD_AT(type, off, obj) (*(type *)((char *)&(obj) + (off)))
void *field_at(struct link *l, const struct merger *m)
{
    return FF_FIELD_AT(void *, m->offset, l->incfg);
}
void *field_direct(struct link *l)
{
    return l->incfg.channel_layouts;
}
