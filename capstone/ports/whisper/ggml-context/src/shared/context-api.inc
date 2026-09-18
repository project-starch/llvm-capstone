/* Expose the actual internal allocator for replay, without tensor kernels. */
size_t wg_object_header_size(void) { return GGML_OBJECT_SIZE; }
void *wg_object_alloc(struct ggml_context *ctx, unsigned type, size_t n) {
  GGML_ASSERT(type <= GGML_OBJECT_TYPE_WORK_BUFFER);
  struct ggml_object *o = ggml_new_object(ctx, type, n);
  if (!o)
    return NULL;
  void *p = (char *)ctx->mem_buffer + o->offs;
#ifdef WG_DOMAIN
  uintptr_t base = (uintptr_t)p;
  /* SHRINK cannot encode an empty interval. A zero-byte allocation keeps
   * its address but carries no memory authority; no payload byte exists. */
  if (!n)
    return (void *)base;
  return __builtin_capstone_cap_shrink(p, base, base + n);
#else
  return p;
#endif
}
