#ifndef CAPSTONE_RUNTIME_QEMU_STATIC_CAP_GLOBALS_RUNTIME_MATERIALIZE_HELPERS_H
#define CAPSTONE_RUNTIME_QEMU_STATIC_CAP_GLOBALS_RUNTIME_MATERIALIZE_HELPERS_H

#include <stdint.h>

/*
 * Small helper layer for reduced runtime-side materialization experiments.
 *
 * These helpers are intentionally tiny and local to the probe. They represent
 * the common primitive operations that both eager and lazy runtime-side
 * materialization would need:
 *
 * - copy raw template bytes into writable object storage,
 * - store a function capability into a slot,
 * - store a pointer/global capability into a slot,
 * - clear a slot.
 */

static inline void static_cap_copy_template_bytes(unsigned char *dst,
                                                  const unsigned char *src,
                                                  uint32_t size) {
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = src[i];
}

static inline void static_cap_zero_bytes(unsigned char *dst, uint32_t size) {
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = 0;
}

static inline void static_cap_store_function_slot(unsigned char *object_base,
                                                  uint32_t field_offset,
                                                  int (*fn)(void)) {
  *(int (**)(void))(object_base + field_offset) = fn;
}

static inline void static_cap_store_ptr_slot(unsigned char *object_base,
                                             uint32_t field_offset,
                                             const void *ptr) {
  *(const void **)(object_base + field_offset) = ptr;
}

#endif

