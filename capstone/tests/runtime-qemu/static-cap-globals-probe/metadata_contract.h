#ifndef CAPSTONE_RUNTIME_QEMU_STATIC_CAP_GLOBALS_METADATA_CONTRACT_H
#define CAPSTONE_RUNTIME_QEMU_STATIC_CAP_GLOBALS_METADATA_CONTRACT_H

/*
 * Minimal strategy-independent metadata contract for reduced static/global
 * capability-bearing objects in the LLVM-generated domain path.
 *
 * This is not yet wired into the runtime. It is a compact, concrete shape for
 * the information that both eager and lazy runtime-side materialization would
 * need.
 */

#include <stdint.h>

enum static_cap_global_slot_kind {
  STATIC_CAP_SLOT_NULL = 0,
  STATIC_CAP_SLOT_FUNCTION = 1,
  STATIC_CAP_SLOT_GLOBAL_OBJECT = 2,
  STATIC_CAP_SLOT_STRING_OBJECT = 3,
};

struct static_cap_global_desc {
  uint32_t object_id;
  uint32_t size;
  uint32_t align;
  uint32_t template_offset;
  uint32_t first_slot_index;
  uint32_t num_slots;
};

struct static_cap_slot_desc {
  uint32_t object_id;
  uint32_t field_offset;
  uint32_t slot_kind;
  uint32_t target_object_id;
  uint64_t target_addend;
};

#endif

