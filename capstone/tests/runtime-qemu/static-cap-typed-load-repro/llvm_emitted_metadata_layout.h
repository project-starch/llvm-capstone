#ifndef CAPSTONE_RUNTIME_QEMU_STATIC_CAP_TYPED_LOAD_REPRO_LLVM_EMITTED_METADATA_LAYOUT_H
#define CAPSTONE_RUNTIME_QEMU_STATIC_CAP_TYPED_LOAD_REPRO_LLVM_EMITTED_METADATA_LAYOUT_H

#include <stdint.h>

/*
 * Proposed minimal compiler-emitted metadata layout for reduced static/global
 * capability-bearing objects in the LLVM-generated Capstone domain path.
 *
 * This header is intentionally small and policy-neutral. It defines the exact
 * record layout that a first LLVM-side emission proof of concept could write
 * into a dedicated ELF section, while a runtime-side materializer walks the
 * records and reconstructs live capability-valued fields into writable object
 * storage.
 */

#define STATIC_CAP_METADATA_MAGIC 0x50414353u /* 'SCAP' */
#define STATIC_CAP_METADATA_VERSION 1u

/* Slot kinds describe how the runtime should reconstruct one field. */
enum static_cap_global_slot_kind {
  STATIC_CAP_SLOT_NULL = 0,
  STATIC_CAP_SLOT_FUNCTION = 1,
  STATIC_CAP_SLOT_GLOBAL_OBJECT = 2,
  STATIC_CAP_SLOT_STRING_OBJECT = 3,
};

/* Flags describing how to interpret the emitted slot target fields. */
enum static_cap_emitted_slot_target_flags {
  /* `target_ref` carries an ELF relocation against a symbol. */
  STATIC_CAP_TARGET_FLAG_RELOCATED_SYMBOL = 1u << 0,
};

/*
 * Section header for the proposed compiler-emitted metadata section.
 *
 * Layout in the section:
 *   [ static_cap_metadata_section_header ]
 *   [ static_cap_emitted_global_desc[object_count] ]
 *   [ static_cap_emitted_slot_desc[slot_count] ]
 *   [ raw template bytes blob ]
 */
struct static_cap_metadata_section_header {
  uint32_t magic;
  uint16_t version;
  uint16_t flags;
  uint32_t object_count;
  uint32_t slot_count;
  uint32_t template_bytes_size;
  uint32_t global_desc_size;
  uint32_t slot_desc_size;
};

/*
 * One record per materialized object.
 *
 * `template_offset` is relative to the start of the trailing template-bytes
 * blob described by the section header.
 */
struct static_cap_emitted_global_desc {
  uint32_t object_id;
  uint32_t size;
  uint32_t align;
  uint32_t template_offset;
  uint32_t first_slot_index;
  uint32_t num_slots;
};

/*
 * One record per capability-bearing field.
 *
 * Interpretation:
 * - STATIC_CAP_SLOT_FUNCTION:
 *     - `target_flags` includes STATIC_CAP_TARGET_FLAG_RELOCATED_SYMBOL
 *     - `target_ref` carries the addend-bearing relocated symbol reference
 *     - `target_object_id` is ignored
 * - STATIC_CAP_SLOT_GLOBAL_OBJECT / STATIC_CAP_SLOT_STRING_OBJECT:
 *     - `target_object_id` names another emitted object record
 *     - `target_addend` is the byte offset inside that target object
 *     - `target_ref` is zero
 * - STATIC_CAP_SLOT_NULL:
 *     - the runtime zeroes the field
 */
struct static_cap_emitted_slot_desc {
  uint32_t object_id;
  uint32_t field_offset;
  uint32_t slot_kind;
  uint32_t target_flags;
  uint32_t target_object_id;
  uint32_t reserved0;
  uint64_t target_addend;
  uint64_t target_ref;
};

#endif

