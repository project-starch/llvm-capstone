// Descriptor-driven positive POC for runtime-side materialization.
//
// This uses the reduced metadata contract directly for one object plus one
// referenced string object. It intentionally stays narrow: the goal is to prove
// that the reduced case can be rebuilt from descriptors and raw templates,
// without yet committing to a fully generic runtime implementation.

#include "metadata_contract.h"
#include "runtime_materialize_helpers.h"

struct pair {
  int (*fn)(void);
  const char *name;
};

enum {
  OBJECT_PAIR = 0,
  OBJECT_STRING_OK = 1,
};

static struct pair g_pair;
static char g_ok[3];

static int helper(void) { return 0x12340000u; }

static const unsigned char kPairTemplate[sizeof(struct pair)] = {
  0,
};

static const unsigned char kOkTemplate[3] = {'o', 'k', '\0'};

static const struct static_cap_global_desc kGlobalDescs[] = {
    {
        OBJECT_PAIR,
        sizeof(struct pair),
        16,
        0,
        0,
        2,
    },
    {
        OBJECT_STRING_OK,
        3,
        1,
        0,
        2,
        0,
    },
};

static const struct static_cap_slot_desc kSlotDescs[] = {
    {
        OBJECT_PAIR,
        0x00,
        STATIC_CAP_SLOT_FUNCTION,
        0,
        0,
    },
    {
        OBJECT_PAIR,
        0x10,
        STATIC_CAP_SLOT_STRING_OBJECT,
        OBJECT_STRING_OK,
        0,
    },
};

static void materialize_reduced_case_from_descriptors(void) {
  const struct static_cap_global_desc *pair_desc = &kGlobalDescs[OBJECT_PAIR];
  const struct static_cap_global_desc *str_desc = &kGlobalDescs[OBJECT_STRING_OK];
  unsigned i;

  static_cap_copy_template_bytes((unsigned char *)&g_pair, kPairTemplate,
                                 pair_desc->size);
  static_cap_copy_template_bytes((unsigned char *)g_ok, kOkTemplate,
                                 str_desc->size);

  for (i = pair_desc->first_slot_index;
       i < pair_desc->first_slot_index + pair_desc->num_slots; ++i) {
    const struct static_cap_slot_desc *slot = &kSlotDescs[i];

    switch (slot->slot_kind) {
    case STATIC_CAP_SLOT_FUNCTION:
      static_cap_store_function_slot((unsigned char *)&g_pair,
                                     slot->field_offset, helper);
      break;
    case STATIC_CAP_SLOT_STRING_OBJECT:
      static_cap_store_ptr_slot((unsigned char *)&g_pair, slot->field_offset,
                                (const void *)(g_ok + slot->target_addend));
      break;
    case STATIC_CAP_SLOT_GLOBAL_OBJECT:
    case STATIC_CAP_SLOT_NULL:
    default:
      static_cap_zero_bytes((unsigned char *)&g_pair + slot->field_offset,
                            sizeof(void *));
      break;
    }
  }
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  materialize_reduced_case_from_descriptors();
  *res = (unsigned)(g_pair.fn() + (unsigned)g_pair.name[0]);
}


