#include "metadata_contract.h"
#include "runtime_materialize_helpers.h"

struct holder {
  int (*fn)(void);
};

enum {
  OBJECT_HOLDER = 0,
};

static struct holder gHolder;

static int helper(void) { return 0x12345678u; }

static const unsigned char kHolderTemplate[sizeof(struct holder)] = {0};

static const struct static_cap_global_desc kGlobalDescs[] = {
    {
        OBJECT_HOLDER,
        sizeof(struct holder),
        16,
        0,
        0,
        1,
    },
};

static const struct static_cap_slot_desc kSlotDescs[] = {
    {
        OBJECT_HOLDER,
        0,
        STATIC_CAP_SLOT_FUNCTION,
        0,
        0,
    },
};

static void materialize_holder_from_descriptors(void) {
  const struct static_cap_global_desc *holder_desc = &kGlobalDescs[OBJECT_HOLDER];
  unsigned i;

  static_cap_copy_template_bytes((unsigned char *)&gHolder, kHolderTemplate,
                                 holder_desc->size);

  for (i = holder_desc->first_slot_index;
       i < holder_desc->first_slot_index + holder_desc->num_slots; ++i) {
    const struct static_cap_slot_desc *slot = &kSlotDescs[i];

    switch (slot->slot_kind) {
    case STATIC_CAP_SLOT_FUNCTION:
      static_cap_store_function_slot((unsigned char *)&gHolder,
                                     slot->field_offset, helper);
      break;
    case STATIC_CAP_SLOT_STRING_OBJECT:
    case STATIC_CAP_SLOT_GLOBAL_OBJECT:
    case STATIC_CAP_SLOT_NULL:
    default:
      static_cap_zero_bytes((unsigned char *)&gHolder + slot->field_offset,
                            sizeof(void *));
      break;
    }
  }
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  materialize_holder_from_descriptors();
  *res = (unsigned)gHolder.fn();
}

