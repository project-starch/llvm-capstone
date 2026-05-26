#include <stdint.h>

struct holder {
  const char *name;
};

__attribute__((used)) static const struct holder kHolder = {
  "ok",
};

extern const unsigned char __llvm_static_cap_gct_begin[];
extern const unsigned char __llvm_static_cap_gct_end[];

#define STATIC_CAP_METADATA_MAGIC 0x50414353u
#define STATIC_CAP_METADATA_VERSION 1u

enum static_cap_global_slot_kind {
  STATIC_CAP_SLOT_NULL = 0,
  STATIC_CAP_SLOT_FUNCTION = 1,
  STATIC_CAP_SLOT_GLOBAL_OBJECT = 2,
  STATIC_CAP_SLOT_STRING_OBJECT = 3,
};

#define GCT_HEADER_MAGIC_OFFSET 0u
#define GCT_HEADER_VERSION_OFFSET 4u
#define GCT_HEADER_OBJECT_COUNT_OFFSET 8u
#define GCT_HEADER_SLOT_COUNT_OFFSET 12u
#define GCT_HEADER_TEMPLATE_BYTES_SIZE_OFFSET 16u
#define GCT_HEADER_GLOBAL_DESC_SIZE_OFFSET 20u
#define GCT_HEADER_SLOT_DESC_SIZE_OFFSET 24u

#define GCT_EXPECTED_OBJECT_COUNT 2u
#define GCT_EXPECTED_SLOT_COUNT 1u
#define GCT_EXPECTED_TEMPLATE_BYTES_SIZE 19u
#define GCT_EXPECTED_GLOBAL_DESC_SIZE 24u
#define GCT_EXPECTED_SLOT_DESC_SIZE 40u

#define GCT_SLOT_KIND_ABS_OFFSET 84u
#define GCT_TARGET_OBJECT_ID_ABS_OFFSET 92u

#define GCT_HOLDER_OBJECT_ID_ABS_OFFSET 28u
#define GCT_STRING_OBJECT_ID_ABS_OFFSET 52u

#define GCT_TEMPLATE_HOLDER_OFFSET 116u
#define GCT_TEMPLATE_HOLDER_SIZE 16u
#define GCT_TEMPLATE_STRING_OFFSET 132u
#define GCT_TEMPLATE_STRING_SIZE 3u

static struct holder gHolder;
static unsigned char gStringStorage[16];

static void copy_bytes(unsigned char *dst, const unsigned char *src, uint32_t size) {
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = src[i];
}

static void zero_bytes(unsigned char *dst, uint32_t size) {
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = 0;
}

static uint32_t read_u32(const unsigned char *base, uint32_t offset) {
  return ((uint32_t)base[offset + 0]) | ((uint32_t)base[offset + 1] << 8) |
         ((uint32_t)base[offset + 2] << 16) |
         ((uint32_t)base[offset + 3] << 24);
}

static uint16_t read_u16(const unsigned char *base, uint32_t offset) {
  return (uint16_t)(((uint16_t)base[offset + 0]) |
                    ((uint16_t)base[offset + 1] << 8));
}

static unsigned materialize_from_emitted_gct(void) {
  const unsigned char *gct = __llvm_static_cap_gct_begin;
  if (read_u32(gct, GCT_HEADER_MAGIC_OFFSET) != STATIC_CAP_METADATA_MAGIC)
    return 0xdead0002u;
  if (read_u16(gct, GCT_HEADER_VERSION_OFFSET) != STATIC_CAP_METADATA_VERSION)
    return 0xdead0003u;
  if (read_u32(gct, GCT_HEADER_OBJECT_COUNT_OFFSET) != GCT_EXPECTED_OBJECT_COUNT)
    return 0xdead0004u;
  if (read_u32(gct, GCT_HEADER_SLOT_COUNT_OFFSET) != GCT_EXPECTED_SLOT_COUNT)
    return 0xdead0005u;
  if (read_u32(gct, GCT_HEADER_TEMPLATE_BYTES_SIZE_OFFSET) !=
      GCT_EXPECTED_TEMPLATE_BYTES_SIZE)
    return 0xdead0006u;
  if (read_u32(gct, GCT_HEADER_GLOBAL_DESC_SIZE_OFFSET) != GCT_EXPECTED_GLOBAL_DESC_SIZE)
    return 0xdead0007u;
  if (read_u32(gct, GCT_HEADER_SLOT_DESC_SIZE_OFFSET) != GCT_EXPECTED_SLOT_DESC_SIZE)
    return 0xdead0008u;
  if (read_u32(gct, GCT_SLOT_KIND_ABS_OFFSET) != STATIC_CAP_SLOT_STRING_OBJECT)
    return 0xdead0009u;
  if (read_u32(gct, GCT_TARGET_OBJECT_ID_ABS_OFFSET) !=
      read_u32(gct, GCT_STRING_OBJECT_ID_ABS_OFFSET))
    return 0xdead000au;
  if (read_u32(gct, GCT_HOLDER_OBJECT_ID_ABS_OFFSET) ==
      read_u32(gct, GCT_STRING_OBJECT_ID_ABS_OFFSET))
    return 0xdead000bu;
  if (GCT_TEMPLATE_HOLDER_SIZE > sizeof(gHolder))
    return 0xdead000cu;
  if (GCT_TEMPLATE_STRING_SIZE > sizeof(gStringStorage))
    return 0xdead000du;

  zero_bytes((unsigned char *)&gHolder, sizeof(gHolder));
  zero_bytes(gStringStorage, sizeof(gStringStorage));
  copy_bytes((unsigned char *)&gHolder, gct + GCT_TEMPLATE_HOLDER_OFFSET,
             GCT_TEMPLATE_HOLDER_SIZE);
  copy_bytes(gStringStorage, gct + GCT_TEMPLATE_STRING_OFFSET,
             GCT_TEMPLATE_STRING_SIZE);
  gHolder.name = (const char *)gStringStorage;

  return (unsigned)gHolder.name[0];
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = materialize_from_emitted_gct();
}






