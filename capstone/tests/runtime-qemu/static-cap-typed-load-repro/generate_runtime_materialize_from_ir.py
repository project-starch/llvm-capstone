#!/usr/bin/env python3

import argparse
import pathlib
import re
import sys


def decode_llvm_c_string(encoded: str) -> bytes:
    out = bytearray()
    i = 0
    while i < len(encoded):
        ch = encoded[i]
        if ch != '\\':
            out.append(ord(ch))
            i += 1
            continue

        if i + 2 < len(encoded) and all(c in '0123456789abcdefABCDEF' for c in encoded[i + 1 : i + 3]):
            out.append(int(encoded[i + 1 : i + 3], 16))
            i += 3
            continue

        if i + 1 >= len(encoded):
            raise ValueError('dangling backslash in LLVM c-string literal')

        nxt = encoded[i + 1]
        escapes = {
            '\\': ord('\\'),
            '"': ord('"'),
            'n': 0x0A,
            'r': 0x0D,
            't': 0x09,
            '0': 0x00,
        }
        if nxt not in escapes:
            raise ValueError(f'unsupported LLVM c-string escape: \\{nxt}')
        out.append(escapes[nxt])
        i += 2

    return bytes(out)


def parse_holder_global(ir: str):
    pattern = re.compile(
        r'^@(?P<symbol>[A-Za-z0-9_.$]+)\s*=\s*(?:internal|private)\s+addrspace\(200\)\s+constant\s+'
        r'(?P<type>%[^ ]+)\s+\{\s*ptr addrspace\(200\)\s+@(?P<target>[A-Za-z0-9_.$]+)\s*\},\s+align\s+(?P<align>\d+)',
        re.MULTILINE,
    )
    match = pattern.search(ir)
    if not match:
        raise ValueError('could not find reduced one-slot holder global in LLVM IR')
    return match.group('symbol'), match.group('type'), match.group('target'), int(match.group('align'))


def parse_function_return(ir: str, symbol: str) -> int:
    pattern = re.compile(
        r'^define\s+.*?@' + re.escape(symbol) + r'\([^\n]*\)\s+addrspace\(200\)\s*#\d+\s*\{(?P<body>.*?)^\}',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(ir)
    if not match:
        raise ValueError(f'could not find definition of function @{symbol} in LLVM IR')

    ret_match = re.search(r'\bret\s+i32\s+([0-9]+)\b', match.group('body'))
    if not ret_match:
        raise ValueError(f'could not find constant i32 return in function @{symbol}')
    return int(ret_match.group(1))


def parse_string_global(ir: str, symbol: str):
    pattern = re.compile(
        r'^@' + re.escape(symbol) +
        r'\s*=\s*(?:private|internal).*?addrspace\(200\)\s+constant\s+\[(?P<size>\d+)\s+x\s+i8\]\s+c"(?P<data>(?:[^"\\]|\\.)*)",\s+align\s+(?P<align>\d+)',
        re.MULTILINE,
    )
    match = pattern.search(ir)
    if not match:
        raise ValueError(f'could not find string global @{symbol} in LLVM IR')

    declared_size = int(match.group('size'))
    data = decode_llvm_c_string(match.group('data'))
    if len(data) != declared_size:
        raise ValueError(
            f'string global @{symbol} declared size {declared_size}, decoded size {len(data)}'
        )
    return data, int(match.group('align'))


def format_byte_list(data: bytes) -> str:
    return ', '.join(f'0x{byte:02x}' for byte in data)


def generate_function_case(ir_path: pathlib.Path, retval: int, holder_align: int) -> str:
    return f'''// Generated from LLVM IR: {ir_path}
// Reduced prototype bridge from compiler-emitted IR to runtime-side descriptor
// materialization for one function-capability slot.

#include <stdint.h>

enum static_cap_global_slot_kind {{
  STATIC_CAP_SLOT_NULL = 0,
  STATIC_CAP_SLOT_FUNCTION = 1,
  STATIC_CAP_SLOT_GLOBAL_OBJECT = 2,
  STATIC_CAP_SLOT_STRING_OBJECT = 3,
}};

struct static_cap_global_desc {{
  uint32_t object_id;
  uint32_t size;
  uint32_t align;
  uint32_t template_offset;
  uint32_t first_slot_index;
  uint32_t num_slots;
}};

struct static_cap_slot_desc {{
  uint32_t object_id;
  uint32_t field_offset;
  uint32_t slot_kind;
  uint32_t target_object_id;
  uint64_t target_addend;
}};

static inline void static_cap_copy_template_bytes(unsigned char *dst,
                                                  const unsigned char *src,
                                                  uint32_t size) {{
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = src[i];
}}

static inline void static_cap_zero_bytes(unsigned char *dst, uint32_t size) {{
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = 0;
}}

static inline void static_cap_store_function_slot(unsigned char *object_base,
                                                  uint32_t field_offset,
                                                  int (*fn)(void)) {{
  *(int (**)(void))(object_base + field_offset) = fn;
}}

struct generated_holder {{
  int (*slot0)(void);
}};

enum {{
  GENERATED_OBJECT_HOLDER = 0,
}};

static struct generated_holder gHolder;

static int generated_target_function(void) {{ return {retval}u; }}

static const unsigned char kGeneratedHolderTemplate[sizeof(struct generated_holder)] = {{0}};

static const struct static_cap_global_desc kGeneratedGlobalDescs[] = {{
    {{
        GENERATED_OBJECT_HOLDER,
        sizeof(struct generated_holder),
        {holder_align},
        0,
        0,
        1,
    }},
}};

static const struct static_cap_slot_desc kGeneratedSlotDescs[] = {{
    {{
        GENERATED_OBJECT_HOLDER,
        0,
        STATIC_CAP_SLOT_FUNCTION,
        0,
        0,
    }},
}};

static void materialize_generated_holder(void) {{
  const struct static_cap_global_desc *holder_desc =
      &kGeneratedGlobalDescs[GENERATED_OBJECT_HOLDER];
  unsigned i;

  static_cap_copy_template_bytes((unsigned char *)&gHolder, kGeneratedHolderTemplate,
                                 holder_desc->size);

  for (i = holder_desc->first_slot_index;
       i < holder_desc->first_slot_index + holder_desc->num_slots; ++i) {{
    const struct static_cap_slot_desc *slot = &kGeneratedSlotDescs[i];

    switch (slot->slot_kind) {{
    case STATIC_CAP_SLOT_FUNCTION:
      static_cap_store_function_slot((unsigned char *)&gHolder,
                                     slot->field_offset,
                                     generated_target_function);
      break;
    case STATIC_CAP_SLOT_STRING_OBJECT:
    case STATIC_CAP_SLOT_GLOBAL_OBJECT:
    case STATIC_CAP_SLOT_NULL:
    default:
      static_cap_zero_bytes((unsigned char *)&gHolder + slot->field_offset,
                            sizeof(void *));
      break;
    }}
  }}
}}

void domain_main(unsigned *res, unsigned func) {{
  (void)func;
  materialize_generated_holder();
  *res = (unsigned)gHolder.slot0();
}}
'''


def generate_string_case(ir_path: pathlib.Path, data: bytes, holder_align: int, string_align: int) -> str:
    byte_list = format_byte_list(data)
    return f'''// Generated from LLVM IR: {ir_path}
// Reduced prototype bridge from compiler-emitted IR to runtime-side descriptor
// materialization for one string/object capability slot.

#include <stdint.h>

enum static_cap_global_slot_kind {{
  STATIC_CAP_SLOT_NULL = 0,
  STATIC_CAP_SLOT_FUNCTION = 1,
  STATIC_CAP_SLOT_GLOBAL_OBJECT = 2,
  STATIC_CAP_SLOT_STRING_OBJECT = 3,
}};

struct static_cap_global_desc {{
  uint32_t object_id;
  uint32_t size;
  uint32_t align;
  uint32_t template_offset;
  uint32_t first_slot_index;
  uint32_t num_slots;
}};

struct static_cap_slot_desc {{
  uint32_t object_id;
  uint32_t field_offset;
  uint32_t slot_kind;
  uint32_t target_object_id;
  uint64_t target_addend;
}};

static inline void static_cap_copy_template_bytes(unsigned char *dst,
                                                  const unsigned char *src,
                                                  uint32_t size) {{
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = src[i];
}}

static inline void static_cap_zero_bytes(unsigned char *dst, uint32_t size) {{
  uint32_t i;
  for (i = 0; i < size; ++i)
    dst[i] = 0;
}}

static inline void static_cap_store_ptr_slot(unsigned char *object_base,
                                             uint32_t field_offset,
                                             const void *ptr) {{
  *(const void **)(object_base + field_offset) = ptr;
}}

struct generated_holder {{
  const char *slot0;
}};

enum {{
  GENERATED_OBJECT_HOLDER = 0,
  GENERATED_OBJECT_STRING = 1,
}};

static struct generated_holder gHolder;
static unsigned char gString[{len(data)}];

static const unsigned char kGeneratedHolderTemplate[sizeof(struct generated_holder)] = {{0}};
static const unsigned char kGeneratedStringTemplate[{len(data)}] = {{{byte_list}}};

static const struct static_cap_global_desc kGeneratedGlobalDescs[] = {{
    {{
        GENERATED_OBJECT_HOLDER,
        sizeof(struct generated_holder),
        {holder_align},
        0,
        0,
        1,
    }},
    {{
        GENERATED_OBJECT_STRING,
        {len(data)},
        {string_align},
        0,
        1,
        0,
    }},
}};

static const struct static_cap_slot_desc kGeneratedSlotDescs[] = {{
    {{
        GENERATED_OBJECT_HOLDER,
        0,
        STATIC_CAP_SLOT_STRING_OBJECT,
        GENERATED_OBJECT_STRING,
        0,
    }},
}};

static void materialize_generated_holder(void) {{
  const struct static_cap_global_desc *holder_desc =
      &kGeneratedGlobalDescs[GENERATED_OBJECT_HOLDER];
  const struct static_cap_global_desc *string_desc =
      &kGeneratedGlobalDescs[GENERATED_OBJECT_STRING];
  unsigned i;

  static_cap_copy_template_bytes((unsigned char *)&gHolder, kGeneratedHolderTemplate,
                                 holder_desc->size);
  static_cap_copy_template_bytes((unsigned char *)gString, kGeneratedStringTemplate,
                                 string_desc->size);

  for (i = holder_desc->first_slot_index;
       i < holder_desc->first_slot_index + holder_desc->num_slots; ++i) {{
    const struct static_cap_slot_desc *slot = &kGeneratedSlotDescs[i];

    switch (slot->slot_kind) {{
    case STATIC_CAP_SLOT_STRING_OBJECT:
      static_cap_store_ptr_slot((unsigned char *)&gHolder,
                                slot->field_offset,
                                (const void *)(gString + slot->target_addend));
      break;
    case STATIC_CAP_SLOT_FUNCTION:
    case STATIC_CAP_SLOT_GLOBAL_OBJECT:
    case STATIC_CAP_SLOT_NULL:
    default:
      static_cap_zero_bytes((unsigned char *)&gHolder + slot->field_offset,
                            sizeof(void *));
      break;
    }}
  }}
}}

void domain_main(unsigned *res, unsigned func) {{
  (void)func;
  materialize_generated_holder();
  *res = (unsigned)gHolder.slot0[0];
}}
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate a reduced descriptor-driven runtime-materialization domain from Capstone LLVM IR.'
    )
    parser.add_argument('--ir', required=True, help='Path to the LLVM IR (.ll) input')
    parser.add_argument('--output-c', required=True, help='Path to write the generated C domain source')
    args = parser.parse_args()

    ir_path = pathlib.Path(args.ir)
    output_path = pathlib.Path(args.output_c)
    ir = ir_path.read_text(encoding='utf-8')

    _, _, target_symbol, holder_align = parse_holder_global(ir)

    if re.search(r'^define\s+.*?@' + re.escape(target_symbol) + r'\(', ir, re.MULTILINE):
        retval = parse_function_return(ir, target_symbol)
        generated = generate_function_case(ir_path, retval, holder_align)
    else:
        string_data, string_align = parse_string_global(ir, target_symbol)
        generated = generate_string_case(ir_path, string_data, holder_align, string_align)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())

