# Minimal LLVM-emitted metadata contract for reduced static capability globals

This note fixes the **minimal compiler/runtime contract** implied by the reduced
standalone repros in this directory.

It is intentionally narrow. The goal is **not** to solve every future static
capability case yet. The goal is to define the smallest emitted shape that is
sufficient for the currently proven reduced cases:

- one object with one **function-pointer** capability slot
- one object with one **string/object-pointer** capability slot

## Why this contract is the next step

The tree has already proven four increasingly strong facts:

1. raw file-scope static capability-typed loads fail
2. handwritten runtime materialization works
3. handwritten descriptor-driven materialization works
4. a prototype bridge from LLVM IR to generated descriptor-driven source works

The missing piece before an in-tree compiler-side POC is an **exact emitted data
contract** between:

- LLVM-side metadata emission, and
- runtime-side materialization.

## Proposed emitted section

Reuse one dedicated ELF metadata section for compiler-emitted static capability
object records.

Proposed name for the first POC:

- `.gct`

Reason:

- `capstone-c` already emits a `.gct` section for the analogous global-capability
  table concept,
- using the same section name for the first LLVM-path POC keeps the relationship
  visible,
- the reduced contract here is still smaller and more explicit than the existing
  `capstone-c` table.

If later bring-up shows that a separate section name is cleaner, the record shape
below still stands; only the section name would change.

## Exact record layout

See:

- `llvm_emitted_metadata_layout.h`

The section payload is:

1. `struct static_cap_metadata_section_header`
2. `struct static_cap_emitted_global_desc[object_count]`
3. `struct static_cap_emitted_slot_desc[slot_count]`
4. one trailing raw template-bytes blob

### Section header

The header provides:

- magic/version
- object count
- slot count
- total template-bytes size
- descriptor record sizes

That is enough for a runtime materializer to:

- sanity-check the section,
- walk the descriptor arrays,
- locate the raw template bytes.

### Global-object descriptor

Each `static_cap_emitted_global_desc` names one materialized object and gives:

- `object_id`
- final object `size`
- required `align`
- `template_offset` into the trailing template-bytes blob
- `first_slot_index`
- `num_slots`

The compiler-emitted template bytes must leave capability-bearing fields in a
safe raw state for later runtime patching.
For the reduced one-slot cases, that means those bytes are simply zero.

### Slot descriptor

Each `static_cap_emitted_slot_desc` describes one capability-bearing field inside
one emitted object.

Common fields:

- `object_id`
- `field_offset`
- `slot_kind`

Target interpretation depends on `slot_kind`.

#### Function slot

For `STATIC_CAP_SLOT_FUNCTION`:

- `target_flags` includes `STATIC_CAP_TARGET_FLAG_RELOCATED_SYMBOL`
- `target_ref` carries the relocated symbol reference
- `target_addend` is the symbol addend
- `target_object_id` is ignored

This means the compiler does **not** try to encode a live capability in metadata.
It only emits a normal symbol reference that the runtime-side materializer uses
as the source for constructing the live function capability.

#### Object / string slot

For `STATIC_CAP_SLOT_GLOBAL_OBJECT` and `STATIC_CAP_SLOT_STRING_OBJECT`:

- `target_object_id` names another emitted object descriptor
- `target_addend` is the byte offset inside that object
- `target_ref` is zero

This is enough for reduced intra-bundle references such as:

- holder object → backing string object

#### Null slot

For `STATIC_CAP_SLOT_NULL`:

- runtime zeroes the field
- target fields are ignored

## Minimal runtime obligations

A first eager-at-init runtime materializer using this contract only needs to:

1. locate the `.gct` section
2. read and validate the section header
3. allocate or choose writable storage for every described object
4. copy each object's raw template bytes into that storage
5. walk the slot descriptors for each object
6. patch the live capability-valued fields according to slot kind

This remains compatible with a later lazy-on-first-use policy.
The emitted contract does **not** force eager vs lazy; it only fixes the data
shape both policies would consume.

## Mapping to the current reduced cases

### Reduced function-pointer case

Source shape:

- `fail_fn_struct_load.c`
- one object `kHolder`
- one capability slot at offset `0`
- target is function `helper`

Emitted metadata shape:

- header:
  - `object_count = 1`
  - `slot_count = 1`
- global desc `OBJECT_HOLDER`:
  - `size = sizeof(holder)`
  - `align = 16`
  - `template_offset = 0`
  - `first_slot_index = 0`
  - `num_slots = 1`
- slot desc 0:
  - `object_id = OBJECT_HOLDER`
  - `field_offset = 0`
  - `slot_kind = STATIC_CAP_SLOT_FUNCTION`
  - `target_flags = STATIC_CAP_TARGET_FLAG_RELOCATED_SYMBOL`
  - `target_ref = reloc(helper)`
  - `target_addend = 0`
- template bytes:
  - one zeroed holder object image

### Reduced string-pointer case

Source shape:

- `fail_str_struct_load.c`
- one holder object `kHolder`
- one backing string object `"ok\0"`
- one capability slot at offset `0`

Emitted metadata shape:

- header:
  - `object_count = 2`
  - `slot_count = 1`
- global desc `OBJECT_HOLDER`
  - zeroed holder template
- global desc `OBJECT_STRING_OK`
  - raw bytes `{'o', 'k', '\0'}`
- slot desc 0:
  - `object_id = OBJECT_HOLDER`
  - `field_offset = 0`
  - `slot_kind = STATIC_CAP_SLOT_STRING_OBJECT`
  - `target_object_id = OBJECT_STRING_OK`
  - `target_addend = 0`

## What the first in-tree compiler-side POC should emit

The first honest compiler-side POC does **not** need to handle all globals.
It only needs to emit `.gct` records for the reduced one-slot cases above.

A good first success criterion is:

- compile the reduced function/string cases,
- emit the `.gct` section with the record layout above,
- keep the raw static object image non-live,
- let a runtime-side materializer rebuild the live slots from `.gct`.

## Non-goals of this first contract

This contract intentionally does **not** yet solve:

- a full live global capability table in `gp`
- lazy-on-first-use policy details
- loader-side eager provisioning
- arbitrary nested aggregate graphs
- deduplication / COMDAT / cross-TU merging questions
- final debug-info or profile-data integration details for a whole-program path

Those come later.
The point here is to freeze the smallest emitted shape that matches the already
proven reduced mechanism.

## Recommended immediate follow-up

After this contract, the next implementation step should be:

> emit a first `.gct` section from the LLVM path for the reduced one-slot cases,
> while keeping runtime-side materialization local and eager for the proof of
> mechanism.

