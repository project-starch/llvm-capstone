# Minimal metadata contract for the reduced static capability globals case

This note records the smallest metadata shape currently needed to describe the
reduced failing case in `static_const_domain.c`.

## Purpose

Both of the currently discussed implementation strategies:

- eager runtime-side materialization during domain init, and
- lazy runtime-side materialization on first use,

need the same basic information:

1. which runtime global objects exist,
2. where their raw template bytes come from,
3. which fields inside them are capability-valued,
4. what each capability-valued field should ultimately point to.

## Reduced case

Source shape:

```c
struct pair {
  int (*fn)(void);
  const char *name;
};

static int helper(void);
static const struct pair kPair = { helper, "ok" };
```

## Minimal object model

### Object 0: `kPair`

- size: 32 bytes
- align: 16 bytes
- raw template bytes: non-capability fields copied directly
- capability-valued slots:
  - offset `0x00`: function capability to `helper`
  - offset `0x10`: string/global-byte-object capability to object 1

### Object 1: `"ok"`

- size: 3 bytes
- align: 1 byte
- raw template bytes: `{'o', 'k', '\0'}`
- no capability-valued slots

## Implication

The loaded image does not need to contain already-usable capabilities for these
fields.

Instead, runtime-side logic only needs:

- the object descriptors,
- the slot descriptors,
- access to template bytes,
- and enough runtime capability context to build the final function/global
  capabilities.

That is exactly the common core needed by both eager and lazy approaches.

## Current evidence in this tree

The sibling probe `runtime_materialize_domain.c` already shows that the reduced
`pair { fn, name }` shape works when those capability-valued fields are created
at runtime in writable global storage rather than consumed directly from a
file-scope `static const` image object.


