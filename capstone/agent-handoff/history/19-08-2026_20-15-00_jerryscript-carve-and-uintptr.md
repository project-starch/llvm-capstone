# JerryScript bring-up: two defects, one of them architectural

Found 2026-08-19 while getting the JerryScript domain past its first capability
fault. Recorded here because neither is JerryScript-specific and the second
outlives this port.

## 1. Globals outside the amalgam TU have no storage at all (port defect, FIXED)

`-capstone-gp-captable` emits its carve descriptor PER TRANSLATION UNIT and only one
survives the link. The JerryScript build applied that to jerry-core and stopped
there, so `jerryscript-math.c`, `jerry_domain.c` and `capstone_libc_extra.c` stayed
separate units. Their **11 globals, 1,784 bytes** — the output buffer and the setjmp
buffer among them — appeared in no surviving descriptor.

Under this ABI `.bss` is NOLOAD (link-gpfree.ld says so deliberately: every global's
storage is carved from dom_data at entry). A global that is neither loaded nor carved
therefore has no backing anywhere.

Measured, not argued:

| | descriptor entries | of size 224 (jmp_buf) | of size 1024 (jd_out) |
|---|---|---|---|
| before | 236 | 0 | 0 |
| after  | 255 | 1 | 1 |

It hid because nothing touched an unbacked global early enough; the domain always
died elsewhere first. Arming a setjmp in `domain_main` made one the very first
access, and it faulted with cause 7 on 208-byte bounds at the first instruction of
`setjmp` — a jmp_buf is 224.

**MicroPython does not have this**: exactly one object owns globals (514 in mpy.o).
Its measurements are unaffected, and it is what shows the invariant is right.

FIX: everything that owns a global goes in one TU. A GATE in the build script now
enforces it per .o and was negative-tested against the previous object directory,
where it fires and names all four offenders.

## 2. uintptr_t cannot hold a pointer on this target (UNFIXED, architectural)

    __SIZEOF_POINTER__ = 16
    __UINTPTR_TYPE__   = long unsigned int      (8 bytes)

The target declares 16-byte pointers and an 8-byte `uintptr_t`. That is
self-contradictory: the language requires `uintptr_t` to round-trip any `void *`.
Every `(uintptr_t) ptr` truncates the capability to its cursor and drops the tag, and
every `(T *) uint` fabricates an untagged pointer.

JerryScript does this in **304 places**, across three shapes (pointer-to-int,
int-to-pointer, and the compressed-pointer helpers). Nobody had seen it because the
build passes `-w`; with `-Wall` the single TU emits 304 warnings and nothing else.

Not every site is fatal. The common pattern is `(uintptr_t)p - (uintptr_t)base` for an
offset and `base + offset` to rebuild — the pointer is reconstructed from a LIVE base
capability, so the tag comes from the base and the truncation is harmless. The
dangerous shape is `(T *) integer` with no base, which yields an untagged pointer that
faults on first use.

Fixing it properly means `uintptr_t` becomes the 128-bit capability carrier — which is
exactly C-23's knot: `i128` IS the capability type and integer i128 arithmetic is a
low-64-bit approximation. So this is the same architectural item seen from the other
end, and it is not a small change.

WHAT TO DO MEANWHILE: build ports with `-Wall` rather than `-w`, and read
pointer-to-int/int-to-pointer warnings as a list of places a tag can be lost. Silencing
them is how this stayed invisible.

## Method note

Three "wedges" during this session turned out not to be. The staged probe prints every
10 GUEST seconds, and guest time under QEMU runs far slower than the wall clock, so a
run with no output for 200 wall-clock seconds may still be loading. Loading a 4 MB
domain over 9p takes minutes. Check `wchan` before calling anything wedged; the
transport was separately proven innocent by reading the same file with `wc -c`, which
returned all 4,137,640 bytes and a correct md5.
