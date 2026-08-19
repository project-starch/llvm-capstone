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

## 3. VERDICT (2026-08-20): JerryScript is blocked on the toolchain, not on bugs

With the carve fixed and the compiler fix in, the ladder bisects cleanly from the
rootfs (domains load in under a second there instead of the 3+ minutes 9p costs):

| stage | does | result |
|---|---|---|
| 0 | nothing | returns 0x1E000006, the exact expected marker |
| 1 | jerry_init + jerry_cleanup | cause 24 in ecma_gc_run, via jerry_heap_gc |

Stage 0 confirms at RUNTIME that both earlier fixes hold: the domain is created,
entered, runs and returns the right value.

Stage 1 faults in `jmem_decompress_pointer`:

    const uintptr_t heap_start = (uintptr_t) &JERRY_HEAP_CONTEXT (first);
    uint_ptr <<= JMEM_ALIGNMENT_LOG;
    uint_ptr += heap_start;
    return (void *) uint_ptr;

The fault is the missing TAG, not a wrong address -- "Cap mem access requires
capability" tests rs1->tag regardless of value.

**This is not one function.** 93 distinct source lines across 60 functions rebuild a
pointer from an integer. JerryScript's object model stores references as compressed
offsets and reconstructs addresses arithmetically throughout.

**It cannot be fixed in the target.** clang's `TargetInfo::IntType` enum has no
128-bit member -- the widest is UnsignedLongLong -- so `uintptr_t` cannot be made
capability-wide. The Capstone target already declares AS0 pointers as 64-bit for
exactly this reason: InitPreprocessor.cpp asserts uintptr_t width == pointer width,
and the datalayout comment calls it a "Workaround for Clang consistency check".

**The control that makes this a conclusion rather than a guess:** MicroPython runs.
Not through luck or effort -- its object word IS a pointer (MICROPY_OBJ_REPR_A), so
tagging is pointer arithmetic and the capability survives. It had exactly ONE
uintptr_t site, and the port fixed it with a config knob
(MICROPY_STREAM_IOCTL_ARG_TYPE = void *). The difference is the object model.

### The three options, honestly

1. **Give the toolchain a capability-carrying integer type** (CHERI's `__intcap`).
   The real fix, fixes all 93 sites with no source change, benefits every future
   port -- and it is the same knot as C-23, where i128 is the capability carrier and
   integer i128 is a low-64-bit approximation. Substantial: it needs clang's IntType
   enum extended.
2. **Patch JerryScript's 93 sites** to derive from a live base pointer. Mechanical
   but large, and it modifies the allocator that is the object of the measurement.
3. **Choose a second allocator with a pointer-based object model.** mruby qualifies
   (`boxing_no.h` -- a struct with a union of real pointers) and is already ported:
   38 commits on musl-capstone-port, eight to nine pinned versions, Rows 12 and 14
   already measured.

For the benchmark goal -- a second allocator with 25-30 temporal CVEs -- (3) is the
route that works today. (1) belongs on its own track.

## Method note

Three "wedges" during this session turned out not to be. The staged probe prints every
10 GUEST seconds, and guest time under QEMU runs far slower than the wall clock, so a
run with no output for 200 wall-clock seconds may still be loading. Loading a 4 MB
domain over 9p takes minutes. Check `wchan` before calling anything wedged; the
transport was separately proven innocent by reading the same file with `wc -c`, which
returned all 4,137,640 bytes and a correct md5.
