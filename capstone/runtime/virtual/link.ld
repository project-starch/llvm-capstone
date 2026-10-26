/* The virtual profile's image: two load segments, mapped by the launcher
 * straight from the sealed image file instead of being copied.
 *
 *   text  R+X  .text .rodata, and the tables that are only READ at run time:
 *              .capstone_application, the init/fini arrays (link addresses as
 *              plain integers), .capstone_cap_init (PC-relative offsets), .gct
 *   data  R+W  .data .tdata .bss: everything written at run time, including
 *              every global a capability initializer stores into (the compiler
 *              places those holders in .data even when they are `const`)
 *
 * The text segment is mapped from the file without write permission, so no
 * store -- scalar or capability -- can reach its pages: they never hold a tag,
 * need no pin and no sweep, and are shared, demand-faulted page-cache pages.
 * The data segment starts on the page right after the text segment
 * (p_vaddr(data) == p_vaddr(text) + p_memsz(text)), and the text segment ends
 * on a page boundary with zeros IN THE FILE, so no byte that is not part of
 * the image is ever mapped. The launcher checks both (exec.c, image_layout).
 *
 * The physical profile keeps my_first_domain/link.ld (one RWX segment). */
ENTRY(_start)

PHDRS {
  text PT_LOAD FLAGS(0x5);
  data PT_LOAD FLAGS(0x6);
  tls PT_TLS;
}

SECTIONS {
  . = 0x10000;

  .text : ALIGN(16) {
    *(.text.entry)
    *(.text .text.*)
  } :text

  .rodata : ALIGN(16) {
    *(.srodata .srodata.*)
    *(.rodata .rodata.*)
  } :text

  .capstone_application : ALIGN(8) {
    KEEP(*(.capstone_application))
  } :text

  /* THE INIT AND FINI ARRAYS, defined so their MARKERS EXIST.
   *
   * musl's exit path walks the fini array:
   *
   *     extern weak hidden void (*const __fini_array_start)(void), (*const
   *                                  __fini_array_end)(void);
   *     uintptr_t a = (uintptr_t)&__fini_array_end;
   *     for (; a > (uintptr_t)&__fini_array_start; a -= sizeof(void(*)()))
   *
   * Undefined weak symbols resolve to ADDRESS ZERO, so the loop correctly never
   * runs -- but forming the address at all does not come free on this target.
   * Address zero derived against the image capability is 0x10000 BELOW its base
   * (the image is linked at 0x10000), and clang's subobject bounds then shrink
   * that to the 16 bytes of a function pointer. The result is a SHRINK whose
   * requested range lies outside the capability it shrinks, which the spec makes
   * an `Illegal operand value (29)`:
   *
   *     SHRINK illegal operand value: requested [1023f0000, 1023f0010),
   *                                   current  [102400000, ...)
   *
   * Reached by mruby's gem test suite, where an unimplemented symlink() raises a
   * Ruby exception and the driver calls exit(). Any domain that calls exit() has
   * the same hole.
   *
   * Defining the sections gives both markers real, EQUAL addresses inside the
   * image, so the comparison is false without any capability being formed out of
   * bounds. They were empty when this was written -- verified across all 150
   * objects of the then-largest domain and all 1346 archive members, none had an
   * .init_array, .fini_array, .ctors or .dtors -- so this added symbols and no
   * bytes. Most domains still have none; see the note below for the first that did.
   *
   * KEEP, and placed HERE rather than left as orphans: an orphan section is
   * placed by the linker wherever it likes, and in this script that could be
   * the wrong segment or after .bss.
   *
   * THE DAY ONE APPEARED (2026-09-24, the tshark port: GLib's and libgpg-error's
   * constructors, libxml2's destructor). The musl runtime now runs both arrays
   * (ports/musl-capstone/runtime/hostcall.c, ISSUES C-64): the slots hold LINK
   * addresses as plain integers, since nothing relocates them, and it turns each
   * into a callable capability. Priority sections (`constructor(N)`) are placed
   * here too, sorted and ahead of the plain ones as the standard scripts do; left
   * out, they would be orphans outside the markers that nothing runs. */
  .init_array : ALIGN(16) {
    __init_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP(*(.init_array))
    __init_array_end = .;
  } :text

  .fini_array : ALIGN(16) {
    __fini_array_start = .;
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP(*(.fini_array))
    __fini_array_end = .;
  } :text

  /* Capability-global initializer table (one 8-byte PC-relative entry per
     module with capability globals): each entry = (initfn - entry). start.S
     iterates [start,end) before domain_main and calls each initializer at
     entry_runtime + offset. Position-independent (the domain processes no
     load-time relocations). Empty (start == end) for domains with none. */
  .capstone_cap_init : ALIGN(8) {
    __capstone_cap_init_start = .;
    KEEP(*(.capstone_cap_init))
    __capstone_cap_init_end = .;
  } :text

  /* Scalar descriptors of the capability globals; nothing writes them. */
  .gct : ALIGN(16) {
    KEEP(*(.gct))
  } :text

  /* The text segment's last page, completed with zeros that are part of the
   * file: BYTE(0) makes the section non-empty, so the padding is loaded data
   * (p_filesz == p_memsz) and not a gap the file could fill with other bytes.
   * The padding reaches the alignment .data asks for, when that exceeds a
   * page (an initialized global with a larger alignment), or the data segment
   * would start past the text segment's end. */
  .text_end : {
    BYTE(0)
    . = ALIGN(MAX(4096, ALIGNOF(.data)));
  } :text
  /* The text segment's layout. .gct is allocated as long as one of its inputs
   * is (lld ORs the flags); the runtime's own start and end markers are, like
   * the compiler's records. With no allocated input the output .gct would be
   * non-allocated at address 0, .text_end (only data commands) would follow it,
   * and the text segment's size would come out negative. The launcher checks
   * the same layout again when it loads the image. */
  ASSERT(ADDR(.text_end) > ADDR(.text),
         "text segment out of order: no allocated .gct input?")
  ASSERT((ADDR(.text_end) + SIZEOF(.text_end)) % 4096 == 0,
         "text segment does not end on a page")
  ASSERT(ADDR(.data) == ADDR(.text_end) + SIZEOF(.text_end),
         "data segment does not start where the text segment ends")

  .data : ALIGN(4096) {
    *(.sdata .sdata.*)
    *(.data .data.*)
  } :data

  /* THREAD-LOCAL STORAGE (C-47): the TEMPLATE, not a thread's storage.
   *
   * .tdata holds the initial values of the __thread variables and .tbss their
   * zero-initialised size; together they are the PT_TLS segment. The compiler
   * reaches a thread-local as tp + %tprel(sym), and lld computes %tprel against
   * this segment (RISC-V variant I: the block starts AT tp). The runtime
   * (musl-capstone runtime/tls.c) copies .tdata into a block it allocates, zeroes
   * the rest, and puts tp there -- the template itself is never written.
   *
   * The three symbols are how it finds the template. They are read as ADDRESSES
   * in the image and only ever subtracted from one another: an absolute symbol
   * (a size or an alignment given as a symbol's value) would come out wrong,
   * because the domain runs at a different base than it is linked at and forms
   * every address relative to itself.
   *
   * WHY .tdata IS ALIGNED TO .tbss's ALIGNMENT. lld's offset for a symbol is
   * (its address - p_vaddr) + (p_vaddr mod p_align), and p_align is the larger
   * of the two sections' alignments. Aligning .tdata to both makes the second
   * term zero, so the template starts at a p_align boundary and the runtime
   * only has to put tp on one: it describes the segment to musl with a 4096-byte
   * alignment and offset 0, and musl's __copy_tls aligns every tp to a page
   * (runtime/tls.c), which covers every alignment the ASSERT below admits.
   *
   * With no thread-locals both sections are empty, nothing moves and nothing is
   * added to the loaded image; the PT_TLS header is empty and the loader only
   * reads PT_LOAD headers. */
  .tdata : ALIGN(MAX(16, ALIGNOF(.tbss))) {
    __capstone_tls_image = .;
    *(.tdata .tdata.*)
    __capstone_tdata_end = .;
  } :data :tls

  .tbss : ALIGN(16) {
    *(.tbss .tbss.*)
    *(.tcommon)
    __capstone_tls_end = .;
  } :data :tls

  ASSERT(ALIGNOF(.tdata) <= 4096 && ALIGNOF(.tbss) <= 4096,
         "a thread-local aligned above 4096 bytes: runtime/tls.c aligns tp to a page")

  /* .bss last in the data segment: NOBITS, so it adds to p_memsz only. */
  .bss (NOLOAD) : ALIGN(16) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(16);
  } :data

  .capstone_virtual 0 : { KEEP(*(.capstone_virtual)) }

  /DISCARD/ : {
    *(.comment)
    *(.note*)
    *(.eh_frame*)
    *(.llvm_addrsig)
    *(.riscv.attributes)
  }
}
