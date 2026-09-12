/* MPY-T29 / upstream 4705, as measured. Extracted verbatim from
   ../../../port/mpy_domain.c, where it lives behind MPY_T29_HIDDEN_ROOT so the
   production glue is byte-identical when the flag is off. Kept here so the case
   directory is self-contained; port/mpy_domain.c is the file that is built.

   Build: MPY_T29_HIDDEN_ROOT=1 in DOMAIN_EXTRA_DEFS, then run the .dom under QEMU.
   Measured retval: 0x29007701. See RESULT.txt.

   The rest of the glue -- the 384 KiB static heap, gc_collect, mp_cstack_init --
   is in mpy_domain.c and is what makes this a NESTED allocator: one capability
   over the whole heap, so gc_free never reaches the hardware. */
#ifdef MPY_T29_HIDDEN_ROOT
/* MPY-T29 / upstream issue 4705, reproduced in THIS glue. 4705's fix is
   "unix/gccollect: Make sure stack/regs get captured properly for GC": roots the
   collector fails to see let it free reachable objects. That fix is port-specific,
   so there is no upstream program to run here, but our gc_collect (above) has the
   same shape of gap -- it scans ONLY the C stack.

   Hiding the pointer by XOR would be WRONG on this target: a capability turned
   into an integer and back is untagged, so the test would measure tag integrity
   and fault with cause 24 for a reason that has nothing to do with lifetime.
   Instead the capability is parked in a GLOBAL, which stays a valid capability and
   is not a scanned root, so the block becomes unreachable to the collector while
   remaining perfectly usable to us.

   The ambiguity this design has to resolve: "freed but unchanged" and "never freed
   at all" both read back 0x33. So after the collect a FRESH block is allocated and
   stamped 0x77. If the stale pointer then reads 0x77, the storage was freed AND
   handed to a live object, which is premature-free demonstrated rather than
   assumed. A RECONSTRUCTION, and the result file says so. */
static byte *mpy_t29_stash;      /* global: valid capability, not a scanned root */

static void mpy_t29_hidden_root(unsigned *res) {
    mpy_t29_stash = m_new(byte, 64);
    mpy_t29_stash[0] = 0x33;

    gc_collect();                /* nothing on the stack refers to the block */

    byte *fresh = m_new(byte, 64);
    fresh[0] = 0x77;

    byte via_stale = mpy_t29_stash[0];   /* USE AFTER (premature) FREE */
    mpy_t29_stash[0] = 0xCC;             /* and a write through it */

    /* low nibble 1 = read back 0x77, so the block was freed and reused and the
       stale pointer now aliases a LIVE object: premature free demonstrated.
       2 = read back 0x33, so it was not reused, and the test proves only that the
       access was not trapped.
       3 = something else entirely. */
    unsigned tag = via_stale == 0x77 ? 1u : (via_stale == 0x33 ? 2u : 3u);
    *res = 0x29000000u | ((unsigned)via_stale << 8) | tag;
}
#endif
#ifdef MPY_T29_HIDDEN_ROOT
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());

/* --- the dispatch arm in mpy_domain_entry, same file --- */
#ifdef MPY_T29_HIDDEN_ROOT
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    mpy_t29_hidden_root(res);
    return;
#endif

/* --- the collector this exploits, mpy_domain.c:102, stack only, no registers ---
   void gc_collect(void) {
       gc_collect_start();
       gc_collect_root(sp_now, ((size_t)(top - (char *)sp_now)) / sizeof(void *));
       gc_collect_end();
   }
   4705 fixed exactly this gap in ports/unix/gccollect.c. */
