/* Corpus reproductions for the MicroPython bug corpora, lifted out of the port's
   mpy_domain.c so that the port carries no case material. Included from there under
   MPY_CORPUS_GLUE; the include path is supplied by the corpus driver, so a port built
   without a corpus define never looks for this file.

   The reproductions themselves: MPY-T07, MPY-T16, MPY-T29 and the spatial counterpart. */

#ifdef MPY_T07_LEXER_UAF
/* MPY-T07 / upstream issue 4128, reproduced in THIS glue rather than in the
   upstream program. The defect is in caller code, not the interpreter: mp_parse()
   consumes and frees the lexer, and 4128's hello-embed.c then passed
   lex->source_name to mp_compile(). The fix hoisted the qstr into a local.
   mp_parse still frees the lexer at our pin, so the same misuse is still available
   and needs no parent build.
   This is a RECONSTRUCTION. Same API misuse, same interpreter, but not the
   upstream binary, and the result file must say so.
   Kept behind its own #ifdef in its own function so the production path is
   byte-identical when the flag is off. */
static void mpy_t07_lexer_uaf(unsigned *res) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, "1+1", 3, 0);
        qstr before = lex->source_name;                          /* while alive */
        mp_parse_tree_t pt = mp_parse(lex, MP_PARSE_FILE_INPUT); /* frees lex */
        qstr after = lex->source_name;                           /* USE AFTER FREE */
        (void)pt;
        nlr_pop();
        /* 0x7...: 1 = the freed block still reads back its old value (untrapped,
           but staleness NOT demonstrated); 2 = it reads back something else, so
           the read demonstrably came from recycled storage. Either way, reaching
           this line at all means the hardware did not object. */
        *res = 0x70000000u | (before == after ? 1u : 2u);
        return;
    }
    *res = 0x7000EEEEu;   /* the interpreter raised; not a capability fault */
}
#endif


#ifdef MPY_T16_DEINIT_AFTER_SWEEP
/* MPY-T16 / upstream issue 5487, reproduced in THIS glue. The reporter's words:
   "when MICROPY_PORT_DEINIT_FUNC is called it's too late, gc ram was already
   deallocated", because the ESP32 port runs gc_sweep_all() before mp_deinit().
   Our domain owns its own teardown, so this reconstructs the ordering rather than
   running the ESP32 program: hold a raw C pointer to a GC block, sweep, then use
   it, which is exactly what a port deinit hook holding hardware state does.
   A RECONSTRUCTION, and the result file says so. */
static void mpy_t16_deinit_after_sweep(unsigned *res) {
    byte *p = m_new(byte, 64);
    p[0] = 0x5A;
    gc_sweep_all();            /* frees everything, including p's block */
    byte after = p[0];         /* USE AFTER FREE: the deinit hook's read */
    p[0] = 0xA5;               /* and its write */
    /* low nibble 1 = the block still reads back 0x5A, so freed-but-unchanged;
       2 = it reads back something else, so the storage was already reused.
       Reaching this line at all means the hardware did not object. */
    *res = 0x16000000u | ((unsigned)after << 8) | (after == 0x5A ? 1u : 2u);
}
#endif


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

#ifdef MPY_SPATIAL_OVERFLOW
/* THE SPATIAL COUNTERPART TO MPY-T29, on the real allocator rather than a model.
   Not a corpus row: the corpus is 30 temporal defects and one spatial one, and
   this is a direct measurement of the mechanism underneath both.

   No lifetime component anywhere. Two GC blocks, both live for the whole run,
   nothing freed and no collection. The write simply walks off the end of the
   first block into the second. If the hardware saw one object per gc_alloc it
   would trap; if bounds are inherited from the one heap array, it cannot.

   The distance is measured at RUN TIME rather than assumed. MICROPY_BYTES_PER_GC_BLOCK
   is 32, so two 64-byte allocations should be 64 bytes apart, but a hardcoded 64
   would silently turn into an in-bounds write to a's own tail if that ever changed,
   and the run would look identical. Reported in the retval so the arm cannot pass
   while measuring the wrong thing.

   uintptr_t and not pointer subtraction: `b - a` lowers to lcc on this target,
   which is wrong on anything untagged, and the port already patches py/gc.c for
   exactly this reason. The values here are tagged, but the idiom should match.

   MPY_SPATIAL_OVERFLOW selects the arm, and it is the ONLY difference between them:
     1 -> off = dist, one past a's end and into b, still inside the heap: expect UNTRAPPED
     2 -> off = past the heap array itself:                               expect TRAPPED
   Arm 2 is the positive control. Without it, a quiet arm 1 and a domain that never
   started look the same. */
static void mpy_spatial_overflow(unsigned *res) {
    byte *a = m_new(byte, 64);
    byte *b = m_new(byte, 64);

    a[0] = 0x11;
    b[0] = 0x22;

    size_t dist = (size_t)((uintptr_t)b - (uintptr_t)a);

#if MPY_SPATIAL_OVERFLOW == 2
    volatile size_t off = sizeof(mpy_heap) + 64u;   /* past the whole heap: must fault */
#else
    volatile size_t off = dist;                     /* out of a, into b */
#endif
    a[off] = 0xAA;                                  /* THE OVERFLOW */

    /* low nibble 1 = b was overwritten through a, so one sub-allocated block wrote
       into another and nothing objected.
       2 = b intact, the write landed elsewhere and the arm measured nothing.
       3 = something else. The middle 16 bits carry dist, so a wrong block layout
       is visible in the result instead of hiding behind a green rung. */
    unsigned tag = b[0] == 0xAA ? 1u : (b[0] == 0x22 ? 2u : 3u);
    *res = 0x5A000000u | ((unsigned)(dist & 0xFFFFu) << 8) | tag;
}
#endif

#if MICROPY_VFS
/* FatFs asks the port for a timestamp on every file it creates (ff.c:257,
   GET_FATTIME). This domain has no clock -- there is deliberately no time module,
   because a stubbed one would make time.time() confidently wrong -- so this returns
   a FIXED date rather than an invented one.
   2026-01-01 00:00:00 in FatFs's packed format: bits 31..25 year-1980, 24..21 month,
   20..16 day, 15..11 hour, 10..5 minute, 4..0 second/2.
   Every file in a domain image therefore carries the same mtime. That is visible and
   constant, which is the point: a wrong-but-plausible clock would not be. */
DWORD get_fattime(void) {
    return ((DWORD)(2026 - 1980) << 25) | ((DWORD)1 << 21) | ((DWORD)1 << 16);
}
#endif

