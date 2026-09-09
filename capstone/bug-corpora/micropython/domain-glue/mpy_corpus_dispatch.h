/* Corpus reproductions for the MicroPython bug corpora, lifted out of the port's
   mpy_domain.c so that the port carries no case material. Included from there under
   MPY_CORPUS_GLUE; the include path is supplied by the corpus driver, so a port built
   without a corpus define never looks for this file.

   The domain_main dispatch for them. Included INSIDE domain_main, which is why it has
   no include guard and is not a standalone header. */

#ifdef MPY_T07_LEXER_UAF
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    mpy_t07_lexer_uaf(res);
    return;
#endif
#ifdef MPY_T16_DEINIT_AFTER_SWEEP
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    mpy_t16_deinit_after_sweep(res);
    return;
#endif
#ifdef MPY_T29_HIDDEN_ROOT
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    mpy_t29_hidden_root(res);
    return;
#endif
#ifdef MPY_SPATIAL_OVERFLOW
    (void)func;
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    mpy_spatial_overflow(res);
    return;
#endif
