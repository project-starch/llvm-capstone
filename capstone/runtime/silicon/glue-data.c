/* B0 (docs/plans/b0-silicon-delegated-runtime.md): data that ports/musl-capstone/runtime/start-musl.S defines in
 * ASSEMBLY, defined here in C instead.
 *
 * Under gp-captable with full LTO an assembly definition is outside the LTO module, so a C `extern` reaches it as
 * an undefined declaration -- derived from gp, then delin'd, which faults on silicon (C-13). Defined in C, each is
 * an ordinary global with a cap-table slot.
 *
 * musl's cancellation handler (src/thread/pthread_cancel.c) tests `pc >= __cp_begin && pc < __cp_end`, so the
 * three markers must lie in that ORDER. Separate symbols, not one array with aliases: a variable alias is the
 * shape gp-captable does not support (the C-75 residual). The section names sort, and link-gpfree-app.ld emits
 * them SORTed, which fixes the order. They are addresses to compare and never execute (runtime signals.c). */
__attribute__((section(".rodata.capstone_cancel_points.0"), used)) const char __cp_begin[1] = {0};
__attribute__((section(".rodata.capstone_cancel_points.1"), used)) const char __cp_end[1] = {0};
__attribute__((section(".rodata.capstone_cancel_points.2"), used)) const char __cp_cancel[1] = {0};
