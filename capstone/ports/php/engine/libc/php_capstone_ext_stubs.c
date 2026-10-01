/* The three externs ext/standard/url.c needs that the engine does not provide.
 *
 * WHY STUBS AND NOT A PATCHED url.c: url.c is compiled BYTE-IDENTICAL from the corpus tree.
 * The functions under test (php_url_parse, php_replace_controlchars) are the real ones, and
 * the experiment is only meaningful if nothing in that file was edited. Two of these three
 * exist solely because of PHP_FUNCTION(get_headers) (url.c:592-645), which is never called
 * here -- the plan predicted it would be the only tie to the streams subsystem, and the
 * undefined-symbol list confirmed it exactly: _php_stream_open_wrapper_ex and
 * _php_stream_free and nothing else. Without -ffunction-sections the linker cannot drop
 * get_headers, so its references must still resolve.
 *
 * php_error_docref1 is different: it IS on the live path, at url.c:314, where zif_parse_url
 * reports a url it could not parse. It must therefore behave, not merely link -- it routes
 * to zend_error(E_WARNING) so the rung's error counter sees it.
 */
#include <zend.h>
#include <zend_API.h>

/* get_headers only -- unreachable here. Returning NULL/failure is the honest stub: if
 * something ever does call them the result is a clean failure, not silent nonsense. */
void *_php_stream_open_wrapper_ex(char *path, char *mode, int options,
                                  char **opened_path, void *context)
{
    (void)path; (void)mode; (void)options; (void)context;
    if (opened_path) { *opened_path = (char *)0; }
    return (void *)0;
}

int _php_stream_free(void *stream, int close_options)
{
    (void)stream; (void)close_options;
    return 0;
}

/* url.c:314, reached when php_url_parse returns NULL. Deliberately routed to zend_error so
 * a failed parse is COUNTED by the rung rather than silently swallowed -- otherwise a run
 * where parse_url merely failed would be indistinguishable from one where it succeeded. */
void php_error_docref1(const char *docref, const char *param1, int type,
                       const char *format, ...)
{
    (void)docref; (void)param1; (void)format;
    zend_error(type, "php_error_docref1");
}

/* Hook for beebs memcpy's BEEBS_MEMCPY_PHASE_NOTE one-shot assertion (default off there).
 * A mismatched-phase memcpy of >=16 bytes is the one shape in that function that can move
 * capability-bearing data and strip its tag, so this answers whether it ever happens on the
 * trigger path. php_fault_report halts and encodes its argument in badaddr, so n comes back
 * as badaddr - 64; the da/sa phases are folded in above it. */
void capstone_memcpy_phase_note(unsigned long da_, unsigned long sa_, unsigned long n_)
{
    extern void php_fault_report(unsigned long);
    php_fault_report((da_ << 56) | (sa_ << 48) | (n_ & 0xFFFFFFFFUL));
}
