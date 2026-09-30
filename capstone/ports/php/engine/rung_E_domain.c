/* Phase 3: run a REAL corpus trigger.
 *
 *   CRASH-110   var_dump(parse_url("file:///"))   ASAN: READ of size 1, url.c:132
 *   CRASH-073   var_dump(parse_url('a:/'))        ASAN: READ of size 3, _estrndup from url.c:292
 *
 * var_dump is NOT needed and is not linked: both over-reads happen INSIDE php_url_parse,
 * before anything formats the result. That keeps ext/standard/var.c out of the image.
 *
 * WHAT IS REAL HERE. ext/standard/url.c is compiled byte-identical from the corpus tree, so
 * php_url_parse and zif_parse_url are PHP's own code, reached the way PHP reaches them:
 * source text -> scanner -> parser -> compiler -> VM -> ZEND_DO_FCALL -> zif_parse_url ->
 * zend_parse_parameters -> php_url_parse. The only thing written here is the one-entry
 * zend_function_entry table, because there is no ext/standard module to register.
 *
 * CRASH-110's mechanism, for the record. "file:///" is 8 chars, so the zval's string is
 * estrndup'd to 9 bytes (indices 0..8). e = strchr(s,':') = &str[4]. url.c:132 is
 *
 *     if (*(e + 5) == ':') {
 *
 * which reads str[9] -- exactly one past the allocation. On stock PHP the allocator's
 * REAL_SIZE rounding puts slack there and the read lands in it; that is the corpus's
 * "crashes_on_pristine_build: false" and why the case needs ASAN.
 *
 * MATCHED PAIR, as run-crash008.sh does it: the CONTROL arm is built with
 * -DZEND_CAP_BOUNDS_REAL_SIZE so every capability is bounded to the ROUNDED size, and must
 * COMPLETE (that is stock PHP). The FAULT arm bounds to the true request and should halt.
 * A fault with no passing control proves nothing.
 *
 * Reported through *res:
 *   bits 0..9    stage bits
 *   bits 10..13  returned zval type (IS_ARRAY 4 on a successful parse)
 *   bits 16..23  number of zend_error calls, saturated
 */
#include <zend.h>
#include <zend_API.h>
#include <zend_compile.h>
#include <zend_execute.h>

#define ST_ENTERED    0x001u
#define ST_SINKED     0x002u
#define ST_STARTED    0x004u
#define ST_COMPILER   0x008u
#define ST_EXECUTOR   0x010u
#define ST_REGISTERED 0x020u  /* parse_url is in CG(function_table) */
#define ST_EVALED     0x040u  /* zend_eval_string returned SUCCESS */
#define ST_ISARRAY    0x080u  /* the trigger returned an array -- parse_url really ran */
#define ST_BAILED     0x100u  /* zend_catch taken */
#define ST_ERRORS     0x200u  /* zend_error seen */

/* CRASH-110 by default; -DRUNG_E_CRASH073 selects the other. No whitespace, and the inner
 * quotes are escaped because the whole thing is one -D argument. */
#ifndef RUNG_E_SOURCE
# ifdef RUNG_E_CRASH073
#  define RUNG_E_SOURCE "parse_url(\"a:/\")"
# else
#  define RUNG_E_SOURCE "parse_url(\"file:///\")"
# endif
#endif

extern void (*php_capstone_sink)(const char *, unsigned long);
/* PHP_FUNCTION(parse_url) in ext/standard/url.c defines this. */
extern void zif_parse_url(INTERNAL_FUNCTION_PARAMETERS);

static void sink(const char *s, unsigned long n) { (void)s; (void)n; }

static unsigned err_count;

static void ub_error(int type, const char *fname, const uint lineno,
                     const char *format, va_list args)
{
    (void)fname; (void)lineno; (void)format; (void)args;
    ++err_count;
    switch (type) {            /* mirrors php_error_cb (main/main.c:773-781) */
        case E_CORE_ERROR:
        case E_ERROR:
        case E_COMPILE_ERROR:
        case E_USER_ERROR:
            EG(exit_status) = 255;
            zend_bailout();
        default:
            break;
    }
}
static int ub_printf(const char *fmt, ...) { (void)fmt; return 0; }
static int ub_write(const char *s, uint n)  { sink(s, n); return (int)n; }
static FILE *ub_fopen(const char *f, char **opened) { (void)f; (void)opened; return (FILE *)0; }
static void ub_message(long msg, void *data) { (void)msg; (void)data; }
static void ub_block(void)   { }
static void ub_unblock(void) { }
static int  ub_getcfg(char *name, uint nlen, zval *contents)
{ (void)name; (void)nlen; (void)contents; return FAILURE; }
static void ub_ticks(int t) { (void)t; }
static void ub_on_timeout(int sec TSRMLS_DC) { (void)sec; }
static int  ub_stream_open(const char *fn, zend_file_handle *h TSRMLS_DC)
{ (void)fn; (void)h; return FAILURE; }
static int ub_vspprintf(char **pbuf, size_t max_len, const char *fmt, va_list ap)
{
    int n = (max_len && max_len < 4096u) ? (int)max_len : 4095;
    *pbuf = (char *) malloc((unsigned long)n + 1);
    if (!*pbuf) { return 0; }
    return vsnprintf(*pbuf, (unsigned long)n + 1, fmt, ap);
}

/* The one hand-written thing: ext/standard has no module to register, so the function is
 * introduced directly. handler is url.c's own zif_parse_url. */
static zend_function_entry rung_e_functions[] = {
    { "parse_url", zif_parse_url, (struct _zend_arg_info *)0, 0, 0 },
    { (char *)0,   (void (*)(INTERNAL_FUNCTION_PARAMETERS))0, (struct _zend_arg_info *)0, 0, 0 }
};

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned st = 0;
    unsigned rtype = 0;
    *res = st;

    st |= ST_ENTERED;              *res = st;
    php_capstone_sink = sink;
    st |= ST_SINKED;               *res = st;

    {
        zend_utility_functions uf;
        uf.error_function              = ub_error;
        uf.printf_function             = ub_printf;
        uf.write_function              = ub_write;
        uf.fopen_function              = ub_fopen;
        uf.message_handler             = ub_message;
        uf.block_interruptions         = ub_block;
        uf.unblock_interruptions       = ub_unblock;
        uf.get_configuration_directive = ub_getcfg;
        uf.ticks_function              = ub_ticks;
        uf.on_timeout                  = ub_on_timeout;
        uf.stream_open_function        = ub_stream_open;
        uf.vspprintf_function          = ub_vspprintf;
        if (zend_startup(&uf, (char **)0, 1) != SUCCESS) { *res = st | 0x8000u; return; }
        st |= ST_STARTED;          *res = st;
    }

    init_compiler(TSRMLS_C);
    st |= ST_COMPILER;             *res = st;
    init_executor(TSRMLS_C);
    st |= ST_EXECUTOR;             *res = st;

    if (zend_register_functions((zend_class_entry *)0, rung_e_functions,
                                CG(function_table), MODULE_PERSISTENT TSRMLS_CC) != SUCCESS) {
        *res = st | 0x4000u; return;
    }
    st |= ST_REGISTERED;           *res = st;

#ifdef RUNG_E_REUSE
    /* Does the FREE-AND-REUSE path hand back an UNTAGGED pointer?
     *
     * The granule pass showed the tag is never killed in _estrndup's p slot -- p arrives
     * untagged -- and that the block involved is 112 bytes, allocated and used during the
     * parse, then freed and handed out again. 112 is above the 1..64 the earlier sweep
     * covered, and above PHP's own cache (real_size < 88), so it is served by the PORT's
     * cache and carve path.
     *
     * Each size: allocate, free, allocate again, and check the tag of BOTH. The second one is
     * the reuse. Reported: bits 16..31 = first size whose REUSED pointer is untagged, bit 10
     * = the first allocation was already untagged (would mean the fresh path, not reuse). */
    {
        unsigned bad_reuse = 0, bad_fresh = 0, n;
        for (n = 64u; n <= 512u && !bad_reuse && !bad_fresh; n += 4u) {
            void *a = emalloc(n);
            if (!a || !__builtin_capstone_cap_get_tag(a)) { bad_fresh = n; break; }
            efree(a);
            void *b = emalloc(n);
            if (!b || !__builtin_capstone_cap_get_tag(b)) { bad_reuse = n; break; }
            efree(b);
        }
        *res = st | (bad_fresh ? 0x400u : 0u)
                  | (((bad_reuse ? bad_reuse : bad_fresh) & 0xFFFFu) << 16);
        return;
    }
#endif
#ifdef RUNG_E_CACHEPROBE
    /* Does PHP's OWN allocator cache hold UNTAGGED pointers?
     *
     * The control-arm fault is cause 24 in memcpy from _estrndup (zend_alloc.c:403) with an
     * untagged destination. _estrndup's codegen is correct -- it stores p with `stc` and
     * reloads it for the memcpy with `ldc` (7ead8) -- so p was ALREADY untagged when
     * _emalloc returned it. And our arena allocator is clean: the ZEND_CAP_TAG_GUARD on both
     * of its return paths never fires. That leaves PHP's own _emalloc, which has exactly two
     * returns: the early CACHE HIT (zend_alloc.c:152, `p = AG(cache)[idx][--count]`) and the
     * fresh ZEND_DO_MALLOC fall-through.
     *
     * AG(v) is plain `alloc_globals.v` in a non-ZTS build, so the cache is directly
     * inspectable. This walks every live entry and counts the untagged ones. Run AFTER
     * startup + compile, which is what populates it, so the state is the realistic one.
     *
     *   bits 16..23  live cache entries examined (saturated)
     *   bits 24..31  of those, how many are UNTAGGED
     *   bit  10      set if any untagged entry was found  */
    {
        unsigned examined = 0, untagged = 0, i, j;
        for (i = 0; i < MAX_CACHED_MEMORY; ++i) {
            unsigned c = AG(cache_count)[i];
            if (c > MAX_CACHED_ENTRIES) { c = MAX_CACHED_ENTRIES; }
            for (j = 0; j < c; ++j) {
                void *q = AG(cache)[i][j];
                ++examined;
                if (!__builtin_capstone_cap_get_tag(q)) { ++untagged; }
            }
        }
        *res = st | (untagged ? 0x400u : 0u)
                  | ((examined > 255u ? 255u : examined) << 16)
                  | ((untagged > 255u ? 255u : untagged) << 24);
        return;
    }
#endif
#ifdef RUNG_E_ALLOCSCAN
    /* Diagnostic: is PHP's own _emalloc (Zend/zend_alloc.c, which wraps our malloc) handing
     * back an UNTAGGED pointer for some size? The control arm faults in memcpy called from
     * _estrndup (zend_alloc.c:403) with an untagged destination, and the guard inside OUR
     * arena allocator never fires -- so the tag is intact when our malloc returns and gone
     * by the time _estrndup uses it. This walks sizes and reports the first bad one.
     *   result: bits 16..31 = first size whose emalloc result is untagged (0 = all clean),
     *           bit 10      = set if a SECOND pass over the same sizes goes bad (cache path) */
    {
        unsigned bad = 0, badcached = 0, n;
        void *keep[64];
        /* Second sweep over LARGER sizes: the trigger path allocates op_arrays and hash
         * tables, not just 4-byte scheme strings, and the first sweep found nothing. */
        for (n = 64u; n <= 8192u && !bad; n += 8u) {
            void *q = emalloc(n);
            if (!q || !__builtin_capstone_cap_get_tag(q)) { bad = n; break; }
            efree(q);
        }
        for (n = 1; n <= 64u && !bad; ++n) {
            void *q = emalloc(n);
            if (!q || !__builtin_capstone_cap_get_tag(q)) { bad = n; break; }
            keep[n - 1] = q;
        }
        /* Free them all, then re-request: that exercises PHP's AG(cache) path, which returns
         * a pointer it stored at _efree time -- the obvious place for a tag to die. */
        if (!bad) {
            for (n = 1; n <= 64u; ++n) { efree(keep[n - 1]); }
            for (n = 1; n <= 64u && !badcached; ++n) {
                void *q = emalloc(n);
                if (!q || !__builtin_capstone_cap_get_tag(q)) { badcached = n; break; }
            }
        }
        *res = st | ((badcached ? 0x400u : 0u))
                  | (((bad ? bad : badcached) & 0xFFFFu) << 16);
        return;
    }
#endif
    {
        zval ret;
        int rc = FAILURE;
        static char source[] = RUNG_E_SOURCE;

        INIT_ZVAL(ret);
        zend_try {
            rc = zend_eval_string(source, &ret, "trigger" TSRMLS_CC);
        } zend_catch {
            st |= ST_BAILED;
        } zend_end_try();

        if (rc == SUCCESS) {
            st |= ST_EVALED;
            rtype = (unsigned)ret.type;
            if (ret.type == IS_ARRAY) { st |= ST_ISARRAY; }
        }
    }

    if (err_count) { st |= ST_ERRORS; }
    *res = st | ((rtype & 0xFu) << 10)
              | (((err_count > 255u) ? 255u : err_count) << 16);
}
