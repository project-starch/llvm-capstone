/* Phase 2 rung B: does zend_startup() complete inside a domain?
 *
 * This is the first rung that runs real engine logic. zend_startup builds the function
 * table, the constant table, the class table and the object store -- i.e. it exercises
 * zend_hash, zend_alloc (through our capability-bounding malloc), the capability-global
 * initialisers, and the static function-pointer tables, all at once.
 *
 * Progress is reported through *res so a fault tells us HOW FAR it got rather than just
 * that it died: each stage ORs in a bit before attempting the next.
 */
#include <zend.h>
#include <zend_API.h>

#define ST_ENTERED   0x01u   /* domain_main reached */
#define ST_SINKED    0x02u   /* output sink installed */
#define ST_STARTED   0x04u   /* zend_startup returned */
#define ST_TABLES    0x08u   /* function/class tables are non-empty */
#define ST_ALLOC     0x10u   /* emalloc/efree round-trip after startup */

extern void (*php_capstone_sink)(const char *, unsigned long);
extern unsigned long php_capstone_dropped;
#include <alloca.h>
#include <setjmp.h>

/* --- stack-depth watchdog plumbing (see libc/php_capstone_depth.c) --- */
extern unsigned long php_depth_floor, php_depth_low, php_depth_mallocs;
extern unsigned long php_depth_frames, php_depth_mode_ra, php_depth_mode_count;
extern int php_depth_armed;
static jmp_buf depth_jb;
extern void (*php_depth_trip_fn)(void);

/* Is depth_jb still intact when the watchdog fires?
 *
 * longjmp is proven correct at 201 frames (probes/setjmp-depth-probe.c), yet after the
 * watchdog's longjmp the domain faults in domain_main with sp tagged, bounds correct, cursor
 * BELOW base -- i.e. a DEEP stack capability. That is what the jmp_buf's sp slot would hold
 * if something overwrote it during startup. Inspect the slot, record what it held, REPAIR it
 * and then unwind: if the repaired unwind returns cleanly, corruption is proven. */
static void *jb_expect_sp;
static unsigned jb_diag;          /* bit0 slot tagged, bit1 cursor matched */
static void php_depth_trip(void)
{
    /* sp lives at offset 16 in the jmp_buf (capstone_setjmp.S), 16-byte aligned. */
    void **slot = (void **)((char *)&depth_jb[0] + 16);
    void *saved = *slot;
    if (__builtin_capstone_cap_get_tag(saved)) { jb_diag |= 1u; }
    if (__builtin_capstone_cap_get_cursor(saved)
        == __builtin_capstone_cap_get_cursor(jb_expect_sp)) { jb_diag |= 2u; }
    /* Report through the fault channel BEFORE attempting any unwind, so the result does
     * not depend on longjmp being correct here.
     *   [63:56] magic 0xD1
     *   [55:40] frames walked
     *   [39:24] repeat count of the most frequent return address
     *   [23: 2] that address's offset from zend_startup (signed via bit 1)
     *   [ 1: 0] jb_diag: bit0 slot tagged, bit1 cursor matched expectation */
    {
        extern void php_fault_report(unsigned long);
        unsigned long refv = __builtin_capstone_cap_get_cursor((void *)&zend_startup);
        unsigned long off  = (php_depth_mode_ra >= refv) ? (php_depth_mode_ra - refv)
                                                         : (refv - php_depth_mode_ra);
        unsigned long neg  = (php_depth_mode_ra >= refv) ? 0UL : 1UL;
        unsigned long v = (0xD1UL << 56)
                        | ((php_depth_frames     & 0xFFFFUL) << 40)
                        | ((php_depth_mode_count & 0xFFFFUL) << 24)
                        | ((off & 0x3FFFFFUL) << 2)
                        | (neg << 1)
                        | (unsigned long)(jb_diag & 1u);
        php_fault_report(v);
    }
    *slot = jb_expect_sp;
    longjmp(depth_jb, 1);
}

static unsigned long sink_bytes;
static void sink(const char *s, unsigned long n) { (void)s; sink_bytes += n; }

/* --- the twelve utility functions zend_startup wants (Zend/zend.h) --- */
static void ub_error(int type, const char *fname, const uint lineno,
                     const char *fmt, va_list ap)
{ (void)type; (void)fname; (void)lineno; (void)fmt; (void)ap; }

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

int vsnprintf(char *, unsigned long, const char *, va_list);
void *malloc(unsigned long);
/* zend_vspprintf allocates the buffer itself. Two passes: measure, then render. */
static int ub_vspprintf(char **pbuf, size_t max_len, const char *fmt, va_list ap)
{
    char probe[1];
    int n = vsnprintf(probe, 1, fmt, ap);
    if (n < 0) { n = 0; }
    if (max_len && (size_t)n > max_len) { n = (int)max_len; }
    *pbuf = (char *) malloc((unsigned long)n + 1);
    if (!*pbuf) { return 0; }
    return vsnprintf(*pbuf, (unsigned long)n + 1, fmt, ap);
}

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned st = 0;
    *res = st;

    st |= ST_ENTERED;              *res = st;
    php_capstone_sink = sink;
    st |= ST_SINKED;               *res = st;

#ifdef RUNG_B_STAGE
    /* Mechanical bisect of zend_startup's PRE-MALLOC steps. The watchdog is proven to fire
     * (RUNG_B_WATCHDOG_SELFTEST) yet never fired during zend_startup, so the stack blows
     * before zend_startup's first allocation (strdup, Zend/zend.c:590). These are the
     * callable steps that run before it. -DRUNG_B_STAGE=N runs stages 1..N and returns N. */
    {
        extern int zend_startup_extensions_mechanism(void);
        extern void zend_init_opcodes_handlers(void);
#if RUNG_B_STAGE >= 1
        zend_startup_extensions_mechanism();
#endif
#if RUNG_B_STAGE >= 2
        zend_init_opcodes_handlers();
#endif
        *res = 0xB000u | (unsigned)RUNG_B_STAGE;
        return;
    }
#endif
#ifdef RUNG_B_WATCHDOG_SELFTEST
    /* POSITIVE CONTROL ON THE INSTRUMENT. Every "the watchdog did not trip" conclusion is
     * worthless until the watchdog is shown to fire when it should. Arm it with a floor
     * ABOVE the current cursor so the very next malloc must trip, then call malloc. */
    {
        void *fp0 = __builtin_frame_address(0);
        unsigned long top0 = __builtin_capstone_cap_get_cursor(fp0);
        php_depth_floor  = top0 + 4096UL;      /* unconditionally true */
        jb_expect_sp = fp;
        php_depth_trip_fn = php_depth_trip;
        php_depth_armed  = 1;
        if (setjmp(depth_jb) != 0) {
            *res = 0x5EFF0000u | (unsigned)(php_depth_frames & 0xFFFFu);  /* tripped */
            return;
        }
        (void)malloc(16);
        *res = 0x5EFF0BADu;                    /* malloc returned without tripping */
        return;
    }
#endif
#ifdef RUNG_B_MEASURE_STACK
    /* Report usable stack headroom and stop. domreq.S says DOMREQ_STACK is "diagnostics
     * only" -- only DOMREQ_DATA binds -- so the only way to know what the carve actually
     * left for the stack is to read it. */
    {
        void *fp = __builtin_frame_address(0);
        unsigned long b = __builtin_capstone_cap_get_base(fp);
        unsigned long c = __builtin_capstone_cap_get_cursor(fp);
        *res = (unsigned)((((c > b) ? (c - b) : 0UL) >> 10) << 8) | st;
        return;
    }
#endif
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

#ifdef RUNG_B_NO_STARTUP
    /* Bisect: everything except zend_startup itself. Separates "the support layer works
     * at this image size" from "zend_startup faults". uf is still built and referenced so
     * the link is identical. */
    if (uf.write_function == (void *)0) { *res = st | 0x40u; return; }
    st |= ST_STARTED;               *res = st;
#else
#ifdef RUNG_B_DEPTH_WATCH
    {
        void *fp = __builtin_frame_address(0);
        unsigned long b = __builtin_capstone_cap_get_base(fp);
        unsigned long top = __builtin_capstone_cap_get_cursor(fp);
        php_depth_low   = top;
        /* Trip EARLY -- after only ~512 KB of the 2.6 MB has gone. The first attempt used
         * a floor near the base, which meant malloc was never called again before the
         * overflow and the watchdog never fired. */
        (void)b;
        php_depth_floor = top - (64UL * 1024UL);   /* trip after only 64 KB */
        php_depth_trip_fn = php_depth_trip;
        php_depth_armed = 1;
        if (setjmp(depth_jb) != 0) {
            /* Tripped: report how much stack was consumed (KB) and how many mallocs in.
             * A LOW malloc count with a HUGE depth means unbounded recursion; a high count
             * with growing depth means the startup path is simply deep. */
            /* Report the MODE of the frame chain: offset of the most frequent return
             * address from zend_startup (resolvable with llvm-nm), plus its repeat count.
             * A high count at one address IS the recursive call site. */
            unsigned long refv = __builtin_capstone_cap_get_cursor((void *)&zend_startup);
            unsigned long d = (php_depth_mode_ra > refv) ? (php_depth_mode_ra - refv)
                                                         : (refv - php_depth_mode_ra);
            unsigned cnt = (unsigned)(php_depth_mode_count > 0xFFFu ? 0xFFFu : php_depth_mode_count);
            unsigned sign = (php_depth_mode_ra >= refv) ? 0u : 1u;
            /* jb_diag in the low 2 bits tells whether the jmp_buf had been clobbered. */
            *res = 0x80000000u | (sign << 30) | ((unsigned)(d & 0x3FFFFu) << 12)
                 | ((cnt & 0x3FFu) << 2) | (jb_diag & 3u);
            return;
        }
    }
#endif
    if (zend_startup(&uf, (char **)0, 1) != SUCCESS) { *res = st | 0x80u; return; }
    /* ST_STARTED was only ever set on the NO_STARTUP branch, so a fully passing real run
     * reported 27 (ENTERED|SINKED|TABLES|ALLOC) and looked like it had skipped startup.
     * Reaching here means zend_startup returned SUCCESS. */
    st |= ST_STARTED;
#endif

    /* zend_startup registered the builtin functions and the standard constants; if the
     * hash tables came up empty, startup "succeeded" without doing anything. */
#ifndef RUNG_B_NO_STARTUP
    if (CG(function_table) && CG(function_table)->nNumOfElements > 0) { st |= ST_TABLES; }
#else
    st |= ST_TABLES;
#endif
    *res = st;

    /* One allocation round-trip through the capability allocator, post-startup. */
    {
        char *p = (char *) emalloc(64);
        if (p) { p[0] = 'z'; p[63] = 'e'; if (p[0] == 'z' && p[63] == 'e') { st |= ST_ALLOC; } efree(p); }
    }
    /* Report the alloca high-water mark alongside the stage bits, so a pass still says
     * HOW MUCH off-stack alloca the startup path needed. Upper bits = KB. */
    *res = st | ((unsigned)(php_capstone_alloca_peak >> 10) << 8);
}
