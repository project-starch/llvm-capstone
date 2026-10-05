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
/* Corpus cases selected by name, never by passing the source through EXTRA_CF -- that is
 * word-split, and an embedded quote silently corrupts every TU's command line.
 *
 * The parse_url pair are EXPRESSIONS, evaluated with a retval so the result can be checked.
 * The language-level cases are STATEMENTS: zend_eval_string wraps its argument in
 * `return <src> ;` when given a retval pointer, which does not parse for a statement list, so
 * those pass retval_ptr = NULL and are judged on completion instead. RUNG_E_STMT says which. */
#ifndef RUNG_E_SOURCE
# ifdef RUNG_E_CRASH073
#  define RUNG_E_SOURCE "parse_url(\"a:/\")"
# elif defined(RUNG_E_CRASH003)
#  define RUNG_E_SOURCE "$h=array();\n$h['me']=&$h;\n$h['me']=42;"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH004)
#  define RUNG_E_SOURCE "$a=array(1,2,3);\n$a=&$a[1];"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH010)
#  define RUNG_E_SOURCE "foreach(array(\"x\") as $f=>$f);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH067)
#  define RUNG_E_SOURCE "function &f() { return ($sa); }\n$sh =& f();\n$sh =& f();\n$sq[] = 1;"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH005)
#  define RUNG_E_SOURCE "a(1"
#  define RUNG_E_STMT   1
/* The object-model lifetime cases. All four are CWE-416 that ASAN cannot see through PHP's own
 * allocator (clean OK on the asan-stock build) -- which is exactly why they are worth running.
 *
 * var_dump is kept rather than removed, and registered as a no-op below. For CRASH-024 the
 * faulting access IS the argument fetch ASAN reports at zend_fetch_var_address, so deleting the
 * call would delete the bug; for the others the fault precedes the call and the stub only keeps
 * the opcode stream faithful. */
# elif defined(RUNG_E_CRASH024)
#  define RUNG_E_SOURCE \
     "class Foo {\n" \
     "  function getThis() { return $this; }\n" \
     "  function destroyThis() { $baz =& $this->getThis(); }\n" \
     "}\n" \
     "$bar = new Foo();\n" \
     "$bar->destroyThis();\n" \
     "var_dump($bar);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH025)
#  define RUNG_E_SOURCE \
     "class Foo { function __construct() { $this->error = array($this,$this); } }\n" \
     "$a =& new Foo();\n" \
     "class Bar { function __construct() { $this->_rme2 = $this; } }\n" \
     "$b =& new Bar();\n" \
     "$b->_rme2 = 0;\n" \
     "var_dump($b);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH042)
#  define RUNG_E_SOURCE \
     "class main { public static $dataAccessor = NULL; }\n" \
     "class relay { public function __get($name) { main::$dataAccessor = new relay; } }\n" \
     "main::$dataAccessor = new relay();\n" \
     "main::$dataAccessor->bar;\n" \
     "echo 'ok';"
#  define RUNG_E_STMT   1
/* Not an object case: a UAF entirely inside Zend/zend_builtin_functions.c, which the engine set
 * already links and zend_startup(..., 1) already registers. debug_print_backtrace fails to reset
 * arg_array across the eval frame, so it reads and re-frees a zval it already released. */
/* A constant redefined: zend_register_constant's failure path frees c->name and then passes the
 * same pointer to zend_error as %s. The read is performed by the message formatter, which is why
 * ub_error above must actually format. */
/* ---- the pure-ext cases, reachable only with PHP_CAP_EXT=pure linked ------------------------
 * Transcribed from corpus/CRASH-NNN.php. Two deliberate substitutions, both forced by the port and
 * neither touching the defect:
 *   - var_dump is the no-op stub registered above (output is not what is under test);
 *   - CRASH-091 writes 3 where the corpus writes ENT_QUOTES, because that constant is registered by
 *     PHP_MINIT_FUNCTION(html) and the domain runs no module init. 3 is its value
 *     (ENT_HTML_QUOTE_DOUBLE | ENT_HTML_QUOTE_SINGLE).
 */
# elif defined(RUNG_E_CRASH022)
#  define RUNG_E_SOURCE \
     "function cmp($x, $y) { global $a;\n" \
     "  unset($a[2]); unset($a[3]); unset($a[4]); unset($a[5]); unset($a[6]);\n" \
     "  $a[100]=1; $a[101]=2; $a[102]=3; $a[103]=4; $a[104]=5;\n" \
     "  return ($x < $y) ? -1 : 1; }\n" \
     "$a = array(5,3,8,1,9,2,7,4,6,0,11,12,13,14,15,16,17,18,19,20);\n" \
     "usort($a, 'cmp');\n" \
     "echo 'done ', count($a);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH035)
#  define RUNG_E_SOURCE \
     "function my_errorhandler($errno,$errormsg) { global $my_var; $my_var = 0; }\n" \
     "set_error_handler('my_errorhandler');\n" \
     "$my_var = str_repeat('A', $my_var[0]->errormsg = 'xyz');\n" \
     "echo 'ok';"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH047)
#  define RUNG_E_SOURCE \
     "function compareByRef(&$first, &$second) { return $first === $second; }\n" \
     "class MyTree implements ArrayAccess {\n" \
     "  public $parent;\n" \
     "  public $children = array();\n" \
     "  public function offsetExists($offset) {}\n" \
     "  public function offsetUnset($offset) {}\n" \
     "  public function offsetSet($offset, $value) {\n" \
     "    $cannonicalName = strtolower($offset);\n" \
     "    $this->children[$cannonicalName] = $value;\n" \
     "    $value->parent = $this; }\n" \
     "  public function offsetGet($offset) {\n" \
     "    $cannonicalName = strtolower($offset);\n" \
     "    return $this->children[$cannonicalName]; }\n" \
     "}\n" \
     "$id = 'Test';\n$root = new MyTree();\n$child = new MyTree();\n" \
     "$root[$id] = $child;\n" \
     "var_dump(compareByRef($root[$id], $child));"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH057)
#  define RUNG_E_SOURCE \
     "$s = 'AB';\n" \
     "$x = ($s[1] = chr(67) . '');\n" \
     "var_dump($x);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH066)
#  define RUNG_E_SOURCE \
     "class obj { var $ryat; public function __destruct() { $this->ryat = null; } }\n" \
     "$poc = 'O:3:\"obj\":1:{';\n" \
     "var_dump(unserialize($poc));"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH074)
#  define RUNG_E_SOURCE \
     "function cb($value, $key, &$userdata) { $userdata = str_repeat('A', 32); }\n" \
     "$array = array('hello', array('world', array('deep')));\n" \
     "$userdata = 'x';\n" \
     "array_walk_recursive($array, 'cb', $userdata);\n" \
     "echo 'Done';"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH080)
#  define RUNG_E_SOURCE \
     "function rd($carry, $item) { return $carry; }\n" \
     "$r = array_reduce(array(1,2,3), 'rd', array('initialarray'));\n" \
     "var_dump($r);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH091)
#  define RUNG_E_SOURCE \
     "$s = \"\\x8f\";\n" \
     "echo htmlspecialchars($s, 3, 'EUC-JP');"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH102)
#  define RUNG_E_SOURCE \
     "$s = str_repeat('A', 16);\n" \
     "$mask = \"A\\0\";\n" \
     "strspn($s, $mask, 0, 4294967296);\n" \
     "echo 'done';"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH103)
#  define RUNG_E_SOURCE \
     "$h = 'Hello World';\n$n = 'world';\n" \
     "$r = stristr($h, $n);\n" \
     "var_dump($r);\nvar_dump($h);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH151)
#  define RUNG_E_SOURCE \
     "$s = str_repeat('A', 8);\n" \
     "$a = true;\n$a->p = $s;\n" \
     "$q1 = array(1,2,3); $q2 = 'XXXXXXXXXXXXXXXX'; $q3 = 1.5;\n" \
     "echo $s;\nvar_dump($s);"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH160)
#  define RUNG_E_SOURCE \
     "class D { function __destruct() { unset($GLOBALS['x']); } }\n" \
     "$x = new D();\n" \
     "extract(array('x' => 'newval'));\n" \
     "echo 'after=', $x;"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH068)
#  define RUNG_E_SOURCE \
     "define('C', 1);\n" \
     "define('C', 2);"
#  define RUNG_E_STMT   1
/* is_a()/is_subclass_of() hold zval** pointing into EG(argument_stack).elements across
 * zend_lookup_class(), which runs userland __autoload; a deep enough nested call ereallocs that
 * block and the pointer dangles. */
# elif defined(RUNG_E_CRASH155)
#  define RUNG_E_SOURCE \
     "class A {}\n" \
     "function deep($n) { if ($n > 0) { deep($n - 1); } return $n; }\n" \
     "function __autoload($c) { deep(800); eval(\"class $c {}\"); }\n" \
     "$o = new A;\n" \
     "var_dump(is_a($o, 'LazyClass'));\n" \
     "var_dump(is_subclass_of($o, 'LazyClass2'));\n" \
     "echo 'done';"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH140)
#  define RUNG_E_SOURCE \
     "function f($a, $b, $c) { debug_print_backtrace(); }\n" \
     "eval('f(1,2,3);');"
#  define RUNG_E_STMT   1
# elif defined(RUNG_E_CRASH065)
#  define RUNG_E_SOURCE \
     "class foo { function foo($arrayobj) { var_dump($arrayobj); } }\n" \
     "new foo(array(new stdClass));"
#  define RUNG_E_STMT   1
/* Named explicitly as well as being the default: the runner derives -DRUNG_E_<case> from the case
 * id, and an id with no branch would silently compile the default source instead of failing. */
# elif defined(RUNG_E_CRASH110)
#  define RUNG_E_SOURCE "parse_url(\"file:///\")"
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
    /* THE MESSAGE IS FORMATTED, NOT DISCARDED, and that is a correctness requirement rather than a
     * nicety. Every real SAPI formats it (php_error_cb -> vspprintf, main/main.c), and at least one
     * corpus defect lives inside that formatting: CRASH-068 frees a constant's name and then hands
     * the same pointer to an E_NOTICE as %s, so the use-after-free IS the formatter's read. A
     * handler that ignored `format` would report that case MISSED for a reason that has nothing to
     * do with capabilities -- a false negative manufactured by the port. */
    char msgbuf[256];
    (void)fname; (void)lineno;
    vsnprintf(msgbuf, sizeof msgbuf, format, args);
    sink(msgbuf, 0);
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
/* A no-op var_dump. The object-model cases call it, and every one of them faults BEFORE the call
 * -- at the argument fetch, the array build or the property write -- so the body can be empty.
 * What matters is that the function EXISTS: an undefined call would bail out of the eval before
 * the opcodes under test had run. */
static void zif_rung_nop(INTERNAL_FUNCTION_PARAMETERS)
{
    (void)ht; (void)return_value; (void)this_ptr; (void)return_value_used;
}

#if defined(PHP_CAP_EXT_PURE)
/* The pure ext/standard entry points the corpus triggers need. Declared rather than included:
 * php_string.h and php_array.h drag in the module machinery, and all that is wanted is the
 * handler address. Registering them here is also what keeps the linker from garbage-collecting
 * the objects -- nothing else in the image references them. */
extern void zif_str_repeat(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_strtolower(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_chr(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_strspn(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_stristr(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_count(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_usort(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_array_walk_recursive(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_array_reduce(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_extract(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_unserialize(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_htmlspecialchars(INTERNAL_FUNCTION_PARAMETERS);
#endif

static zend_function_entry rung_e_functions[] = {
    { "parse_url", zif_parse_url,  (struct _zend_arg_info *)0, 0, 0 },
    { "var_dump",  zif_rung_nop,   (struct _zend_arg_info *)0, 0, 0 },
#if defined(PHP_CAP_EXT_PURE)
    { "str_repeat",           zif_str_repeat,           (struct _zend_arg_info *)0, 0, 0 },
    { "strtolower",           zif_strtolower,           (struct _zend_arg_info *)0, 0, 0 },
    { "chr",                  zif_chr,                  (struct _zend_arg_info *)0, 0, 0 },
    { "strspn",               zif_strspn,               (struct _zend_arg_info *)0, 0, 0 },
    { "stristr",              zif_stristr,              (struct _zend_arg_info *)0, 0, 0 },
    { "count",                zif_count,                (struct _zend_arg_info *)0, 0, 0 },
    { "usort",                zif_usort,                (struct _zend_arg_info *)0, 0, 0 },
    { "array_walk_recursive", zif_array_walk_recursive, (struct _zend_arg_info *)0, 0, 0 },
    { "array_reduce",         zif_array_reduce,         (struct _zend_arg_info *)0, 0, 0 },
    { "extract",              zif_extract,              (struct _zend_arg_info *)0, 0, 0 },
    { "unserialize",          zif_unserialize,          (struct _zend_arg_info *)0, 0, 0 },
    { "htmlspecialchars",     zif_htmlspecialchars,     (struct _zend_arg_info *)0, 0, 0 },
#endif
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
#ifdef RUNG_E_STMT
            /* Statements: no retval, so zend_eval_string does not wrap in `return ... ;`. */
            rc = zend_eval_string(source, (zval *)0, "trigger" TSRMLS_CC);
#else
            rc = zend_eval_string(source, &ret, "trigger" TSRMLS_CC);
#endif
        } zend_catch {
            st |= ST_BAILED;
        } zend_end_try();

        if (rc == SUCCESS) {
            st |= ST_EVALED;
#ifndef RUNG_E_STMT
            rtype = (unsigned)ret.type;
            if (ret.type == IS_ARRAY) { st |= ST_ISARRAY; }
#else
            st |= ST_ISARRAY;   /* statement cases: completion IS the pass condition */
#endif
        }
    }

    if (err_count) { st |= ST_ERRORS; }
    *res = st | ((rtype & 0xFu) << 10)
              | (((err_count > 255u) ? 255u : err_count) << 16);
}
