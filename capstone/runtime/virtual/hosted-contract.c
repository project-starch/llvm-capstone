/* Hosted C semantics of virtual applications, and capability tags across the
 * copies a hosted compiler is free to inline.
 *
 * Applications compile hosted: clang knows memcpy, memmove and memset, and may
 * inline them or turn a loop into one of them. Every copy of a capability
 * must keep its tag. Each check reads the tag with the compiler builtin and
 * then the value through the copy, so a lost tag is reported, not faulted on.
 * The byte-copy control must lose the tag, which shows the tag read can
 * report a loss at all. Copies go through file-scope objects followed by a
 * compiler barrier, so the compiler cannot forward the source instead of
 * reading the copy. Native builds run the same checks without the tag reads.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __CAPSTONE_PURECAP__
#define TAG(p) __builtin_capstone_cap_get_tag((void *)(p))
#else
#define TAG(p) 1
#endif
#define BARRIER() __asm__ volatile("" ::: "memory")
#define N 64

static int cells[N];
static int checks, failures;

static void check(int ok, const char *name)
{
    ++checks;
    failures += !ok;
    printf("HOSTED_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
}

/* Tagged, the same pointer, and the value it names. */
static int good(int *p, int i)
{
    return TAG(p) && p == &cells[i] && *p == i * 7 + 1;
}

static int all_good(int *const *v, int n, int shift)
{
    for (int i = 0; i < n; ++i)
        if (!good(v[i], i + shift))
            return 0;
    return 1;
}

struct pair { int *a; int *b; };
struct many { int *v[N]; };
struct mixed { int *p; char c; };

static struct pair pair_src, pair_dst, pair_opaque;
static struct many many_src, many_dst;
static struct mixed mixed_src, mixed_dst;
static int *src[N], *dst[N], *loop_dst[N], *moved[N + 1];
static _Alignas(16) unsigned char head_src[48], head_dst[48], skew[48 + 8];

/* The compiler sees only void pointers: no alignment to rely on. */
__attribute__((noinline)) static void copy_pair(void *d, const void *s)
{
    memcpy(d, s, sizeof(struct pair));
}
__attribute__((noinline)) static void copy_n(void *d, const void *s, size_t n) { memcpy(d, s, n); }
__attribute__((noinline)) static void copy_loop(int **d, int *const *s, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        d[i] = s[i];
}
/* External and never inlined: no caller's alignment reaches the copy, as in
 * Perl's pack 'p', which stores a pointer with Copy(buf, cur, 16, char). Both
 * buffers are aligned at run time, so every whole granule must keep its tag
 * (C-81: the compiler used to inline these as byte loads and stores). */
__attribute__((noinline)) void hosted_copy16(char *d, const char *s) { memcpy(d, s, 16); }
__attribute__((noinline)) void hosted_copy24(char *d, const char *s) { memcpy(d, s, 24); }
__attribute__((noinline)) void hosted_copy48(char *d, const char *s) { memcpy(d, s, 48); }
__attribute__((noinline)) void hosted_move32(char *d, const char *s) { memmove(d, s, 32); }
/* A constant aggregate holding pointers, passed by value: the caller copies it
 * into its frame with memcpy (C-77: the call passed both operands untagged and
 * memcpy faulted on the source). The zero constant is the reported shape. */
struct byval_ptrs { int *a, *b, *c, *d, *e; unsigned long off; _Bool lex; };
static const struct byval_ptrs byval_zero = { 0 };
static const struct byval_ptrs byval_cells = { &cells[11], &cells[12], &cells[13], &cells[14],
                                               &cells[15], 5, 1 };
__attribute__((noinline)) int hosted_byval_zero(struct byval_ptrs c)
{
    return !c.a && !c.e && c.off == 0 && !c.lex;
}
__attribute__((noinline)) int hosted_byval_cells(struct byval_ptrs c)
{
    return good(c.a, 11) && good(c.c, 13) && good(c.e, 15) && c.off == 5 && c.lex;
}
#ifdef __CAPSTONE_PURECAP__
__attribute__((noinline)) static void copy_bytes(void *d, const void *s, size_t n)
{
    volatile unsigned char *to = d;
    const volatile unsigned char *from = s;
    for (size_t i = 0; i < n; ++i)
        to[i] = from[i];
}
#endif
static int by_address(const void *x, const void *y)
{
    int *a = *(int *const *)x, *b = *(int *const *)y;
    return (a > b) - (a < b);
}

/* One read() or write() on a regular file moves the whole request unless the
 * file ends, as on Linux; the runtime must not stop at its exchange region.
 * The pipe control must still return what is available at once: there a short
 * count is the kernel's answer, and waiting for the rest would block forever. */
/* alloca(0): the result is bounded to a region of its own, and SHRINK refuses
 * an empty range, so a zero-byte allocation still takes stack space. The size
 * comes through a volatile so the allocations stay dynamic, and the helper is
 * not inlined so they are not hoisted into a fixed frame. */
static char *volatile alloca_sink;

static __attribute__((noinline)) void zero_size_allocas(void)
{
    volatile size_t zero = 0, one = 1;
    char *a = __builtin_alloca(zero);
    BARRIER();
    char *b = __builtin_alloca(zero);
    BARRIER();
    char *c = __builtin_alloca(one);
    c[0] = 'c';
    alloca_sink = c; /* keeps the one-byte allocation and its store */
    BARRIER();
    check(a && b && alloca_sink[0] == 'c', "alloca_zero_survives");
#ifdef __CAPSTONE_PURECAP__
    check(a != b && TAG(a) && TAG(b) && TAG(alloca_sink), "alloca_zero_distinct_tagged");
#endif
}

static void transfers(void)
{
    enum { BIG = (1 << 20) + 123 };
    unsigned char *out = malloc(BIG), *in = malloc(BIG);
    for (int i = 0; i < BIG; ++i)
        out[i] = (unsigned char)(i * 131 + 7);
    const char *path = "/tmp/hosted-contract-transfer";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    check(fd >= 0 && write(fd, out, BIG) == BIG, "write_regular_file_whole");
    memset(in, 0, BIG);
    check(pread(fd, in, BIG, 0) == BIG && !memcmp(in, out, BIG), "pread_regular_file_whole");
    memset(in, 0, BIG);
    check(lseek(fd, 0, SEEK_SET) == 0 && read(fd, in, BIG) == BIG && !memcmp(in, out, BIG),
          "read_regular_file_whole");
    check(read(fd, in, BIG) == 0, "read_regular_file_end");
    check(pread(fd, in, BIG, BIG - 1000) == 1000 && !memcmp(in, out + BIG - 1000, 1000),
          "pread_short_at_end");
    close(fd);
    unlink(path);
    int p[2];
    check(pipe(p) == 0 && write(p[1], out, 100) == 100 && read(p[0], in, BIG) == 100,
          "control_pipe_read_returns_available");
    close(p[0]);
    close(p[1]);
    free(out);
    free(in);
}

/* __builtin_return_address and __builtin_frame_address. A return address is
 * the address after the call, carried by an untagged pointer: it names code and
 * grants nothing. A frame address is the frame pointer, a tagged capability to
 * the stack, and one level up is the caller's. Each builtin is used the three
 * ways the compiler lowers differently: returned, passed as an argument, and
 * stored. The callers are never inlined and each call is followed by code, so
 * every frame in the chain exists when it is walked. */
static void *ra_arg, *ra_stored[2];
static char *fa_stored[2];

__attribute__((noinline)) static void record_return_address(void *ra) { ra_arg = ra; }

/* The barrier keeps two calls two: without a side effect the compiler sees a
 * pure function and merges them, and both return to the first call site. */
__attribute__((noinline)) static void *returned_return_address(void)
{
    BARRIER();
    return __builtin_return_address(0);
}

__attribute__((noinline)) static void frame_leaf(void)
{
    ra_stored[1] = __builtin_return_address(1);
    fa_stored[1] = __builtin_frame_address(1);
    BARRIER();
}

__attribute__((noinline)) static void frame_middle(void)
{
    ra_stored[0] = __builtin_return_address(0);
    fa_stored[0] = __builtin_frame_address(0);
    record_return_address(__builtin_return_address(0));
    frame_leaf();
    BARRIER();
}

static void frame_addresses(void);

/* A return address lies in frame_addresses, after its entry. A loose bound:
 * the order check below is what pins each value to its own call site. */
static int in_caller(void *ra)
{
    char *at = ra, *entry = (char *)(void *)frame_addresses;
    return at > entry && at - entry < 4096;
}

__attribute__((noinline)) static void frame_addresses(void)
{
    void *before = returned_return_address();
    frame_middle();
    void *after = returned_return_address();
    BARRIER();
    check(in_caller(before) && in_caller(after), "return_address_returned");
    check(in_caller(ra_stored[0]) && ra_arg == ra_stored[0], "return_address_argument");
    /* Three calls in straight-line code return to three increasing addresses. */
    check((char *)before < (char *)ra_stored[0] && (char *)ra_stored[0] < (char *)after,
          "return_address_call_site_order");
#ifdef __CAPSTONE_PURECAP__
    check(!TAG(before) && !TAG(after) && !TAG(ra_stored[0]) && !TAG(ra_arg),
          "return_address_untagged");
    /* One level up from the leaf is the middle frame's own view. Depth 1 needs
     * every frame to keep a frame pointer, which the virtual profile does and a
     * native -O2 build need not. */
    check(ra_stored[1] == ra_stored[0] && !TAG(ra_stored[1]), "return_address_depth_1");
    check(TAG(fa_stored[0]) && TAG(fa_stored[1]) && fa_stored[1] == fa_stored[0],
          "frame_address_depth_1");
#endif
}

/* The library memset over every destination offset 0-47 and every length up
 * to 300, through a pointer the compiler cannot inline: the bytes inside the
 * range take the fill value (negative and wide values truncated), the bytes
 * outside keep theirs, it returns its destination, every 16-byte granule the
 * range touches loses its tag and every other keeps it; and exactly bounded
 * objects of 1-300 bytes are filled without a store outside them. */
static void *(*volatile memset_call)(void *, int, size_t) = memset;
static int *slots[24];

static void memset_ranges(void)
{
    unsigned char before[sizeof slots];
    unsigned char *b = (unsigned char *)slots;
    int bytes = 1, returns = 1, bounded = 1;
#ifdef __CAPSTONE_PURECAP__
    int tags = 1;
#endif
    for (size_t off = 0; off < 48; ++off)
        for (size_t len = 0; len <= 300 && off + len <= sizeof slots; ++len) {
            int fill = (int)(off * 37 + len * 11) % 1024 - 512;
            for (int i = 0; i < 24; ++i)
                slots[i] = &cells[i];
            for (size_t i = 0; i < sizeof before; ++i)
                before[i] = b[i];
            returns &= memset_call(b + off, fill, len) == b + off;
            BARRIER();
            for (size_t i = 0; i < sizeof before; ++i)
                bytes &= b[i] == (i >= off && i < off + len ? (unsigned char)fill : before[i]);
#ifdef __CAPSTONE_PURECAP__
            for (size_t g = 0; g < 24; ++g)
                tags &= TAG(slots[g]) == !(len && off < g * 16 + 16 && off + len > g * 16);
#endif
        }
    for (size_t n = 1; n <= 300; ++n) {
        unsigned char *p = malloc(n);
        if (!p) { bounded = 0; break; }
        memset_call(p, 0x5a, n);
        BARRIER();
        for (size_t i = 0; i < n; ++i)
            bounded &= p[i] == 0x5a;
        free(p);
    }
    check(bytes, "memset_ranges_bytes");
    check(returns, "memset_ranges_returns_destination");
#ifdef __CAPSTONE_PURECAP__
    check(tags, "memset_ranges_clear_touched_granules_only");
#endif
    check(bounded, "memset_bounded_objects");
}

/* The library memcpy and memmove at every source and destination offset mod 16
 * (memmove: 0-31, overlapping in both directions) and every length up to 80,
 * through pointers the compiler cannot inline, with both ranges exactly
 * bounded: the bytes copied are the source's, nothing else changes, and a
 * destination granule keeps the tag of its source granule when the whole
 * granule is copied and both agree mod 16, loses its tag when the range
 * touches it otherwise, and keeps its own when untouched. */
static void *(*volatile memcpy_call)(void *, const void *, size_t) = memcpy;
static void *(*volatile memmove_call)(void *, const void *, size_t) = memmove;
static int *copy_src[16], *copy_dst[16], *move_buf[16];

static void fill_tagged(int **v, int shift)
{
    for (int i = 0; i < 16; ++i)
        v[i] = &cells[(i + shift) % N];
}

#ifdef __CAPSTONE_PURECAP__
static unsigned char *bounded(void *p, size_t off, size_t len)
{
    unsigned char *q = (unsigned char *)p + off;
    return len ? __builtin_capstone_cap_shrink(q, (unsigned long)q, (unsigned long)q + len) : q;
}
#else
static unsigned char *bounded(void *p, size_t off, size_t len) { (void)len; return (unsigned char *)p + off; }
#endif

#ifdef __CAPSTONE_PURECAP__
/* Expected tag of granule g of a destination after copying len bytes from
 * source offset so to destination offset doff, given the tags before. */
static int expect_tag(size_t g, size_t so, size_t doff, size_t len, const int *src_tag, const int *dst_tag)
{
    size_t lo = g * 16, hi = lo + 16;
    if (!len || hi <= doff || lo >= doff + len)
        return dst_tag[g];
    if (lo >= doff && hi <= doff + len && (so - doff) % 16 == 0)
        return src_tag[(lo - doff + so) / 16];
    return 0;
}
#endif

static void copy_ranges(void)
{
    unsigned char sb[sizeof copy_src], db[sizeof copy_dst], mb[sizeof move_buf];
    int stag[16], dtag[16], bytes = 1, returns = 1;
#ifdef __CAPSTONE_PURECAP__
    int tags = 1;
#endif
    for (size_t so = 0; so < 16; ++so)
        for (size_t doff = 0; doff < 16; ++doff)
            for (size_t len = 0; len <= 80; ++len) {
                fill_tagged(copy_src, 1);
                fill_tagged(copy_dst, 9);
                memcpy(sb, copy_src, sizeof sb);
                memcpy(db, copy_dst, sizeof db);
                for (int g = 0; g < 16; ++g) {
                    stag[g] = (int)TAG(copy_src[g]);
                    dtag[g] = (int)TAG(copy_dst[g]);
                }
                unsigned char *d = bounded(copy_dst, doff, len);
                returns &= memcpy_call(d, bounded(copy_src, so, len), len) == d;
                BARRIER();
                unsigned char *dn = (unsigned char *)copy_dst, *sn = (unsigned char *)copy_src;
                for (size_t i = 0; i < sizeof db; ++i) {
                    bytes &= dn[i] == (i >= doff && i < doff + len ? sb[i - doff + so] : db[i]);
                    bytes &= sn[i] == sb[i];
                }
#ifdef __CAPSTONE_PURECAP__
                for (size_t g = 0; g < 16; ++g)
                    tags &= (int)TAG(copy_dst[g]) == expect_tag(g, so, doff, len, stag, dtag);
#endif
            }
    check(bytes, "memcpy_ranges_bytes");
    check(returns, "memcpy_ranges_returns_destination");
#ifdef __CAPSTONE_PURECAP__
    check(tags, "memcpy_ranges_tags");
#endif
    bytes = 1; returns = 1;
#ifdef __CAPSTONE_PURECAP__
    tags = 1;
#endif
    for (size_t so = 0; so < 32; ++so)
        for (size_t doff = 0; doff < 32; ++doff)
            for (size_t len = 0; len <= 80 && so + len <= sizeof mb && doff + len <= sizeof mb; ++len) {
                fill_tagged(move_buf, 3);
                memcpy(mb, move_buf, sizeof mb);
                for (int g = 0; g < 16; ++g)
                    stag[g] = dtag[g] = (int)TAG(move_buf[g]);
                unsigned char *d = bounded(move_buf, doff, len);
                returns &= memmove_call(d, bounded(move_buf, so, len), len) == d;
                BARRIER();
                unsigned char *b = (unsigned char *)move_buf;
                for (size_t i = 0; i < sizeof mb; ++i)
                    bytes &= b[i] == (i >= doff && i < doff + len ? mb[i - doff + so] : mb[i]);
#ifdef __CAPSTONE_PURECAP__
                /* memmove onto itself writes nothing, so no tag changes */
                for (size_t g = 0; g < 16; ++g)
                    tags &= (int)TAG(move_buf[g]) ==
                            (so == doff ? dtag[g] : expect_tag(g, so, doff, len, stag, dtag));
#endif
            }
    (void)stag; (void)dtag;
    check(bytes, "memmove_ranges_bytes");
    check(returns, "memmove_ranges_returns_destination");
#ifdef __CAPSTONE_PURECAP__
    check(tags, "memmove_ranges_tags");
#endif
}

int main(void)
{
    for (int i = 0; i < N; ++i) {
        cells[i] = i * 7 + 1;
        src[i] = &cells[i];
        many_src.v[i] = &cells[i];
    }
    printf("HOSTED_INFO:stdc_hosted=%d\n", __STDC_HOSTED__);

    pair_src = (struct pair){&cells[1], &cells[2]};
    memcpy(&pair_dst, &pair_src, sizeof pair_dst);
    BARRIER();
    check(good(pair_dst.a, 1) && good(pair_dst.b, 2), "memcpy_struct_known_alignment");

    many_dst = many_src;
    BARRIER();
    check(all_good(many_dst.v, N, 0), "struct_assignment_64_pointers");

    copy_pair(&pair_opaque, &pair_src);
    BARRIER();
    check(good(pair_opaque.a, 1) && good(pair_opaque.b, 2), "memcpy_constant_size_unknown_alignment");

    copy_n(dst, src, sizeof src);
    BARRIER();
    check(all_good(dst, N, 0), "memcpy_runtime_size");

    copy_loop(loop_dst, src, N);
    BARRIER();
    check(all_good(loop_dst, N, 0), "loop_copy_runtime_count");

    memcpy(moved, src, sizeof src);
    memmove(moved + 1, moved, sizeof src);
    BARRIER();
    check(all_good(moved + 1, N, 0), "memmove_overlapping");

    mixed_src = (struct mixed){&cells[3], 'x'};
    memcpy(&mixed_dst, &mixed_src, sizeof mixed_dst);
    BARRIER();
    check(good(mixed_dst.p, 3) && mixed_dst.c == 'x', "memcpy_padded_struct");

    /* 24 bytes: a capability in the first granule, plain data after it. */
    memcpy(head_src, &src[4], sizeof src[4]);
    memset(head_src + 16, 0x5a, 8);
    memcpy(head_dst, head_src, 24);
    BARRIER();
    int *head;
    memcpy(&head, head_dst, sizeof head);
    check(good(head, 4) && head_dst[23] == 0x5a, "memcpy_24_capability_head");

    /* The same copy through the builtin, which inlines even under -fno-builtin:
     * this one checks the compiler's own lowering of a partial-granule copy. */
    memset(head_dst, 0, sizeof head_dst);
    __builtin_memcpy(head_dst, head_src, 24);
    BARRIER();
    memcpy(&head, head_dst, sizeof head);
    check(good(head, 4) && head_dst[23] == 0x5a, "builtin_memcpy_24_capability_head");

    /* Constant-size copies whose alignment the compiler cannot see. */
    static _Alignas(16) char opaque_src[64], opaque_dst[64];
    int *got, *got2;
    memset(opaque_src, 0, sizeof opaque_src);
    memset(opaque_dst, 0, sizeof opaque_dst);
    memcpy(opaque_src, &src[7], sizeof src[7]);
    hosted_copy16(opaque_dst, opaque_src);
    BARRIER();
    memcpy(&got, opaque_dst, sizeof got);
    check(good(got, 7), "memcpy_16_unknown_alignment");

    memset(opaque_dst, 0, sizeof opaque_dst);
    memset(opaque_src + 16, 0x6b, 8);
    hosted_copy24(opaque_dst, opaque_src);
    BARRIER();
    memcpy(&got, opaque_dst, sizeof got);
    check(good(got, 7) && opaque_dst[23] == 0x6b && opaque_dst[24] == 0,
          "memcpy_24_unknown_alignment");

    /* Plain data around a capability in the middle granule. */
    memset(opaque_src, 0x3c, 48);
    memcpy(opaque_src + 16, &src[8], sizeof src[8]);
    memset(opaque_dst, 0, sizeof opaque_dst);
    hosted_copy48(opaque_dst, opaque_src);
    BARRIER();
    memcpy(&got, opaque_dst + 16, sizeof got);
    check(good(got, 8) && opaque_dst[0] == 0x3c && opaque_dst[47] == 0x3c && opaque_dst[48] == 0,
          "memcpy_48_unknown_alignment_interior");

    /* Overlapping, forwards by one granule: both capabilities move intact. */
    memset(opaque_dst, 0, sizeof opaque_dst);
    memcpy(opaque_dst, &src[9], sizeof src[9]);
    memcpy(opaque_dst + 16, &src[10], sizeof src[10]);
    hosted_move32(opaque_dst + 16, opaque_dst);
    BARRIER();
    memcpy(&got, opaque_dst + 16, sizeof got);
    memcpy(&got2, opaque_dst + 32, sizeof got2);
    check(good(got, 9) && good(got2, 10), "memmove_32_unknown_alignment_overlap");

    int **grown = malloc(4 * sizeof *grown);
    for (int i = 0; i < 4; ++i)
        grown[i] = &cells[10 + i];
    int **old = grown;
    grown = realloc(grown, 1 << 20);
    check(grown && grown != old && all_good(grown, 4, 10), "realloc_moving");
    free(grown);

    int *shuffled[N];
    for (int i = 0; i < N; ++i)
        shuffled[i] = &cells[(i * 37) % N];
    qsort(shuffled, N, sizeof shuffled[0], by_address);
    check(all_good(shuffled, N, 0), "qsort_pointers");

    memset(dst, 0, sizeof dst);
    BARRIER();
#ifdef __CAPSTONE_PURECAP__
    check(!TAG(dst[0]) && !TAG(dst[N - 1]), "memset_clears_tags");
#endif

    /* A capability cannot live at an address that is not granule-aligned. */
    memcpy(skew + 8, &src[5], sizeof src[5]);
    int *unaligned;
    memcpy(&unaligned, skew + 8, sizeof unaligned);
#ifdef __CAPSTONE_PURECAP__
    check(!TAG(unaligned), "misaligned_copy_has_no_tag");
#endif
    check(unaligned == &cells[5], "misaligned_copy_keeps_address");

#ifdef __CAPSTONE_PURECAP__
    int *control;
    copy_bytes(&control, &src[6], sizeof control);
    check(!TAG(control) && control == &cells[6], "control_byte_copy_drops_tag");
#endif

    frame_addresses();
    transfers();
    zero_size_allocas();
    memset_ranges();
    copy_ranges();

    /* Last: before C-77 was fixed the first one faulted. */
    check(hosted_byval_zero(byval_zero), "byval_const_zero_aggregate");
    check(hosted_byval_cells(byval_cells), "byval_const_pointer_aggregate");

    printf("HOSTED_RESULT:%d/%d\n", checks - failures, checks);
    return failures != 0;
}
