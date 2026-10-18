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

    transfers();
    memset_ranges();

    printf("HOSTED_RESULT:%d/%d\n", checks - failures, checks);
    return failures != 0;
}
