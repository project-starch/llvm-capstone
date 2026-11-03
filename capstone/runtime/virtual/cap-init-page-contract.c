/* Acceptance fixture for page-lazy capability globals. Build with the virtual
 * SDK and cap-init-page-probe.S. Run each mode in a fresh process: observing a
 * slot before the selected operation would invalidate a first-touch test.
 * CAP_INIT_WEAK builds a separate image because an eager weak initializer can
 * fail before main. See the page-init design for the expected observations. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define PAGE 4096
#define COUNT 8

unsigned char cap_init_target[64] = {37};
extern unsigned char cap_init_alias[64]
    __attribute__((alias("cap_init_target")));

static int target_function(void) { return 37; }

struct page_slots {
    unsigned char *object, *interior, *end;
    int (*function)(void);
    const char *literal;
    unsigned char *alias;
    unsigned char padding[PAGE - 6 * sizeof(void *)];
};
_Static_assert(sizeof(struct page_slots) == PAGE, "one holder per page");
#define SLOTS {cap_init_target, cap_init_target + 7, cap_init_target + 64, \
               target_function, "page-init", cap_init_alias, {0}}
volatile struct page_slots cap_init_pages[COUNT] __attribute__((aligned(PAGE))) = {
    SLOTS, SLOTS, SLOTS, SLOTS, SLOTS, SLOTS, SLOTS, SLOTS
};

static __thread unsigned char *thread_pointer;

#ifdef CAP_INIT_WEAK
extern int cap_init_missing_object __attribute__((weak));
extern void cap_init_missing_function(void) __attribute__((weak));
int *volatile cap_init_weak_object = &cap_init_missing_object;
void (*volatile cap_init_weak_function)(void) = cap_init_missing_function;
#endif

static void require(int ok, const char *what)
{
    if (!ok) { fprintf(stderr, "CAP_INIT_FAIL:%s\n", what); exit(1); }
}

/* Compare both the tag and the entire stored encoding against an ordinary
 * materialization. This also compares the lifetime identity, which LCC cannot
 * query. Volatile integer loads read the encoding without dereferencing p. */
__attribute__((noinline))
static void same_cap(void *p, void *expected)
{
    volatile uint64_t a[2] __attribute__((aligned(16)));
    volatile uint64_t b[2] __attribute__((aligned(16)));
    __asm__ volatile("stc %1, 0(%0)" : : "r"(a), "r"(p) : "memory");
    __asm__ volatile("stc %1, 0(%0)" : : "r"(b), "r"(expected) : "memory");
    require(__builtin_capstone_cap_get_tag(p) ==
            __builtin_capstone_cap_get_tag(expected), "tag");
    require(a[0] == b[0] && a[1] == b[1], "capability encoding");
}

static void check_page(unsigned i)
{
    same_cap(cap_init_pages[i].object, cap_init_target);
    same_cap(cap_init_pages[i].interior, cap_init_target + 7);
    same_cap(cap_init_pages[i].end, cap_init_target + 64);
    same_cap((void *)cap_init_pages[i].function, (void *)target_function);
    same_cap((void *)cap_init_pages[i].literal, (void *)"page-init");
    same_cap(cap_init_pages[i].alias, cap_init_alias);
    require(cap_init_pages[i].function() == 37, "function call");
    require(!strcmp(cap_init_pages[i].literal, "page-init"), "literal");
    require(__builtin_capstone_cap_get_tag((void *)cap_init_pages[i].literal),
            "literal tag");
    require(cap_init_pages[i].object[0] == 37, "object load");
}

extern unsigned long cap_init_page_probe(void *page, unsigned long narrow);

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";
    /* Address formation alone must not admit the holder pages. The module's
     * diag_admit trace can then count just this interval, excluding libc. */
    printf("CAP_INIT_REGION:%lx:%lu\n", (unsigned long)cap_init_pages,
           (unsigned long)sizeof(cap_init_pages));
    if (!strcmp(mode, "cold")) {
        /* Deliberately leave all slots untouched. */
    } else if (!strcmp(mode, "one")) {
        check_page(3);
    } else if (!strcmp(mode, "all")) {
        for (unsigned i = 0; i < COUNT; ++i) check_page(i);
        thread_pointer = cap_init_target;
        same_cap(thread_pointer, cap_init_target);
    } else if (!strcmp(mode, "overwrite")) {
        /* A scalar write clears the tag. Initialization must precede this
         * first write, and no retry or later access may resurrect the slot. */
        volatile uint64_t *slot = (volatile uint64_t *)&cap_init_pages[3].object;
        slot[0] = 0;
        require(!cap_init_pages[3].object, "first scalar store kept");
        require(!__builtin_capstone_cap_get_tag(cap_init_pages[3].object),
                "scalar store tag stays clear");
        same_cap(cap_init_pages[3].interior, cap_init_target + 7);
    } else if (!strcmp(mode, "narrow") || !strcmp(mode, "broad")) {
        void *p = mmap(NULL, PAGE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(p != MAP_FAILED, "probe mapping");
        require(cap_init_page_probe(p, !strcmp(mode, "narrow")) == 0,
                "probe load");
        require(!munmap(p, PAGE), "probe unmap");
#ifdef CAP_INIT_WEAK
    } else if (!strcmp(mode, "weak")) {
        same_cap(cap_init_weak_object, NULL);
        same_cap((void *)cap_init_weak_function, NULL);
#endif
    } else {
        require(0, "unknown mode");
    }
    printf("CAP_INIT_OK:%s\n", mode);
    return 0;
}
