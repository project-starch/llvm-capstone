/* SPDX-License-Identifier: MIT */
/* Check-then-copy race experiment, domain side (module built with
 * CV_RACE_TEST). Thread A issues a native read() from a pipe into a
 * capability buffer of two pages. A second domain thread B, a real context
 * with its own register frame, is run by the module at one deterministic
 * point of the checked copy. B revokes the buffer's object (CREVOKE) and
 * re-derives a sentinel object over the same addresses, or munmaps the
 * buffer's mapping and maps a new one. This program only reports what it
 * observes; judge.py holds the pre-registered expectations.
 *
 * usage: race.dom VARIANT SITE PAGE2 TOUCH
 *   VARIANT obj | map | nop   SITE hold | between | fpre | fpost | window
 *   PAGE2   res | absent | cow  (state of the second page when A reads)
 *   TOUCH   1: B writes the sentinel bytes, 0: B only revokes/unmaps */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <capstone/capability.h>
#include "../../vm.h"

extern long rt_service(long nr, long a0, long a1);
extern long __capstone_vm_thread_create_frame(void *frame);
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
#define RT_ARM 0x7a10
#define RT_QUERY 0x7a11
#define RT_PREP 0x7a13
#define RT_IDS 0x7a14
#define RT_COLLECT 0x7a15
enum { S_HOLD = 1, S_BETWEEN, S_FPRE, S_FPOST, S_WINDOW };
#define PAGE 4096UL
#define SIZE (2 * PAGE)

struct ctl {
    void *parent, *child, *replacement, *newmap;
    unsigned long mode, off, len, touch, map_base, map_bytes, runs;
    long munmap_rc;
};

static long call(long nr, long fd, void *buf, long bytes)
{
    return __capstone_native_syscall(nr, (void *)(__uintcap_t)(unsigned long)fd, buf,
                                     (void *)(__uintcap_t)(unsigned long)bytes, 0, 0, 0);
}

/* Thread B: domain code only. Registers and the stack are its own. */
__attribute__((noinline, used)) static void b_entry(struct ctl *ctl)
{
    for (;;) {
        ++ctl->runs;
        if (ctl->mode == 1 || ctl->mode == 3) {
            capstone_cap_revoke_child(ctl->parent, ctl->child);
            if (ctl->mode == 3) rt_service(RT_COLLECT, 0, 0);
            volatile unsigned char *rep = capstone_cap_derive(ctl->parent, ctl->off, ctl->len);
            ctl->replacement = (void *)rep;
            if (ctl->touch)
                for (unsigned long i = 0; i < ctl->len; ++i) rep[i] = 'S';
        } else if (ctl->mode == 2) {
            sublet_cap slot;
            ctl->munmap_rc = __capstone_vm_unmap(ctl->map_base, ctl->map_bytes);
            if (!cap_vm_acquire(&slot, ctl->map_bytes, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) {
                volatile unsigned char *m = cap_vm_copyable(&slot);
                ctl->newmap = (void *)m;
                if (ctl->touch)
                    for (unsigned long i = 0; i < ctl->map_bytes; ++i) m[i] = 'S';
            }
        }
        __capstone_vm_wait();
    }
}

static unsigned long count(const volatile unsigned char *p, unsigned long n, unsigned char c)
{
    unsigned long k = 0;
    for (unsigned long i = 0; i < n; ++i) k += p[i] == c;
    return k;
}
static void fill(volatile unsigned char *p, unsigned long n, unsigned char c)
{
    for (unsigned long i = 0; i < n; ++i) p[i] = c;
}
static long q(long i) { return rt_service(RT_QUERY, i, 0); }

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    const char *variant = argv[1], *site = argv[2], *p2 = argv[3];
    int obj = !strcmp(variant, "obj") || !strcmp(variant, "objgc"), gc = !strcmp(variant, "objgc"), map = !strcmp(variant, "map"), touch = atoi(argv[4]);
    int site_id = !strcmp(site, "hold") ? S_HOLD : !strcmp(site, "between") ? S_BETWEEN :
                  !strcmp(site, "fpre") ? S_FPRE : !strcmp(site, "fpost") ? S_FPOST : S_WINDOW;
    sublet_cap slot, fslot;
    if (cap_vm_acquire(&slot, SIZE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) return 3;
    unsigned char *R = cap_vm_copyable(&slot);
    unsigned long base = __builtin_capstone_cap_get_cursor(R);
    fill(R, PAGE, 'A');
    if (!strcmp(p2, "res")) fill(R + PAGE, PAGE, 'A');
    long prep = -1;
    if (!strcmp(p2, "cow")) prep = rt_service(RT_PREP, base + PAGE, 0);
    struct ctl *ctl = malloc(sizeof(*ctl));
    memset(ctl, 0, sizeof(*ctl));
    unsigned char *buf = R, *zone = NULL;
    unsigned long zone_len = 0;
    if (obj) {
        ctl->mode = gc ? 3 : 1;
        ctl->parent = capstone_cap_derive(R, 0, SIZE);
        ctl->child = capstone_cap_derive(ctl->parent, 0, SIZE);
        buf = ctl->child;
        ctl->off = site_id == S_WINDOW ? 0 : PAGE;
        ctl->len = PAGE;
        zone = R + ctl->off; zone_len = PAGE;
    } else if (map) {
        ctl->mode = 2;
        ctl->map_base = base; ctl->map_bytes = SIZE;
    }
    ctl->touch = touch;
    if (obj || map || !strcmp(variant, "nop")) {
        if (cap_vm_acquire(&fslot, PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) return 4;
        unsigned char *frame = cap_vm_copyable(&fslot);
        unsigned char *stack = malloc(32768);
        void *gp;
        __asm__ volatile("movc %0, gp" : "=r"(gp));
        memset(frame, 0, PAGE);
        ((uint64_t *)frame)[7] = 3;
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16), (void *)b_entry);
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack + 32768);
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 3 * 16), gp);
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 10 * 16), ctl);
        capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)b_entry);
        long id = __capstone_vm_thread_create_frame(frame);
        if (id <= 0) {
            char t[64];
            int k = snprintf(t, sizeof t, "RACE setup-failed thread=%ld\n", id);
            write(1, t, k);
            return 5;
        }
        if (!strcmp(variant, "nop")) ctl->mode = 0;
    }
    rt_service(RT_ARM, strcmp(variant, "nop") ? site_id : S_BETWEEN, 0);
    long n = call(63, 0, buf, SIZE);          /* no I/O between arm and read */
    long fired = q(0), cause = q(1), done = q(2), bruns = q(3), held = q(4), unmap_held = q(5);
    unsigned long oldb = q(6), newb = q(7);
    long mrc = q(8), arc = q(9);
    unsigned long zq = 0, zs = 0, dq = 0;
    if (map && ctl->newmap) {
        zone = ctl->newmap; zone_len = SIZE;
    }
    if (zone) { zq = count(zone, zone_len, 'Q'); zs = count(zone, zone_len, 'S'); }
    if (obj) dq = count(R, SIZE, 'Q') - zq;
    if (obj) rt_service(RT_IDS, (long)__builtin_capstone_cap_get_cursor(&ctl->child), (long)__builtin_capstone_cap_get_cursor(&ctl->replacement));
    unsigned char *fresh = malloc(64);
    memset(fresh, 'A', 64);
    long n2 = call(63, 0, fresh, 16);
    char line[640];
    int k = snprintf(line, sizeof line,
        "RACE variant=%s site=%s p2=%s touch=%d n=%ld fired=%ld cause=%ld done=%ld bruns=%ld "
        "ctlruns=%lu zone_len=%lu zone_q=%lu zone_s=%lu data_q=%lu held=%ld unmap_held=%ld "
        "munmap_rc=%ld mmap_rc=%ld old_base=%lx new_base=%lx n2=%ld n2_q=%lu prep=%ld resolves=%ld pte1=%ld pte2=%ld collections=%ld\n",
        variant, site, p2, touch, n, fired, cause, done, bruns, ctl->runs, zone_len, zq, zs, dq,
        held, unmap_held, mrc, arc, oldb, newb, n2, count(fresh, 16, 'Q'), prep, q(11), q(12), q(13), q(14));
    write(1, line, k);
    return 0;
}
