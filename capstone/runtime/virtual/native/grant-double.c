/* Executable interface double, not an MM or processor implementation. */
#include "../native-grant.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct mapping {
    uint64_t ns, id, base, bytes, epoch;
    unsigned rights, holds;
    int mapped, live, descendants, locked, sentinel;
};
static const char *mutant = "none";
static unsigned checks, failed, released_live;
#define CHECK(x) do { ++checks; if (!(x)) { ++failed; \
    fprintf(stderr, "FAIL line=%d check=%s\n", __LINE__, #x); } } while (0)
static int broken(const char *which) { return !strcmp(mutant, which); }
static struct mapping fresh(void)
{
    return (struct mapping){.ns=9, .id=17, .base=0x10000, .bytes=8192,
        .epoch=3, .rights=6, .mapped=1, .live=1, .descendants=1, .sentinel=42};
}
static int pair(const struct mapping *m) { return m->mapped == m->live; }
static int revoke(struct mapping *m)
{
    if (m->locked) return -EBUSY;
    m->live = m->descendants = 0;
    ++m->epoch;
    return 0;
}
static void release(struct mapping *m)
{
    if (m->live) ++released_live;
    m->mapped = 0;
}
static int unmap_double(struct mapping *m, uint64_t ns,
                        const struct cv_grant_transaction *r)
{
    int rc = 0;
    if (r->version != CV_GRANT_ABI_VERSION || r->flags || r->reserved ||
        r->inject > CV_GRANT_INJECT_RETIRE) return -EINVAL;
    if (r->operation != CV_GRANT_UNMAP) return -EOPNOTSUPP;
    if ((!broken("namespace") && m->ns != ns) || m->id != r->id ||
        m->base != r->address || m->bytes != r->bytes || !m->live)
        return -EACCES;
    if (m->holds || m->locked) return -EBUSY;
    m->locked = 1;
    if (broken("retire-first")) {
        m->live = m->descendants = 0;
        ++m->epoch;
    }
    /* All five failures precede the authority commit. */
    for (unsigned step = 1; step <= CV_GRANT_INJECT_RETIRE; ++step)
        if (r->inject == step) { rc = -ENOMEM; goto out; }
    if (broken("unmap-first")) release(m);
    if (!broken("skip-retire")) {
        m->live = m->descendants = 0;
        ++m->epoch;
    }
    release(m);
out:
    m->locked = 0;
    return rc;
}

static int fault_double(struct mapping *m, uint64_t ns, uint64_t event,
                        uint64_t cause, uint64_t address, int sleeping_revoker)
{
    enum cv_fault_class kind = cv_fault_classify(event, cause);
    unsigned required = kind == CV_FAULT_PAGE_WRITE ? 2 :
                        kind == CV_FAULT_PAGE_EXEC ? 1 : 4;
    uint64_t epoch = m->epoch;
    if (kind == CV_FAULT_DENIED) return -EACCES;
    if (!broken("grant-check") &&
        (m->ns != ns || !m->live || !m->mapped || address < m->base ||
         address - m->base >= m->bytes || !(m->rights & required)))
        return -EACCES;
    m->locked = !broken("sleep-revoke");
    if (sleeping_revoker && !revoke(m)) {
        /* Reuse by a different authority while the pager sleeps. */
        m->live = m->mapped = 1;
        m->id++;
        m->sentinel = 99;
    }
    /* A lock-dropping backend must reject this epoch change. The deliberate
     * mutant skips both exclusion and revalidation. */
    if (m->epoch != epoch && !broken("sleep-revoke")) {
        m->locked = 0;
        return -ESTALE;
    }
    if (kind == CV_FAULT_PAGE_WRITE) m->sentinel = 43;
    m->locked = 0;
    return 0;
}

int main(int argc, char **argv)
{
    struct mapping m;
    struct cv_grant_transaction r = {.version=CV_GRANT_ABI_VERSION,
        .operation=CV_GRANT_UNMAP, .id=17, .address=0x10000, .bytes=8192};
    if (argc == 2 && !strncmp(argv[1], "--mutant=", 9)) mutant = argv[1] + 9;
    else if (argc != 1) return 2;
    if (strcmp(mutant,"none") && strcmp(mutant,"namespace") &&
        strcmp(mutant,"retire-first") && strcmp(mutant,"unmap-first") &&
        strcmp(mutant,"skip-retire") && strcmp(mutant,"grant-check") &&
        strcmp(mutant,"sleep-revoke")) return 2;
    CHECK(sizeof(r) == 48);
    for (unsigned injection=1; injection<=CV_GRANT_INJECT_RETIRE; ++injection) {
        m=fresh(); r.inject=injection;
        CHECK(unmap_double(&m,9,&r) < 0);
        CHECK(m.mapped && m.live && m.descendants && m.epoch == 3);
        CHECK(pair(&m));
    }
    r.inject=0; m=fresh();
    CHECK(!unmap_double(&m,9,&r));
    CHECK(!m.mapped && !m.live && !m.descendants && m.epoch == 4);
    CHECK(pair(&m) && released_live == 0);
    CHECK(unmap_double(&m,9,&r) < 0); /* Replay. */
    m=fresh();
    CHECK(unmap_double(&m,10,&r) < 0);
    CHECK(m.live && m.descendants && m.epoch == 3);
    m=fresh(); m.holds=1;
    CHECK(unmap_double(&m,9,&r) == -EBUSY && m.live);
    m=fresh(); r.id++;
    CHECK(unmap_double(&m,9,&r) < 0 && m.live); r.id--;
    r.bytes=4096;
    CHECK(unmap_double(&m,9,&r) < 0 && m.live); r.bytes=8192;
    r.address=UINT64_MAX-4095;
    CHECK(unmap_double(&m,9,&r) < 0 && m.live); r.address=0x10000;
    for (unsigned op=CV_GRANT_MAP; op<=CV_GRANT_REMAP+1; ++op) {
        if (op == CV_GRANT_UNMAP) continue;
        r.operation=op;
        CHECK(unmap_double(&m,9,&r) == -EOPNOTSUPP && m.live);
    }
    r.operation=CV_GRANT_UNMAP; r.version++;
    CHECK(unmap_double(&m,9,&r) == -EINVAL && m.live); r.version--;
    r.flags=1;
    CHECK(unmap_double(&m,9,&r) == -EINVAL && m.live); r.flags=0;
    r.reserved=1;
    CHECK(unmap_double(&m,9,&r) == -EINVAL && m.live); r.reserved=0;
    for (unsigned cause=0; cause<32; ++cause) {
        int page = cause == 12 || cause == 13 || cause == 15;
        CHECK((cv_fault_classify(4,cause) != CV_FAULT_DENIED) == page);
        CHECK(cv_fault_classify(2,cause) == CV_FAULT_DENIED);
    }
    m=fresh(); CHECK(!fault_double(&m,9,4,15,0x10000,0));
    CHECK(m.sentinel == 43);
    m=fresh(); CHECK(fault_double(&m,9,4,15,0x12000,0) < 0);
    CHECK(m.sentinel == 42);
    m=fresh(); CHECK(fault_double(&m,8,4,15,0x10000,0) < 0);
    CHECK(m.sentinel == 42);
    m=fresh(); m.live=0;
    CHECK(fault_double(&m,9,4,15,0x10000,0) < 0);
    CHECK(m.sentinel == 42);
    m=fresh(); m.rights=4;
    CHECK(fault_double(&m,9,4,15,0x10000,0) < 0);
    CHECK(m.sentinel == 42);
    m=fresh(); CHECK(fault_double(&m,9,4,25,0x10000,0) < 0);
    CHECK(m.sentinel == 42);
    m=fresh(); CHECK(!fault_double(&m,9,4,15,0x10000,1));
    CHECK(m.epoch == 3 && m.id == 17 && m.sentinel == 43);
    m=fresh(); CHECK(!revoke(&m));
    CHECK(fault_double(&m,9,4,15,0x10000,0) < 0);
    printf("grant-double mutant=%s checks=%u failed=%u released_live=%u\n",
           mutant, checks, failed, released_live);
    return failed ? 1 : 0;
}
