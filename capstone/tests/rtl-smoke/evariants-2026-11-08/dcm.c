/* dcm: L1D / DTLB geometry model for CVA6 wt_dcache, one pass, several geometries.
 * L1D: 16-byte lines, 4 KiB per way (VIPT), physically tagged; LOADS allocate (wt_dcache_missunit: MSHR for
 * reads only, miss_o = the perf event), STORES write through and never allocate; replacement = first invalid
 * way, else pseudo-random (an xorshift per geometry; the RTL uses an 8-bit LFSR advanced per miss).
 * DTLB: fully associative, true LRU (the RTL is tree-PLRU), 4 KiB pages for non-kernel addresses and 2 MiB for
 * kernel addresses (the linear map), every load and store.
 * Accesses are classed by the executing instruction's vaddr: K (bit 63 set), F (firmware, [0x8000_0000,
 * 0x8100_0000)), U (everything else). Control: per class, the count of executed custom-2 LDC/STC (funct3 3/4)
 * and the memory callbacks they produced. LDC executed with 0 callbacks = helper accesses invisible = VOID.
 * args: out=<file> */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <qemu-plugin.h>
QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define NC 3
enum { CK, CF, CU, NCLASS };
static const char *cname[NCLASS] = { "K", "F", "U" };
static const int ways_cfg[] = { 8, 16, 32, 8 };    /* last = 8-way true LRU (policy sensitivity) */
#define NL 4
static const int tlb_cfg[] = { 16, 32, 64 };
#define NT 3
#define SETS 256

typedef struct { uint64_t tag[SETS][32]; uint8_t vld[SETS][32]; uint64_t age[SETS][32]; uint64_t rnd; } l1_t;
static l1_t *l1[NL];
typedef struct { uint64_t page[64]; uint64_t age[64]; int n; } tlb_t;
static tlb_t tlb[NT];
static uint64_t tick;
static uint64_t loads[NCLASS], stores[NCLASS], l1miss[NL][NCLASS], tlbmiss[NT][NCLASS];
static uint64_t capins[NCLASS], capmem[NCLASS], io[NCLASS];
static char *outpath;

static inline uint64_t xs(uint64_t *s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x; }

static void l1_load(int c, uint64_t pa, int cls)
{
    l1_t *m = l1[c]; int W = ways_cfg[c]; uint64_t set = (pa >> 4) & (SETS - 1), tag = pa >> 12;
    for (int w = 0; w < W; w++)
        if (m->vld[set][w] && m->tag[set][w] == tag) { m->age[set][w] = tick; return; }
    l1miss[c][cls]++;
    int v = -1;
    for (int w = 0; w < W; w++) if (!m->vld[set][w]) { v = w; break; }
    if (v < 0) {
        if (c == NL - 1) { v = 0; for (int w = 1; w < W; w++) if (m->age[set][w] < m->age[set][v]) v = w; }
        else v = xs(&m->rnd) % W;
    }
    m->vld[set][v] = 1; m->tag[set][v] = tag; m->age[set][v] = tick;
}

static void tlb_access(uint64_t va, int cls)
{
    uint64_t pg = (va >> 63) ? ((va >> 21) | (1ULL << 62)) : (va >> 12);
    for (int t = 0; t < NT; t++) {
        tlb_t *b = &tlb[t]; int N = tlb_cfg[t], hit = -1;
        for (int i = 0; i < b->n; i++) if (b->page[i] == pg) { hit = i; break; }
        if (hit >= 0) { b->age[hit] = tick; continue; }
        tlbmiss[t][cls]++;
        int v;
        if (b->n < N) v = b->n++;
        else { v = 0; for (int i = 1; i < N; i++) if (b->age[i] < b->age[v]) v = i; }
        b->page[v] = pg; b->age[v] = tick;
    }
}

static void mem_cb(unsigned int vcpu, qemu_plugin_meminfo_t info, uint64_t va, void *ud)
{
    uintptr_t u = (uintptr_t)ud; int cls = u & 3, iscap = (u >> 2) & 1;
    struct qemu_plugin_hwaddr *hw = qemu_plugin_get_hwaddr(info, va);
    tick++;
    if (iscap) capmem[cls]++;
    if (hw && qemu_plugin_hwaddr_is_io(hw)) { io[cls]++; return; }
    uint64_t pa = hw ? qemu_plugin_hwaddr_phys_addr(hw) : va;
    if (qemu_plugin_mem_is_store(info)) { stores[cls]++; }
    else { loads[cls]++; for (int c = 0; c < NL; c++) l1_load(c, pa, cls); }
    tlb_access(va, cls);
}

static void cap_exec(unsigned int vcpu, void *ud) { capins[(uintptr_t)ud & 3]++; }

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t pc = qemu_plugin_insn_vaddr(insn);
        int cls = (pc >> 63) ? CK : (pc >= 0x80000000ULL && pc < 0x81000000ULL) ? CF : CU;
        uint32_t op = 0; const uint8_t *d = qemu_plugin_insn_data(insn);
        if (qemu_plugin_insn_size(insn) == 4) memcpy(&op, d, 4);
        int iscap = (op & 0x7f) == 0x5b && (((op >> 12) & 7) == 3 || ((op >> 12) & 7) == 4);
        uintptr_t ud = cls | (iscap << 2);
        if (iscap) qemu_plugin_register_vcpu_insn_exec_cb(insn, cap_exec, QEMU_PLUGIN_CB_NO_REGS, (void *)ud);
        qemu_plugin_register_vcpu_mem_cb(insn, mem_cb, QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, (void *)ud);
    }
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    FILE *f = outpath ? fopen(outpath, "w") : stderr;
    if (!f) f = stderr;
    for (int k = 0; k < NCLASS; k++) {
        fprintf(f, "DCM %s loads %" PRIu64 " stores %" PRIu64 " io %" PRIu64 " capins %" PRIu64 " capmem %" PRIu64,
                cname[k], loads[k], stores[k], io[k], capins[k], capmem[k]);
        for (int c = 0; c < NL; c++) fprintf(f, " l1_%d%s %" PRIu64, ways_cfg[c], c == NL - 1 ? "lru" : "", l1miss[c][k]);
        for (int t = 0; t < NT; t++) fprintf(f, " tlb_%d %" PRIu64, tlb_cfg[t], tlbmiss[t][k]);
        fprintf(f, "\n");
    }
    fprintf(f, "DCM-END\n");
    if (f != stderr) fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    for (int i = 0; i < argc; i++) if (!strncmp(argv[i], "out=", 4)) outpath = strdup(argv[i] + 4);
    for (int c = 0; c < NL; c++) { l1[c] = calloc(1, sizeof(l1_t)); l1[c]->rnd = 0x9e3779b97f4a7c15ULL + c; }
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
