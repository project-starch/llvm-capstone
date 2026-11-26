/* dcplugin.c (nested lane, 2026-11-26): the TCG plugin half of the record-access trace's format v3. It owns the D$
 * model (qemu target/riscv/cap_rev_dc.h, b42c8c644's wt_dcache geometry and replacement), publishes its address in
 * $CAPSTONE_DC_MODEL for the tracer in the same process, and feeds it every guest data READ at its physical address
 * (the program's, the kernel's and the firmware's alike; I/O excluded). Writes are counted: write-through with no
 * allocation, they change no model. QEMU's own page-table walks are not memory operations a plugin sees, so the PTW's
 * reads through the D$ are missing from occupancy (a bias toward warm, stated with every reading).
 * Args: out=FILE (totals at exit), seed0..seed3=N (the LFSRs' initial states, 1..255).
 * A capability load or store is 16 bytes and line-aligned; an access that crosses a 16-byte line reads both lines.
 * Window (2026-11-27): snap=FILE writes one line per store to the UART's transmit register (virt's UART0 THR,
 * physical 0x10000000 as the plugin API reports it for I/O), so each console byte carries the model's totals at the
 * moment it is written. A console marker line then gives a window's start and end in model counts. The UART lies in
 * a sub-page I/O region whose name is "anon...", not "serial" (the first build matched the name and counted 0), so
 * the test is the address; the I/O stores are histogrammed by (device, address) at exit as its check. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qemu-plugin.h>
#include "cap_rev_dc.h"

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static DcModel *m;
static char out_path[4096];
static uint64_t io_reads, nohw_reads, cross_reads;
static FILE *snap;
static uint64_t io_writes, serial_writes, thr_writes;
#define IOH_N 48
static struct { const char *name; uint64_t pa, n; } ioh[IOH_N];
static uint64_t ioh_other;
#define UART0_THR 0x10000000ULL
static void snap_line(void)
{
    uint64_t r = 0, h[DC_SEEDS] = { 0 };
    for (int s = 0; s < DC_SRCS; s++) {
        r += m->reads[s];
        for (int k = 0; k < DC_SEEDS; k++) {
            h[k] += m->read_hits[k][s];
        }
    }
    fprintf(snap, "u %" PRIu64 " reads %" PRIu64 " misses", thr_writes, r);
    for (int k = 0; k < DC_SEEDS; k++) {
        fprintf(snap, " %" PRIu64, r - h[k]);
    }
    fprintf(snap, " src");
    for (int s = 0; s < DC_SRCS; s++) {
        fprintf(snap, " %" PRIu64, m->reads[s]);
    }
    fprintf(snap, "\n");
    fflush(snap);
}
/* The count control: executed instructions by class, counted by an inline add independent of the memory callbacks.
 * load: LOAD/LOAD-FP and their compressed forms; store: STORE/STORE-FP and theirs; amo: AMO (LR/SC included);
 * ldc/stc: Capstone csldc/csstc (custom-2 0x5b funct3 3 / 4). The memory callbacks see load+amo reads (plus cross
 * reads) and store+amo writes; ldc/stc reach the model only through the tracer's helper hook (src cap). */
enum { K_LOAD, K_STORE, K_AMO, K_LDC, K_STC, K_N };
static uint64_t kcount[K_N];
static int insn_class(const uint8_t *b, size_t n)
{
    if (n == 2) {
        unsigned h = b[0] | b[1] << 8, q = h & 3, f3 = h >> 13;
        if (q == 0) return (f3 == 1 || f3 == 2 || f3 == 3) ? K_LOAD : (f3 == 5 || f3 == 6 || f3 == 7) ? K_STORE : -1;
        if (q == 2) return (f3 == 1 || f3 == 2 || f3 == 3) ? K_LOAD : (f3 == 5 || f3 == 6 || f3 == 7) ? K_STORE : -1;
        return -1;
    }
    if (n != 4) return -1;
    unsigned w = b[0] | b[1] << 8 | b[2] << 16 | (unsigned)b[3] << 24, op = w & 0x7f, f3 = (w >> 12) & 7;
    if (op == 0x03 || op == 0x07) return K_LOAD;
    if (op == 0x23 || op == 0x27) return K_STORE;
    if (op == 0x2f) return K_AMO;
    if (op == 0x5b && f3 == 3) return K_LDC;
    if (op == 0x5b && f3 == 4) return K_STC;
    return -1;
}

static void mem_cb(unsigned int vcpu, qemu_plugin_meminfo_t info, uint64_t vaddr, void *udata)
{
    (void)vcpu; (void)udata;
    if (qemu_plugin_mem_is_store(info)) {
        dc_write(m, DC_SRC_PROG);
        if (snap) {
            struct qemu_plugin_hwaddr *hs = qemu_plugin_get_hwaddr(info, vaddr);
            if (hs && qemu_plugin_hwaddr_is_io(hs)) {
                io_writes++;
                const char *nm = qemu_plugin_hwaddr_device_name(hs);
                uint64_t pa = qemu_plugin_hwaddr_phys_addr(hs);
                int j;
                for (j = 0; j < IOH_N && ioh[j].name && !(ioh[j].name == nm && ioh[j].pa == pa); j++) {
                }
                if (j < IOH_N) {
                    ioh[j].name = nm;
                    ioh[j].pa = pa;
                    ioh[j].n++;
                } else {
                    ioh_other++;
                }
                if ((pa & ~0xffULL) == UART0_THR) {
                    serial_writes++;
                }
                if (pa == UART0_THR) {
                    snap_line();
                    thr_writes++;
                }
            }
        }
        return;
    }
    struct qemu_plugin_hwaddr *hw = qemu_plugin_get_hwaddr(info, vaddr);
    if (!hw) {
        nohw_reads++;
        return;
    }
    if (qemu_plugin_hwaddr_is_io(hw)) {
        io_reads++;
        return;
    }
    uint64_t pa = qemu_plugin_hwaddr_phys_addr(hw);
    unsigned size = 1u << qemu_plugin_mem_size_shift(info);
    dc_read(m, pa, DC_SRC_PROG);
    if (((pa & 15) + size) > 16) {
        cross_reads++;
        dc_read(m, (pa | 15) + 1, DC_SRC_PROG);
    }
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *in = qemu_plugin_tb_get_insn(tb, i);
        qemu_plugin_register_vcpu_mem_cb(in, mem_cb, QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, NULL);
        int k = insn_class(qemu_plugin_insn_data(in), qemu_plugin_insn_size(in));
        if (k >= 0) {
            qemu_plugin_register_vcpu_insn_exec_inline(in, QEMU_PLUGIN_INLINE_ADD_U64, &kcount[k], 1);
        }
    }
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    (void)id; (void)p;
    FILE *f = out_path[0] ? fopen(out_path, "w") : NULL;
    if (!f) {
        return;
    }
    fprintf(f, "dcplugin sets %d ways %d seeds %d\n", DC_SETS, DC_WAYS, DC_SEEDS);
    fprintf(f, "nohw_reads %" PRIu64 " io_reads %" PRIu64 " cross_reads %" PRIu64 "\n", nohw_reads, io_reads, cross_reads);
    static const char *const src[DC_SRCS] = { "prog", "rec", "dir", "hdr", "cap" };
    for (int s = 0; s < DC_SRCS; s++) {
        fprintf(f, "src %s reads %" PRIu64 " writes %" PRIu64, src[s], m->reads[s], m->writes[s]);
        for (int k = 0; k < DC_SEEDS; k++) {
            fprintf(f, " hits%d %" PRIu64, k, m->read_hits[k][s]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "insns load %" PRIu64 " store %" PRIu64 " amo %" PRIu64 " ldc %" PRIu64 " stc %" PRIu64 "\n",
            kcount[K_LOAD], kcount[K_STORE], kcount[K_AMO], kcount[K_LDC], kcount[K_STC]);
    fprintf(f, "window snap %d io_writes %" PRIu64 " serial_writes %" PRIu64 " thr_writes %" PRIu64 "\n", snap != NULL,
            io_writes, serial_writes, thr_writes);
    for (int j = 0; j < IOH_N && ioh[j].name; j++) {
        fprintf(f, "iostore %s pa 0x%" PRIx64 " n %" PRIu64 "\n", ioh[j].name, ioh[j].pa, ioh[j].n);
    }
    fprintf(f, "iostore other n %" PRIu64 "\n", ioh_other);
    if (snap) {
        snap_line();
        fclose(snap);
        snap = NULL;
    }
    fprintf(f, "END\n");
    fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    (void)info;
    uint8_t seeds[DC_SEEDS] = { 0x01, 0x5a, 0xa5, 0x3c };
    for (int i = 0; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) {
            continue;
        }
        *eq = 0;
        if (!strcmp(argv[i], "out")) {
            snprintf(out_path, sizeof(out_path), "%s", eq + 1);
        } else if (!strcmp(argv[i], "snap")) {
            snap = fopen(eq + 1, "w");
            if (!snap) {
                fprintf(stderr, "dcplugin: cannot open %s\n", eq + 1);
                return -1;
            }
        } else if (!strncmp(argv[i], "seed", 4) && argv[i][4] >= '0' && argv[i][4] < '0' + DC_SEEDS && !argv[i][5]) {
            unsigned v = (unsigned)strtoul(eq + 1, NULL, 0);
            if (v < 1 || v > 255) {
                fprintf(stderr, "dcplugin: seed must be 1..255\n");
                return -1;
            }
            seeds[argv[i][4] - '0'] = (uint8_t)v;
        }
    }
    m = malloc(sizeof(*m));
    if (!m) {
        return -1;
    }
    dc_init(m, seeds);
    char buf[32];
    snprintf(buf, sizeof(buf), "%llx", (unsigned long long)(uintptr_t)m);
    setenv("CAPSTONE_DC_MODEL", buf, 1);
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
