/* zones.c: a TCG plugin that counts executed guest instructions by the virtual address of the instruction, exactly
 * (nested lane, 2026-11-20). For the oneop kit (one hart, sv39, native task route):
 *   image      [img_lo, img_hi): the native image's code, mapped below 0x3ff8000000 (args img_lo= img_hi=, hex)
 *   user       other addresses below 2^38 that are not firmware: every other user-mode program
 *   firmware   [fw_lo, fw_hi) (default 0x80000000-0x80200000): M-mode code runs untranslated at its physical address
 *   kernel     addresses with bit 63 set
 *   other      the rest
 * Named ranges (arg fn=NAME:LO:HI, hex, up to 32; or arg syms=FILE, lines `NAME LO HI` in hex, up to 65536,
 * sorted by LO and not overlapping) are counted besides: instructions executed inside each, and
 * entries (executions of its first instruction, i.e. calls if nothing jumps back to it).
 * The counts are written to the file named by arg out= at exit. For the twin runner (which passes only out= and
 * shift=): shift= is ignored, the image window defaults to [0x3ff0000000, 0x3ff8000000), and $ZONES_SYMS names a
 * symbol file (as syms=). The plugin changes no guest-visible state: under
 * -icount the guest's instret is the same with and without it. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

enum { IMAGE, USER, FIRMWARE, KERNEL, OTHER, ZONES };
static const char *const zone_name[ZONES] = { "image", "user", "firmware", "kernel", "other" };
static uint64_t count[ZONES];
static uint64_t img_lo, img_hi, fw_lo = 0x80000000ull, fw_hi = 0x80200000ull;
static char out_path[4096];
#define MAX_FN 65536
static struct { char name[64]; uint64_t lo, hi, insns, entries; } fn[MAX_FN];
static int n_fn;

static int zone_of(uint64_t va)
{
    if (va >= img_lo && va < img_hi) return IMAGE;
    if (va >> 63) return KERNEL;
    if (va >= fw_lo && va < fw_hi) return FIRMWARE;
    if (va < (1ull << 38)) return USER;
    return OTHER;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t va = qemu_plugin_insn_vaddr(insn);
        qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &count[zone_of(va)], 1);
        /* the last range that starts at or below va (ranges sorted, disjoint) */
        int lo = 0, hi = n_fn;
        while (lo < hi) { int mid = (lo + hi) / 2; if (fn[mid].lo <= va) lo = mid + 1; else hi = mid; }
        for (int f = lo - 1; f >= 0 && f == lo - 1; f--) {
            if (va >= fn[f].lo && va < fn[f].hi) {
                qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &fn[f].insns, 1);
                if (va == fn[f].lo) {
                    qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &fn[f].entries, 1);
                }
            }
        }
    }
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    FILE *f = out_path[0] ? fopen(out_path, "w") : NULL;
    uint64_t total = 0;
    if (!f) return;
    for (int z = 0; z < ZONES; z++) { fprintf(f, "ZONE %s %" PRIu64 "\n", zone_name[z], count[z]); total += count[z]; }
    for (int k = 0; k < n_fn; k++) {
        fprintf(f, "FN %s insns %" PRIu64 " entries %" PRIu64 " range 0x%" PRIx64 "-0x%" PRIx64 "\n",
                fn[k].name, fn[k].insns, fn[k].entries, fn[k].lo, fn[k].hi);
    }
    fprintf(f, "ZONE total %" PRIu64 "\nZONE image_range 0x%" PRIx64 "-0x%" PRIx64 "\nEND\n", total, img_lo, img_hi);
    fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) return -1;
        *eq = 0;
        if (!strcmp(argv[i], "img_lo")) img_lo = strtoull(eq + 1, NULL, 16);
        else if (!strcmp(argv[i], "img_hi")) img_hi = strtoull(eq + 1, NULL, 16);
        else if (!strcmp(argv[i], "fw_lo")) fw_lo = strtoull(eq + 1, NULL, 16);
        else if (!strcmp(argv[i], "fw_hi")) fw_hi = strtoull(eq + 1, NULL, 16);
        else if (!strcmp(argv[i], "out")) snprintf(out_path, sizeof(out_path), "%s", eq + 1);
        else if (!strcmp(argv[i], "shift")) { /* the twin runner's; ignored */ }
        else if (!strcmp(argv[i], "syms")) {
            FILE *sf = fopen(eq + 1, "r");
            char name[64];
            unsigned long long a, b;
            if (!sf) return -1;
            while (n_fn < MAX_FN && fscanf(sf, "%63s %llx %llx", name, &a, &b) == 3) {
                if (b <= a || (n_fn && a < fn[n_fn - 1].hi)) { fclose(sf); return -1; }
                snprintf(fn[n_fn].name, sizeof(fn[n_fn].name), "%s", name);
                fn[n_fn].lo = a; fn[n_fn].hi = b; n_fn++;
            }
            fclose(sf);
        }
        else if (!strcmp(argv[i], "fn") && n_fn < MAX_FN) {
            char *c1 = strchr(eq + 1, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
            if (!c1 || !c2) return -1;
            *c1 = 0;
            snprintf(fn[n_fn].name, sizeof(fn[n_fn].name), "%s", eq + 1);
            fn[n_fn].lo = strtoull(c1 + 1, NULL, 16);
            fn[n_fn].hi = strtoull(c2 + 1, NULL, 16);
            if (fn[n_fn].hi <= fn[n_fn].lo) return -1;
            n_fn++;
        }
        else return -1;
    }
    if (!img_lo && !img_hi) { img_lo = 0x3ff0000000ull; img_hi = 0x3ff8000000ull; }
    if (getenv("ZONES_SYMS") && !n_fn) {
        FILE *sf = fopen(getenv("ZONES_SYMS"), "r");
        char name[64];
        unsigned long long a, b;
        if (!sf) return -1;
        while (n_fn < MAX_FN && fscanf(sf, "%63s %llx %llx", name, &a, &b) == 3) {
            if (b <= a || (n_fn && a < fn[n_fn - 1].hi)) { fclose(sf); return -1; }
            snprintf(fn[n_fn].name, sizeof(fn[n_fn].name), "%s", name);
            fn[n_fn].lo = a; fn[n_fn].hi = b; n_fn++;
        }
        fclose(sf);
    }
    if (!out_path[0] || img_hi <= img_lo) return -1;
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
