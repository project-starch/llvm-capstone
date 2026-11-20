/* zones.c: a TCG plugin that counts executed guest instructions by the virtual address of the instruction, exactly
 * (nested lane, 2026-11-20). For the oneop kit (one hart, sv39, native task route):
 *   image      [img_lo, img_hi): the native image's code, mapped below 0x3ff8000000 (args img_lo= img_hi=, hex)
 *   user       other addresses below 2^38 that are not firmware: every other user-mode program
 *   firmware   [fw_lo, fw_hi) (default 0x80000000-0x80200000): M-mode code runs untranslated at its physical address
 *   kernel     addresses with bit 63 set
 *   other      the rest
 * The counts are written to the file named by arg out= at exit. The plugin changes no guest-visible state: under
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
        qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64,
                                                   &count[zone_of(qemu_plugin_insn_vaddr(insn))], 1);
    }
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    FILE *f = out_path[0] ? fopen(out_path, "w") : NULL;
    uint64_t total = 0;
    if (!f) return;
    for (int z = 0; z < ZONES; z++) { fprintf(f, "ZONE %s %" PRIu64 "\n", zone_name[z], count[z]); total += count[z]; }
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
        else return -1;
    }
    if (!out_path[0] || img_hi <= img_lo) return -1;
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
