/* opcount.c: executed-instruction census of the Capstone opcode 0x5B by code region (nested lane, 2026-11-20).
 * Every executed 32-bit instruction whose opcode (bits 6:0) is 0x5B is counted by region and by its funct3, and for
 * funct3 1 by funct7 as well (the R-type lifetime and domain instructions); LCC (funct3 1, funct7 0x04) is split by
 * whether its zimm (the rs2 field) is 0. Regions, by the instruction's virtual address:
 *   image     [0x3ff0000000, 0x3ff8000000): native images are mapped just below 0x3ff8000000
 *   vmlinux   [0xffffffff80000000, 2^64): the kernel image mapping on riscv64
 *   modules   [0xffffffff00000000, 0xffffffff80000000): the module area below the kernel image
 *   kernel    other addresses with bit 63 set
 *   firmware  [0x80000000, 0x80200000): M-mode firmware at its physical address
 *   user      other addresses below 2^38
 *   other     the rest
 * Also counted: all executed instructions per region (the denominator). Args: out=FILE (written at exit); shift= is
 * accepted and ignored (the twin runner passes it). The plugin changes no guest-visible state. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

enum { IMAGE, VMLINUX, MODULES, KERNEL, FIRMWARE, USER, OTHER, REGIONS };
static const char *const region_name[REGIONS] = { "image", "vmlinux", "modules", "kernel", "firmware", "user", "other" };
static uint64_t all_insns[REGIONS];
/* [region][funct3][funct7 for funct3 1, else 0][lcc zimm0 flag] */
static uint64_t cnt[REGIONS][8][128][2];
static char out_path[4096];

static int region_of(uint64_t va)
{
    if (va >= 0x3ff0000000ull && va < 0x3ff8000000ull) return IMAGE;
    if (va >= 0xffffffff80000000ull) return VMLINUX;
    if (va >= 0xffffffff00000000ull) return MODULES;
    if (va >> 63) return KERNEL;
    if (va >= 0x80000000ull && va < 0x80200000ull) return FIRMWARE;
    if (va < (1ull << 38)) return USER;
    return OTHER;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        int r = region_of(qemu_plugin_insn_vaddr(insn));
        qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &all_insns[r], 1);
        if (qemu_plugin_insn_size(insn) != 4) continue;
        uint32_t w;
        memcpy(&w, qemu_plugin_insn_data(insn), 4);
        if ((w & 0x7f) != 0x5b) continue;
        unsigned f3 = (w >> 12) & 7, f7 = f3 == 1 ? (w >> 25) & 0x7f : 0;
        unsigned z = (f3 == 1 && f7 == 0x04 && ((w >> 20) & 0x1f) == 0) ? 1 : 0;
        qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &cnt[r][f3][f7][z], 1);
    }
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    FILE *f = out_path[0] ? fopen(out_path, "w") : NULL;
    if (!f) return;
    for (int r = 0; r < REGIONS; r++)
        fprintf(f, "ALL %s %" PRIu64 "\n", region_name[r], all_insns[r]);
    for (int r = 0; r < REGIONS; r++)
        for (int f3 = 0; f3 < 8; f3++)
            for (int f7 = 0; f7 < 128; f7++)
                for (int z = 0; z < 2; z++)
                    if (cnt[r][f3][f7][z])
                        fprintf(f, "OP %s funct3=%d funct7=0x%02x zimm0=%d %" PRIu64 "\n", region_name[r], f3, f7, z, cnt[r][f3][f7][z]);
    fprintf(f, "END\n");
    fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) return -1;
        *eq = 0;
        if (!strcmp(argv[i], "out")) snprintf(out_path, sizeof(out_path), "%s", eq + 1);
        else if (strcmp(argv[i], "shift")) return -1;
    }
    if (!out_path[0]) return -1;
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
