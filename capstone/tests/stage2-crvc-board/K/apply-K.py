#!/usr/bin/env python3
"""apply-K.py LINUX_TREE: the stage-2 kernel patch K on a copy of trapk's tree (plan 2026-11-25, "The mode's home"):
1. the ELF loader binds table_pa | crvc, crvc = e_flags & EF_CAPSTONE_CRVC (0x100), and refuses a flagged image when
   the boot probe found no CRVC mode;
2. the boot probe (core_initcall, before /init): write 1 to srevroot, read it back, write 0;
3. select_root compares (old & ~1UL), so a CRVC process does not rewrite srevroot per capability window.
Plus one twin-only mutant, SUBLET_MUTANT_CRVC_LEAK: switch_mm writes 1 (the mode bit and no table) for an ordinary mm
whose task is named crvcleak*. Every edit asserts exactly one match. Run once per tree."""
import sys
T = sys.argv[1].rstrip('/') + '/arch/riscv'


def edit(path, old, new):
    p = f'{T}/{path}'
    s = open(p).read()
    assert s.count(old) == 1, (path, s.count(old), old[:60])
    open(p, 'w').write(s.replace(old, new))


# 1. the loader
edit('kernel/sublet/binfmt.h',
     'struct dom_image {\n    Elf64_Phdr text, data;\n    u64 entry, bytes, stack;\n};',
     'struct dom_image {\n    Elf64_Phdr text, data;\n    u64 entry, bytes, stack;\n'
     '    unsigned long crvc;  /* stage 2: 1 when e_flags carries EF_CAPSTONE_CRVC */\n};\n'
     '#define DOM_EF_CAPSTONE_CRVC 0x100')
edit('kernel/sublet/binfmt.h',
     '    im->entry = h.e_entry - im->text.p_vaddr;\n',
     '    im->entry = h.e_entry - im->text.p_vaddr;\n'
     '    im->crvc = (h.e_flags & DOM_EF_CAPSTONE_CRVC) ? 1 : 0;\n')
edit('kernel/sublet/binfmt.h',
     'static int dom_enter(struct sublet_context *c, struct linux_binprm *bprm, unsigned long entry,\n'
     '\t\t     unsigned long **slots, unsigned long sp, unsigned long top)\n',
     'static int dom_enter(struct sublet_context *c, struct linux_binprm *bprm, unsigned long entry,\n'
     '\t\t     unsigned long **slots, unsigned long sp, unsigned long top, unsigned long crvc)\n')
edit('kernel/sublet/binfmt.h',
     '\trc = riscv_capstone_bind(c, c->table_pa);\n\tif (rc)\n\t\treturn rc;\n',
     '\t/* Stage 2: bit 0 of the root is the CRVC mode; table_pa is page-aligned. */\n'
     '\trc = riscv_capstone_bind(c, c->table_pa | crvc);\n\tif (rc)\n\t\treturn rc;\n'
     '\tif (crvc)\n'
     '\t\tpr_info("sublet: %s[%d] binds in the CRVC mode\\n", current->comm, task_pid_nr(current));\n')
edit('kernel/sublet/binfmt.h',
     '\trc = dom_enter(c, bprm, base + (unsigned long)im.entry, slots, sp, top);\n',
     '\trc = dom_enter(c, bprm, base + (unsigned long)im.entry, slots, sp, top, im.crvc);\n')
edit('kernel/sublet/binfmt.h',
     '\trc = dom_parse(bprm->file, &im);\n\tif (rc)\n\t\treturn rc;\n',
     '\trc = dom_parse(bprm->file, &im);\n\tif (rc)\n\t\treturn rc;\n'
     '\t/* A hart without the mode would run c.ldcsp as c.fldsp: refuse the image. */\n'
     '\tif (im.crvc && !sublet_crvc_hw) {\n'
     '\t\tpr_info("sublet: %s[%d] CRVC image refused: this hart has no CRVC mode\\n",\n'
     '\t\t\tcurrent->comm, task_pid_nr(current));\n'
     '\t\treturn -ENOEXEC;\n\t}\n')

# 2. the boot probe
edit('include/asm/capstone_task.h',
     'int riscv_capstone_bind(void *context, unsigned long root);\n#endif\n',
     'int riscv_capstone_bind(void *context, unsigned long root);\n'
     '/* Stage 2: the boot probe found the CRVC mode (bit 0 of an srevroot write reads back). */\n'
     'extern bool sublet_crvc_hw;\n#endif\n')
edit('kernel/capstone_task.c',
     '\tmutex_unlock(&current->signal->cred_guard_mutex);\n\treturn error;\n}\n#endif\n',
     '\tmutex_unlock(&current->signal->cred_guard_mutex);\n\treturn error;\n}\n\n'
     '/* Stage 2, before any protected process exists: write 1 and read it back. A hart with the mode\n'
     ' * reads bit 0; srevroot_q stays 0 throughout, so nothing decodes in the mode. Then write 0. */\n'
     'bool sublet_crvc_hw __ro_after_init;\n'
     'static int __init sublet_crvc_probe(void)\n{\n'
     '\tunsigned long got;\n\n'
     '\tcsr_write(0x5c1, 1);\n\tgot = csr_read(0x5c1);\n\tcsr_write(0x5c1, 0);\n'
     '\tsublet_crvc_hw = got & 1;\n'
     '\tpr_info("sublet: CRVC probe read 0x%lx after writing 1: %s\\n", got,\n'
     '\t\tsublet_crvc_hw ? "the CRVC mode is present" : "no CRVC mode, flagged images are refused");\n'
     '\treturn 0;\n}\ncore_initcall(sublet_crvc_probe);\n#endif\n')

# 3. select_root
edit('mm/sublet-table.c',
     '    if (old != c->table_pa)\n        csr_write(0x5c1, c->table_pa);\n',
     '    if ((old & ~1UL) != c->table_pa)  /* stage 2: bit 0 is the CRVC mode */\n'
     '        csr_write(0x5c1, c->table_pa);\n')

# the twin-only leak mutant
edit('mm/context.c',
     '#ifdef SUBLET_MUTANT_LEAK_ROOT\n\tif (next->context.capstone_root)\n#endif\n'
     '\tcsr_write(0x5c1, next->context.capstone_root);\n',
     '#ifdef SUBLET_MUTANT_CRVC_LEAK\n'
     '\t/* Twin only: the mode bit with no table, for one named ordinary process. */\n'
     '\tif (!next->context.capstone_root && task && !strncmp(task->comm, "crvcleak", 8)) {\n'
     '\t\tstatic int said;\n\n\t\tcsr_write(0x5c1, 1);\n'
     '\t\tif (!said++)\n\t\t\tprintk_deferred(KERN_EMERG "sublet: mutant CRVC_LEAK wrote 1 for %s[%d]\\n",\n'
     '\t\t\t\t\ttask->comm, task_pid_nr(task));\n'
     '\t} else\n#endif\n'
     '#ifdef SUBLET_MUTANT_LEAK_ROOT\n\tif (next->context.capstone_root)\n#endif\n'
     '\tcsr_write(0x5c1, next->context.capstone_root);\n')
print('apply-K: 10 edits applied to', T)
