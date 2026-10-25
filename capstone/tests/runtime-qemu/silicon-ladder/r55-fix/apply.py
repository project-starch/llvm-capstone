"""R-55 fix: in cap_env_init, the MMIO carves the FPGA monitor dereferences at their cursor set that cursor
explicitly. Usage: apply.py <lib/sbi/sbi_capstone_dom.c>. Idempotence guard: refuses a file already carrying R-55."""
import sys
p = sys.argv[1]; s = open(p).read()
if "R-55" in s:
    sys.exit("already patched")
uart_old = "    unsigned *cap_uart = split_out_cap(CAPSTONE_UART_BASE, CAPSTONE_UART_LEN, 0);\n"
uart_new = (uart_old +
    "    /* R-55: set, not assumed -- see the mtime carve below. */\n"
    "    cap_uart = __setcursor(cap_uart, CAPSTONE_UART_BASE);\n")
mt_old = ("    unsigned *cap = split_out_cap(SBI_MTIME_ADDR, 8, 0);\n"
          "    mtime = cap;\n"
          "    cap = split_out_cap(SBI_MTIMECMP_ADDR, 8, 0);\n"
          "    mtimecmp = cap;\n")
mt_new = ("    unsigned *cap = split_out_cap(SBI_MTIME_ADDR, 8, 0);\n"
          "#ifdef CAPSTONE_TARGET_FPGA\n"
          "    /* R-55 (2026-10-15): SPLIT gives the piece split_out_cap returns its parent's DECODED start as cursor. Under a\n"
          "       full-scheme bounds encoder (the FP bitstreams' option (b), the Sublet format) that start rounds below the\n"
          "       carve point: mtime came out as [0x2000000, 0x200c000) with its cursor on the CLINT's msip, so\n"
          "       handle_exception's `*mtime` emulated every S-mode rdtime as 0 and Linux hung in start_kernel. A carve\n"
          "       that is dereferenced at its cursor sets the cursor itself. FPGA only: QEMU's bases stay exact, its cursor\n"
          "       is already here, and its monitor stays byte-identical. */\n"
          "    cap = __setcursor(cap, SBI_MTIME_ADDR);\n"
          "#endif\n"
          "    mtime = cap;\n"
          "    cap = split_out_cap(SBI_MTIMECMP_ADDR, 8, 0);\n"
          "#ifdef CAPSTONE_TARGET_FPGA\n"
          "    cap = __setcursor(cap, SBI_MTIMECMP_ADDR);   /* R-55: exact under today's layout; set, not assumed */\n"
          "#endif\n"
          "    mtimecmp = cap;\n")
for old, new in ((uart_old, uart_new), (mt_old, mt_new)):
    if s.count(old) != 1:
        sys.exit(f"anchor not found exactly once: {old[:60]!r}")
    s = s.replace(old, new)
open(p, "w").write(s)
print("R-55 applied:", p)
