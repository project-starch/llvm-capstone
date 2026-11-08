"""Read where a stuck hart is, over the board's GDB/OpenOCD session, after a wedge has been declared.

Call `probe(console, log)` from a board driver after it has read the wedge apertures and BEFORE it powers the board
off. It halts the hart, samples the pc `samples` times across resume/halt (a live loop shows different pcs, a truly
stopped hart the same pc every time), then reads ra, sp, mcause, mepc, mtval, mstatus and the disassembly around the
last pc, and ends the GDB session (which resumes the hart). Every value is printed as `GDB <name> 0x...` or
`GDB <name> UNREAD`. It never resets the board: a reset zeroes the CSRs being read.

Adapted from fpga_driver/probe_sqlite_wedge.py (wait for the `$n = 0x..` VALUE, not the prompt: the previous
command's trailing prompt lands after `start`). CLAUDE.md records the debug path returning AXI error-slave junk
(0xca11ab1ebadcab1e) on a wedged core, so a reading counts only after this reader has read a KNOWN stuck pc correctly
(tickcost E2, PREREG.md addendum 2).
"""
import re
import time

VAL = r"\$\d+ = 0x[0-9a-fA-F]+"


def _read(console, expr, timeout=25.0):
    start = len(console.gdb_text)
    console._emit("gdb_input", text=f"p/x {expr}\n")
    try:
        m = console.wait_gdb(VAL, timeout=timeout, search_from=start)
        return m.group(0).split("=")[1].strip()
    except Exception:
        return None


def probe(console, log=print, samples=8, gap_s=1.0, prompt=r"\(gdb\)\s*$"):
    lines = []
    console.gdb_start()
    try:
        console.gdb_cmd("monitor halt", prompt, timeout=30.0)
        for i in range(samples):
            v = _read(console, "$pc")
            lines.append(f"GDB pc sample {i} " + (v or "UNREAD"))
            if i < samples - 1:
                console.gdb_cmd("monitor resume", prompt, timeout=20.0)
                time.sleep(gap_s)
                console.gdb_cmd("monitor halt", prompt, timeout=20.0)
        for expr in ("$ra", "$sp", "$mcause", "$mepc", "$mtval", "$mstatus", "$scause", "$sepc"):
            v = _read(console, expr)
            lines.append(f"GDB {expr[1:]} " + (v or "UNREAD"))
        start = len(console.gdb_text)
        console._emit("gdb_input", text="x/12i $pc-16\n")
        try:
            console.wait_gdb(prompt, timeout=30.0, search_from=start)
        except Exception:
            pass
        for ln in console.gdb_text[start:].splitlines():
            if re.search(r"0x[0-9a-f]+", ln):
                lines.append("GDB insn " + ln.strip())
    finally:
        console.gdb_stop()
    for ln in lines:
        log(ln)
    return lines
