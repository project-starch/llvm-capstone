#!/usr/bin/env python3
"""console-readings.py <driver.log>: the kernel's SBI readings of ONE boot, from the reassembled UART AFTER this run's own
`monitor load_image` (the console replays the previous boot on connect, so a whole-log count would include it).
Prints: the load_image line number, uart chunks before/after it, the count of "remote fence extension is not available"
lines, whether "SBI RFENCE extension detected" appears, the "Runtime SBI Version" line, and the last kernel timestamp.
Exit 2 (NO DATA) when there is no load_image line or no UART after it."""
import re, sys
lines = open(sys.argv[1], errors="replace").read().split("\n")
li = next((i for i, l in enumerate(lines) if "monitor load_image" in l and "gdb_input" in l), None)
if li is None:
    print("NO DATA: no monitor load_image in the log"); sys.exit(2)
def uart(seg):
    return "".join(m for l in seg for m in re.findall(r"\[uart\] '(.*?)'$", l)).replace("\\r\\n", "\n").replace("\\n", "\n")
before, after = uart(lines[:li]), uart(lines[li:])
if not after.strip():
    print("NO DATA: no UART after load_image"); sys.exit(2)
warn = len(re.findall(r"remote fence extension is not available", after))
det = len(re.findall(r"SBI RFENCE extension detected", after))
ver = re.findall(r"Runtime SBI Version\s*:\s*(\S+)", after)
ts = re.findall(r"\[\s*(\d+\.\d+)\]", after)
print(f"load_image at log line {li + 1}; uart chars before {len(before)} after {len(after)}")
print(f"remote-fence warnings {warn}; RFENCE detected {det}; Runtime SBI Version {ver[:1]}; last kernel ts {ts[-1] if ts else None}")
print(f"(before load_image, i.e. the replay: remote-fence warnings {len(re.findall(r'remote fence extension is not available', before))})")
