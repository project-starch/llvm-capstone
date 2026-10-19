#!/usr/bin/env python3
"""kcount.py FILE PATTERN [PATTERN...]: count module/kernel lines in a guest console log, but only if the module's
load banner ('capstone_vm: loaded, diag') is in the SAME file; without it every count is void and the exit is 2.
Prints 'PATTERN=count' per pattern. Exit 0 = counted with the banner present; 2 = no banner / no file (VOID)."""
import sys
from pathlib import Path
if len(sys.argv) < 3: sys.exit('usage: kcount.py FILE PATTERN...')
f = Path(sys.argv[1])
if not f.is_file(): print(f'VOID: no file {f}'); sys.exit(2)
text = f.read_text(errors='replace')
banner = text.count('capstone_vm: loaded, diag')
if not banner: print(f'VOID: no load banner in {f}; every module-line count here is meaningless'); sys.exit(2)
print(f'file={f} banner={banner} ' + ' '.join(f'{p!r}={text.count(p)}' for p in sys.argv[2:]))
