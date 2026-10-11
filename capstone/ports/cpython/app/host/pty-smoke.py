"""os.openpty in a delegated domain: the pair opens, bytes cross it, the slave is
a terminal with a name, and the terminal requests behind it are answered."""
import os
import sys
import termios

master, slave = os.openpty()
print("opened", master, slave, flush=True)
name = os.ttyname(slave)
print("slave", name, "isatty", os.isatty(slave), flush=True)
os.write(master, b"ping\n")
data = os.read(slave, 100)
print("read from slave", data, flush=True)
attrs = termios.tcgetattr(slave)
attrs[3] &= ~termios.ECHO
termios.tcsetattr(slave, termios.TCSANOW, attrs)
try:
    pgrp = os.tcgetpgrp(slave)
    print("tcgetpgrp slave", pgrp, flush=True)
except OSError as e:
    print("tcgetpgrp slave errno", e.errno, "(not our controlling terminal)", flush=True)
ok = name.startswith("/dev/pts/") and os.isatty(slave) and data == b"ping\n"
os.close(slave)
os.close(master)
print("PTY", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
