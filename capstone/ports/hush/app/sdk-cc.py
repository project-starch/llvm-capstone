#!/usr/bin/env python3
"""BusyBox's final-link groups are already provided by the common SDK driver."""
import os
import sys

cc = os.environ["CAPSTONE_HUSH_SDK_CC"]
args = [arg for arg in sys.argv[1:]
        if arg not in ("-Wl,--start-group", "-Wl,--end-group", "-Wl,--verbose")]
os.execv(cc, [cc, *args])
