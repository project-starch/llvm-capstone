"""Allocator-independent fault recovery verdicts."""

import re

FAULT = r"domain capability fault delivered: cause = (\d+), pc = (0x[0-9a-f]+)"
SITE = r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),"
SENTINEL = 0x0FA017ED


def clean(serial, status):
    return status == 0 and not any(
        text in serial
        for text in ("domain halted by capability fault", "Assertion", "_FAILED__")
    )


def reentry_passed(serial, kind, status):
    faults = re.findall(FAULT, serial)
    sites = re.findall(SITE, serial)
    return (
        clean(serial, status)
        and len(faults) == 1
        and len(sites) == 1
        and faults[0] == ("5" if kind == 0 else "24", sites[0])
        and re.findall(
            r"FAULT_REENTRY (\d+) RESULT (\d+) ENTERED (\d+) AFTER (\d+)", serial
        )
        == [(str(i), str(SENTINEL), "1", "0") for i in range(3)]
        and "__CAPSTONE_FAULT_REENTRY_DONE__" in serial
    )


def fallback_passed(serial, kind, status):
    halted = re.findall(
        r"domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)", serial
    )
    delivered = re.findall(FAULT, serial)
    sites = re.findall(SITE, serial)
    if not (status == 1 and len(halted) == 1 and len(sites) == 1):
        return False
    if "FAULT_REENTRY" in serial or "Assertion" in serial:
        return False
    if kind == 4:
        return not delivered and halted[0] == ("24", sites[0])
    return (
        delivered == [("24", sites[0])]
        and halted[0][0] == "24"
        and halted[0][1] != sites[0]
    )


def isolation_passed(serial, status):
    if not clean(serial, status):
        return False
    events = re.findall(r"RUNTIME_SUPERVISOR (HEALTHY|SIGSEGV)", serial)
    if events != ["HEALTHY", "SIGSEGV", "SIGSEGV", "SIGSEGV", "SIGSEGV", "HEALTHY"]:
        return False
    blocks = serial.split("RUNTIME_SUPERVISOR ")
    if len(blocks) != 7 or blocks[0].count("RUNTIME_CLIENT HEALTHY") != 1:
        return False
    for kind, block in enumerate(blocks[1:5]):
        sites = re.findall(SITE, block)
        faults = re.findall(FAULT, block)
        if len(sites) != 1 or faults != [("5" if kind == 0 else "24", sites[0])]:
            return False
    return (
        serial.count("RUNTIME_CLIENT HEALTHY") == 2
        and blocks[5].count("RUNTIME_CLIENT HEALTHY") == 1
        and "__CAPSTONE_RUNTIME_ISOLATION_DONE__" in blocks[6]
        and len(re.findall(FAULT, serial)) == 4
    )
