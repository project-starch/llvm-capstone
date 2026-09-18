"""Classify only a completed control or a fault at the declared memory access."""

import re


def classify(serial, kind, test, mode, runner_exit):
    faults = re.findall(
        r"domain (?:halted by capability fault|capability fault delivered): cause = (\d+), pc = (0x[0-9a-f]+)",
        serial,
    )
    fault = mode == "sublet" and test not in (0, 7, 10, 14)
    marker = f"Print = Scalar(0x{0xcf17000000000000 | (kind << 8) | test:x})"
    row = {
        "kind": kind,
        "case": test,
        "mode": mode,
        "expected": "fault" if fault else "complete",
        "runner_exit": runner_exit,
    }
    if fault:
        following = serial.split(marker, 1)[-1] if marker in serial else ""
        sites = re.findall(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),", following)
        site = 1 if test in (5, 6) else 0
        cause, pc = faults[-1] if faults else ("0", "0")
        expected = sites[site] if len(sites) > site else None
        row.update(cause=int(cause), pc=pc, expected_pc=expected)
        row["passed"] = (
            marker in serial
            and len(faults) == 1
            and "_FAILED__" not in serial
            and expected is not None
            and int(pc, 16) == int(expected, 16)
            and int(cause) in ((5,) if test == 9 else (24, 25))
            and (
                "domain capability fault delivered" not in serial
                or (
                    "__CAPSTONE_PG_DOMAIN_FAULT__" in serial
                    and "__EXIT_CODE__139" in serial
                )
            )
        )
    else:
        row["passed"] = (
            runner_exit == 0
            and not faults
            and "__CAPSTONE_PG_CONTEXTS_GOOD__" in serial
            and "_FAILED__" not in serial
        )
        if test != 10:
            row["passed"] &= marker in serial
        else:
            stats = re.findall(r"PG_CONTEXT_RESULT 0 POLICY (\d+)", serial)
            row["passed"] &= len(stats) == 1
            if stats:
                row["policy"] = int(stats[0])
    return row
