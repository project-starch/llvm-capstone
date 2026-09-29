#!/usr/bin/env python3
"""Gate a four-arm application study on recorded build and workload parity.

Each arm's compile_argv and driver_compile_argv are the actual argv used for
the application and workload driver, recorded by the build wrapper. This
checks explicit -D/-U SQLite options and optimization; it does not infer
undocumented compiler defaults.
"""

import argparse
import json
import re
import sys
from pathlib import Path


ARMS = {
    "capstone": "capstone",
    "capstone-sublet": "capstone",
    "poisoncap-spatial": "cheribsd",
    "poisoncap-temporal": "cheribsd",
}
SHARED = ("upstream_sha256", "driver_sha256", "workload_sha256",
          "runtime_sqlite_config")
PAIRED = ("compiler_sha256", "target", "vfs", "base_port_patch_sha256",
          "port_source_sha256")
# SQLITE_OS_OTHER selects the Capstone VFS. It cannot be equal on CheriBSD.
PLATFORM_ONLY_DEFINES = {"SQLITE_OS_OTHER"}
DRIVER_ONLY_DEFINES = {"SQLITE_STUDY_CAPSTONE_MEMSYS5", "SQLITE_STUDY_MEMSYS5"}
HASH_FIELDS = ("upstream_sha256", "driver_sha256", "workload_sha256",
               "compiler_sha256", "base_port_patch_sha256", "port_source_sha256",
               "protection_patch_sha256", "binary_sha256")


def compile_settings(argv):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("compile_argv must be a nonempty argv array")
    definitions = {}
    optimization = None
    for i, arg in enumerate(argv):
        if re.fullmatch(r"-O(?:[0-3sz]|fast)", arg):
            optimization = arg
        if arg in ("-D", "-U"):
            if i + 1 == len(argv):
                raise ValueError(f"{arg} has no operand")
            option = arg + argv[i + 1]
        else:
            option = arg
        match = re.fullmatch(r"-([DU])(SQLITE_[A-Za-z0-9_]+)(?:=(.*))?", option)
        if match:
            definitions[match[2]] = (match[3] or "1") if match[1] == "D" else None
    if optimization is None:
        raise ValueError("SQLite translation unit has no explicit optimization level")
    return optimization, definitions


def audit(data):
    issues = []
    if data.get("schema") != 1:
        issues.append("expected build manifest schema 1")
    arms = data.get("arms", {})
    if set(arms) != set(ARMS):
        return issues + [f"expected arms {sorted(ARMS)}, got {sorted(arms)}"]
    settings = {}
    for name, platform in ARMS.items():
        arm = arms[name]
        for field in (*SHARED, *PAIRED, "binary_sha256", "port_source_sha256",
                      "protection_patch_sha256", "compile_argv",
                      "driver_compile_argv", "link_argv"):
            if not arm.get(field):
                issues.append(f"{name}: missing {field}")
        for field in HASH_FIELDS:
            value = arm.get(field)
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                issues.append(f"{name}: {field} is not a SHA-256 digest")
        if arm.get("platform") != platform:
            issues.append(f"{name}: wrong platform")
        try:
            settings[name] = (compile_settings(arm["compile_argv"]),
                              compile_settings(arm["driver_compile_argv"]))
        except (KeyError, ValueError) as exc:
            issues.append(f"{name}: {exc}")
    if issues:
        return issues
    first = arms["capstone"]
    for field in SHARED:
        for name in ARMS:
            if arms[name][field] != first[field]:
                issues.append(f"{field}: {name} differs from capstone")
    for left, right in (("capstone", "capstone-sublet"),
                        ("poisoncap-spatial", "poisoncap-temporal")):
        for field in PAIRED:
            if arms[left][field] != arms[right][field]:
                issues.append(f"{field}: {left} differs from {right}")
    for unit, excluded in ((0, PLATFORM_ONLY_DEFINES),
                           (1, PLATFORM_ONLY_DEFINES | DRIVER_ONLY_DEFINES)):
        label = "application" if unit == 0 else "driver"
        optimizations = {name: value[unit][0] for name, value in settings.items()}
        if len(set(optimizations.values())) != 1:
            issues.append(f"{label} optimization differs: {optimizations}")
        keys = set().union(*(value[unit][1] for value in settings.values())) - excluded
        for key in sorted(keys):
            values = {name: settings[name][unit][1].get(key, "<default>") for name in ARMS}
            if len(set(values.values())) != 1:
                issues.append(f"{label} {key} differs: {values}")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        issues = audit(json.loads(args.manifest.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        return 1
    print("PASS: shared workload/source/configuration/SQLite flags; paired platform builds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
