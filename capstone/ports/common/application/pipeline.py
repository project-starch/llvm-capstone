#!/usr/bin/env python3
"""What build-virtual.sh resolved, as a record beside the build it ran.

Written for every build and for --dry-run, so the pipeline a run used can be
read back without rereading the script that chose it. The remaining
inconsistency it exists to make visible is shared_sdk: six of the eight ports
build an SDK inside their own recipe and two take one built for them.
"""
import json
import os

environment = os.environ
shared = environment["CAPSTONE_PIPELINE_SHARED_SDK"] == "1"
out = environment["CAPSTONE_PIPELINE_OUT"]
selected = environment["CAPSTONE_PIPELINE_SELECTED"]
print(json.dumps({
    "application": environment["CAPSTONE_PIPELINE_APP"],
    "lifetimes": int(environment["CAPSTONE_PIPELINE_LIFETIMES"]),
    "port_variable": selected or None,
    "shared_sdk": shared,
    "relinked_by_build_py": environment["CAPSTONE_PIPELINE_RELINK"] == "1",
    "output": {"source": f"{out}/source", "image": f"{out}/image",
               "sdk": f"{out}/sdk" if shared else None},
    "dry_run": environment["CAPSTONE_PIPELINE_DRY"] == "1",
}, indent=2))
