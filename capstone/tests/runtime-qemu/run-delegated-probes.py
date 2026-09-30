#!/usr/bin/env python3
"""The runtime-qemu probes that run as delegated applications (ABI v2).

Each probe is an ordinary C program, built with an application SDK's
capstone-cc and run in a provisioned guest through the host CLI
(python3 -m capstone_vm --state STATE run), one application per variant. A
variant is judged on what the launcher's task reports -- its exit status or
signal and the fault record -- and on its two streams, never on a missing
line: a variant with no result is a FAIL.

  run-delegated-probes.py --sdk SDK --work DIR --build-only
  run-delegated-probes.py --sdk SDK --work DIR --state STATE [--only NAME...]
                          [--arith-expect trap|cheri] [--report FILE]
  run-delegated-probes.py --self-test

The images are staged in a fresh directory inside the VM's share and removed
afterwards. The variants run in the order below: every one expected to return
first, then the ones expected to fault, and last untagged-cap-arith's case 2,
which aborts a QEMU without the scc fix.

What the probes check, and what replaced their HostCall v0 controls: see
README.md in this directory. The controls that linked an older runtime file
(a pinned hostcall.c, tls.c or host) cannot be expressed against an SDK, whose
runtime archive is linked whole; --self-test shows that every check fails on
the symptom its old control produced.

movc runs in any guest and is judged by the switch that guest's QEMU was
started with (CAPSTONE_MOVC_NULL_SCALAR, recorded in the state directory);
run it once in a guest with the switch off and once in one with it on. It is
built twice: movc-rule with the compiler's live-source copy rule (C-32's fix)
and movc-keep with +movc-keeps-integer-source, a plain movc for every copy.
With the switch on, keep must lose C-32's two values (the positive control)
and rule must not. The build refuses (exit 2) a toolchain that lacks either,
because a keep built without the feature is a duplicate of rule and a rule
built without the pass reports the compiler as broken.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
HOST = HERE.parents[1] / "runtime" / "host"

LARGE_SIZE = 64 * 1024
BIG_LINE = 5000


def large_pattern(n=LARGE_SIZE):
    return bytes((i * 7 + 3) & 255 for i in range(n))


def big_stdout_expected():
    line = bytes(ord("a") + i % 26 for i in range(BIG_LINE))
    return line + b"\nBIG-STDOUT-END\n"


# ---- the checks: (record, stdout, stderr, context) -> list of problems ------
# record is capstone-job's {"kind", "value"[, "fault"]}; an empty list passes.

def exits(record, value):
    got = (record.get("kind"), record.get("value"))
    problems = [] if got == ("exit", value) else [f"got {got[0]} {got[1]}, want exit {value}"]
    if "fault" in record:
        problems.append(f"unexpected fault record: {record['fault']}")
    return problems


def faults(record, cause=None):
    if (record.get("kind"), record.get("value")) != ("signal", 11) or "fault" not in record:
        return [f"got {record.get('kind')} {record.get('value')}, want SIGSEGV with a fault record"]
    fields = dict(w.split("=", 1) for w in record["fault"].split() if "=" in w)
    if cause is not None and fields.get("cause") != str(cause):
        return [f"fault cause {fields.get('cause')}, want {cause}"]
    return []


def lines(text, pattern):
    return [l for l in text.splitlines() if re.match(pattern, l)]


def check_init_fini(record, out, err, ctx):
    got = lines(out, r"(CTOR|MAIN|DTOR) ")
    want = ctx["init-fini-native"].splitlines()
    problems = exits(record, 0)
    if not want:
        problems.append("no native reference")
    if got != want:
        problems.append(f"order {got}, native {want}")
    return problems


def check_exit(value):
    def check(record, out, err, ctx):
        problems = exits(record, value)
        for line in ("EXIT-TEST before exit", "EXIT-TEST atexit handler ran"):
            if line not in out.splitlines():
                problems.append(f"missing '{line}'")
        return problems
    return check


def check_return_flush(record, out, err, ctx):
    got = lines(out, r"RETURN-FLUSH (line [123]|atexit handler ran)$")
    problems = exits(record, 5)
    if len(got) != 4:
        problems.append(f"{len(got)} of 4 lines")
    return problems


UNSERVED_LINE = "capstone-domain: UNSERVED syscalls: 214x2"


def check_unserved(record, out, err, ctx):
    problems = exits(record, 0)
    if not lines(out, r"UNSERVED-TEST two brk calls"):
        problems.append("missing the program's line")
    reports = lines(err, r"capstone-domain: UNSERVED syscalls:")
    if reports != [UNSERVED_LINE]:
        problems.append(f"report {reports}, want ['{UNSERVED_LINE}']")
    return problems


def check_done(marker, passes=None, pass_pattern=None):
    def check(record, out, err, ctx):
        problems = exits(record, 0)
        if f"{marker} failures=0" not in out.splitlines():
            problems.append(f"no '{marker} failures=0'")
        failed = lines(out, r"(FAIL |\S+ FAIL )")
        if failed:
            problems.append(f"failed: {failed}")
        if passes is not None and len(lines(out, pass_pattern)) != passes:
            problems.append(f"{len(lines(out, pass_pattern))} of {passes} checks passed")
        return problems
    return check


def check_large_read(record, out, err, ctx):
    problems = check_done("LARGE-READ-DONE", 5, r"LARGE-READ PASS ")(record, out, err, ctx)
    written = ctx.get("large-written")
    if written != large_pattern():
        problems.append("written.bin on the share is not the pattern "
                        f"({'missing' if written is None else len(written)} bytes)")
    return problems


def check_big_stdout(record, out, err, ctx):
    problems = exits(record, 0)
    got = ctx.get("big-stdout-file")
    want = big_stdout_expected()
    if got != want:
        longest = max((len(l) for l in (got or b"").split(b"\n")), default=0)
        problems.append(f"the 9p file holds {'nothing' if got is None else len(got)} bytes, "
                        f"longest line {longest}; want {len(want)} bytes")
    return problems


def check_mmap_control(record, out, err, ctx):
    problems = exits(record, 0)
    if "MMAP-SHM-CONTROL rc=-1 errno=38" not in out.splitlines():
        problems.append(f"the syscall layer served mmap: {lines(out, r'MMAP-SHM-CONTROL')}")
    if not lines(err, r"capstone-domain: UNSERVED syscalls: 222$"):
        problems.append("no unserved report for mmap (222)")
    return problems


def check_tls_overrun(record, out, err, ctx):
    problems = faults(record)
    if not lines(out, r"TLS-TEST overrun: writing zeroed"):
        problems.append("did not reach the write")
    if lines(out, r"TLS-TEST overrun: NOT stopped"):
        problems.append("the write past the thread-local was not stopped")
    return problems


def check_movc(variant):
    def check(record, out, err, ctx):
        problems = exits(record, 0)
        probe = re.search(r"^MOVC-PROBE b=(\d+) c=(\d+)$", out, re.M)
        c32 = re.search(rf"^MOVC-C32 {variant} got=(0x[0-9a-f]+)", out, re.M)
        iconv = re.search(rf"^MOVC-ICONV {variant} n=(\d+)$", out, re.M)
        if not probe or not c32 or not iconv:
            return problems + [f"no MOVC-PROBE, MOVC-C32 {variant} or MOVC-ICONV {variant} line"]
        on = ctx.get("movc-switch") == "1"
        b, c, got, n = probe.group(1), probe.group(2), c32.group(1), int(iconv.group(1))
        if on and (b, c) == ("5", "5"):
            problems.append("switch on changed nothing: this QEMU has no CAPSTONE_MOVC_NULL_SCALAR")
        elif (b, c) != ("5", "0" if on else "5"):
            problems.append(f"probe b={b} c={c}, want b=5 c={'0' if on else '5'}")
        if not on or variant == "rule":
            # QEMU's default keeps every value; under the RTL's rule the
            # live-source copy rule must keep them too.
            if (got, n) != ("0x5000", 3):
                problems.append(f"c32 got={got} iconv n={n}, want 0x5000 and 3"
                                + (": the live-source copy rule missed a copy" if on else ""))
        elif got != "0x1" or n >= 3:
            problems.append(f"c32 got={got} iconv n={n}, want 0x1 and n<3: "
                            "the positive control did not see C-32")
        notices = ctx.get("movc-notices")
        if notices is not None and bool(notices) != on:
            problems.append(f"{notices} switch notices in qemu.log with the switch {'on' if on else 'off'}")
        return problems
    return check


INTCAP_INTEGER = ["arith", "cmp", "fromint", "int", "shift", "switch"]
INTCAP_POINTER = ["atomic", "ptr", "ptrarith"]


def check_intcap(variant):
    def check(record, out, err, ctx):
        ok = sorted(set(re.findall(rf"^INTCAP {variant} ([a-z]+) ok$", out, re.M)))
        bad = lines(out, rf"INTCAP {variant} [a-z]+ BAD")
        end = lines(out, rf"INTCAP {variant} END ")
        if variant == "intcap":
            problems = exits(record, 0)
            if ok != sorted(INTCAP_INTEGER + INTCAP_POINTER):
                problems.append(f"ok for {ok}, want all nine cases")
            if end != ["INTCAP intcap END bad=0"]:
                problems.append(f"end {end}, want ['INTCAP intcap END bad=0']")
        else:
            # The control: an unsigned long drops the tag, so the integer
            # cases pass and the first dereference faults.
            problems = faults(record)
            if ok != INTCAP_INTEGER:
                problems.append(f"ok for {ok}, want the six integer cases")
            if f"INTCAP {variant} integers done bad=0" not in out.splitlines():
                problems.append("did not finish the integer cases")
            if end:
                problems.append(f"reached the end: {end}; the pointer cases cannot fail")
        if bad:
            problems.append(f"bad: {bad}")
        return problems
    return check


def check_arith(case):
    def check(record, out, err, ctx):
        if case == 0:
            problems = exits(record, 0)
            if "ARITH-CASE 0 cinc=1 scc=1" not in out.splitlines():
                problems.append(f"control: {lines(out, r'ARITH-CASE')}")
            return problems
        if ctx.get("arith-expect", "trap") == "cheri":
            want = "0x5008" if case == 1 else "0x6000"
            problems = exits(record, 0)
            if f"ARITH-CASE {case} result={want}" not in out.splitlines():
                problems.append(f"want result={want}: {lines(out, r'ARITH-CASE')}")
            return problems
        problems = faults(record, 24)
        if lines(out, rf"ARITH-CASE {case} "):
            problems.append(f"printed a result: {lines(out, r'ARITH-CASE')}")
        return problems
    return check


# ---- the variants, in run order ---------------------------------------------
# name: sources (relative to this directory), compile flags, arguments, guest
# environment, check. ARGS entries may name "{large}" and "{written}".
VARIANTS = {
    "init-fini": (["init-fini/ctors.c"], ["-O1"], [], ["INIT_FINI_ENV=before-main"], check_init_fini),
    "exit-default": (["exit-hook/exit_test.c"], ["-O1"], [], [], check_exit(7)),
    "exit-hook": (["exit-hook/exit_test.c"], ["-O1", "-DWITH_HOOK"], [], [], check_exit(42)),
    "return-flush": (["return-flush/return_flush.c"], ["-O1"], [], [], check_return_flush),
    "unserved-report": (["unserved-report/closefd1.c"], ["-O1"], [], [], check_unserved),
    "large-read": (["large-io/large_read.c"], ["-O1"], ["{large}", "{written}"], [], check_large_read),
    "big-stdout": (["large-io/big_stdout.c"], ["-O1"], [], [], check_big_stdout),
    "mmap-shm": (["mmap-shm/mmap_shm.c"], ["-O2"], [], [], check_done("MMAP-SHM-DONE")),
    "mmap-shm-control": (["mmap-shm/mmap_shm.c"], ["-O2", "-DRAW_SYSCALL_CONTROL"], [], [],
                         check_mmap_control),
    "tls-O0": (["thread-local/tls_test.c", "thread-local/tls_other.c"], ["-O0"], [], [],
               check_done("TLS-TEST-DONE", 9, r"TLS-TEST PASS ")),
    "tls-O2": (["thread-local/tls_test.c", "thread-local/tls_other.c"], ["-O2"], [], [],
               check_done("TLS-TEST-DONE", 9, r"TLS-TEST PASS ")),
    "cap-atomics-O0": (["capability-atomics/cap_atomics.c"], ["-O0"], [], [],
                       check_done("CAP-ATOMICS-DONE")),
    "cap-atomics-O2": (["capability-atomics/cap_atomics.c"], ["-O2"], [], [],
                       check_done("CAP-ATOMICS-DONE")),
    "subword-O0": (["subword-atomics/subword_atomics.c"], ["-O0"], [], [],
                   check_done("SUBWORD-ATOMICS-DONE")),
    "subword-O2": (["subword-atomics/subword_atomics.c"], ["-O2"], [], [],
                   check_done("SUBWORD-ATOMICS-DONE")),
    "movc-rule": (["movc-null-scalar/movc_test.c"], ["-O2", '-DVARIANT="rule"'], [], [],
                  check_movc("rule")),
    "movc-keep": (["movc-null-scalar/movc_test.c"],
                  ["-O2", '-DVARIANT="keep"', "-Xclang", "-target-feature",
                   "-Xclang", "+movc-keeps-integer-source"], [], [], check_movc("keep")),
    "intcap": (["intcap/intcap_test.c"], ["-O2", '-DVARIANT="intcap"', "-DUSE_INTCAP"], [], [],
               check_intcap("intcap")),
    "arith-0": (["untagged-cap-arith/arith_test.c"], ["-O2", "-DCASE=0"], [], [], check_arith(0)),
    # expected to fault: after everything that returns
    "tls-overrun": (["thread-local/tls_test.c", "thread-local/tls_other.c"], ["-O2", "-DOVERRUN"],
                    [], [], check_tls_overrun),
    "intcap-uptr": (["intcap/intcap_test.c"], ["-O2", '-DVARIANT="uptr"'], [], [],
                    check_intcap("uptr")),
    "arith-1": (["untagged-cap-arith/arith_test.c"], ["-O2", "-DCASE=1"], [], [], check_arith(1)),
    # a QEMU without the scc fix aborts here: last
    "arith-2": (["untagged-cap-arith/arith_test.c"], ["-O2", "-DCASE=2"], [], [], check_arith(2)),
}


def tool(sdk_config, name):
    path = Path(sdk_config["cc"]).parent / name
    if not path.exists():
        raise SystemExit(f"run-delegated-probes: no {name} next to {sdk_config['cc']}")
    return str(path)


def check_movc_toolchain(config, work):
    """Exit 2 unless llc has the live-source copy pass and recognises
    +movc-keeps-integer-source. The exit status is the condition: a
    missing llc or a crash must not pass as "present", and an unknown
    -mattr is only a warning, so the feature also needs the text."""
    llc = tool(config, "llc")
    module = work / "movc-identity.ll"
    module.write_text("define void @x() {\n  ret void\n}\n")
    for extra, missing in ((["-stop-after=capstone-live-source-copy"], "not registered"),
                           (["-mattr=+movc-keeps-integer-source"], "not a recognized feature")):
        result = subprocess.run([llc, "-mtriple=capstone64", *extra, "-o", os.devnull, str(module)],
                                capture_output=True, text=True)
        if result.returncode or missing in result.stderr:
            print(f"run-delegated-probes: COULD NOT CHECK movc -- {llc} {extra[0]} exited "
                  f"{result.returncode}:\n{result.stderr.strip()}\n"
                  "This toolchain lacks the live-source copy pass or +movc-keeps-integer-source; "
                  "rebuild it.", file=sys.stderr)
            raise SystemExit(2)
    module.unlink()


def build(sdk, work, only):
    config = json.loads((sdk / "sdk.json").read_text())
    if config.get("application_abi") != 2:
        raise SystemExit(f"run-delegated-probes: {sdk} is not an ABI v2 SDK")
    work.mkdir(parents=True, exist_ok=True)
    if any(name.startswith("movc-") and (not only or name in only) for name in VARIANTS):
        check_movc_toolchain(config, work)
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, (sources, flags, _, _, _) in VARIANTS.items():
        if only and name not in only:
            continue
        out = images / f"{name}.dom"
        out.unlink(missing_ok=True)
        cmd = [str(sdk / "capstone-cc"), "-std=c11", *flags, *(str(HERE / s) for s in sources),
               "-o", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode or not out.exists():
            raise SystemExit(f"run-delegated-probes: {name} did not build (rc {result.returncode}):\n"
                             f"{' '.join(cmd)}\n{result.stderr[-2000:]}")
        built[name] = out
    readelf = tool(config, "llvm-readelf")
    if "tls-O2" in built:
        # The page-aligned thread-local makes PT_TLS 4096-aligned, so the
        # align4096 check exercises the runtime's placement, not luck.
        text = subprocess.run([readelf, "-lW", str(built["tls-O2"])], capture_output=True,
                              text=True, check=True).stdout
        align = [l.split()[-1] for l in text.splitlines() if l.split()[:1] == ["TLS"]]
        if align != ["0x1000"]:
            raise SystemExit(f"run-delegated-probes: tls-O2 PT_TLS alignment {align}, want ['0x1000']")
    if "init-fini" in built:
        # A constructor section outside link.ld's array markers would never run.
        text = subprocess.run([readelf, "-SW", str(built["init-fini"])], capture_output=True,
                              text=True, check=True).stdout
        orphans = sorted(set(re.findall(r"\.(?:init_array|fini_array)\.\S+|\.(?:ctors|dtors)\S*", text)))
        if orphans:
            raise SystemExit(f"run-delegated-probes: constructor sections outside the markers: {orphans}")
        # The reference order: the same file, natively.
        native = work / "ctors-native"
        subprocess.run(["cc", "-O1", "-o", str(native), str(HERE / "init-fini/ctors.c")], check=True)
        ref = subprocess.run([str(native)], env={"INIT_FINI_ENV": "before-main"}, capture_output=True,
                             text=True, check=True).stdout
        if not lines(ref, r"MAIN "):
            raise SystemExit("run-delegated-probes: the native reference printed no MAIN line")
        (work / "init-fini.native.txt").write_text(ref)
    if "large-read" in built:
        (work / "large.bin").write_bytes(large_pattern())
    return config, built


def run_variant(cli, env, name, image, arguments, environment, timeout):
    with tempfile.TemporaryDirectory(prefix=f"probe-{name}-") as directory:
        result_file = Path(directory) / "result.json"
        command = [*cli, "run", "--cwd", "/tmp", "--result", str(result_file)]
        for item in environment:
            command += ["-e", item]
        command += [image, *arguments]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=env) as proc:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The CLI owns cancellation; killing it directly would orphan the job.
                proc.send_signal(signal.SIGTERM)
                out, err = proc.communicate(timeout=30)
                err += b"\n(timed out)"
        record = json.loads(result_file.read_text()) if result_file.exists() else {}
    return record, out.decode(errors="replace"), err.decode(errors="replace")


def run(args, config, built):
    state = args.state.resolve()
    vm = json.loads((state / "config.json").read_text())
    share = Path(vm["share"])
    env = dict(os.environ, PYTHONPATH=str(HOST))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(state)]
    ctx = {"arith-expect": args.arith_expect,
           "movc-switch": vm.get("identity", {}).get("environment", {}).get("CAPSTONE_MOVC_NULL_SCALAR", "0")}
    native = args.work / "init-fini.native.txt"
    ctx["init-fini-native"] = native.read_text() if native.exists() else ""
    stage = Path(tempfile.mkdtemp(prefix="runtime-probes-", dir=share))
    guest = "/mnt/host/" + str(stage.relative_to(share))
    results, failed = {}, 0
    try:
        for name in VARIANTS:
            if name not in built:
                continue
            shutil.copyfile(built[name], stage / f"{name}.dom")
        if "large-read" in built:
            shutil.copyfile(args.work / "large.bin", stage / "large.bin")
        qemu_log = state / "qemu.log"
        for name, (_, _, arguments, environment, check) in VARIANTS.items():
            if name not in built:
                continue
            image = f"{guest}/{name}.dom"
            arguments = [a.format(large=f"{guest}/large.bin", written=f"{guest}/written.bin")
                         for a in arguments]
            started = time.monotonic()
            if name == "big-stdout":
                # The launcher's stdout on a 9p file, as the old host's was.
                target = stage / "big-stdout.out"
                result = subprocess.run([*cli, "exec", "sh", "-c",
                                         f"capstone-exec -- {image} > {guest}/big-stdout.out"],
                                        env=env, capture_output=True, text=True, timeout=args.timeout)
                record = {"kind": "exit", "value": result.returncode}
                out, err = result.stdout, result.stderr
                ctx["big-stdout-file"] = target.read_bytes() if target.exists() else None
            else:
                record, out, err = run_variant(cli, env, name, image, arguments, environment,
                                               args.timeout)
            if name == "large-read":
                written = stage / "written.bin"
                ctx["large-written"] = written.read_bytes() if written.exists() else None
            if name.startswith("movc-") and qemu_log.exists():
                # The switch prints its notice once per QEMU, at the first
                # nulled source, which may be the monitor's while booting.
                text = qemu_log.read_bytes().decode(errors="replace")
                ctx["movc-notices"] = text.count("MOVC-NULL-SCALAR first non-zero source nulled")
            problems = check(record, out, err, ctx) if record else ["no result from the guest"]
            verdict = "FAIL" if problems else "PASS"
            failed += bool(problems)
            results[name] = {"verdict": verdict, "record": record, "problems": problems,
                             "seconds": round(time.monotonic() - started, 2),
                             "stdout_tail": out[-800:], "stderr_tail": err[-800:]}
            print(f"{verdict:<5} {name:<17} {record.get('kind')} {record.get('value')}"
                  + (f"  {'; '.join(problems)}" if problems else ""), flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"{len(results) - failed} of {len(results)} variants pass")
    if args.report:
        args.report.write_text(json.dumps({
            "sdk": str(args.sdk), "cc": config["cc"], "state": str(state),
            "vm_files": {k: v.get("sha256") for k, v in vm.get("identity", {}).get("files", {}).items()},
            "qemu_environment": vm.get("identity", {}).get("environment", {}),
            "arith_expect": args.arith_expect, "results": results}, indent=1) + "\n")
    return 1 if failed or not results else 0


def self_test():
    """Every check against a passing outcome and against the symptom its
    HostCall v0 control produced; a check that cannot fail is not a check."""
    ok_exit = lambda v: {"kind": "exit", "value": v}
    segv = {"kind": "signal", "value": 11, "fault": "capstone-exec: domain fault cause=24 pc=0x1"}
    native = "CTOR p 1\nCTOR a 2 env=before-main\nCTOR b 3\nMAIN 4\nDTOR b 5\nDTOR a 6\nDTOR p 7\n"
    ctx = {"init-fini-native": native, "large-written": large_pattern(),
           "big-stdout-file": big_stdout_expected(), "arith-expect": "trap"}
    exit_out = "EXIT-TEST before exit\nEXIT-TEST atexit handler ran\n"
    rf = "RETURN-FLUSH line 1\nRETURN-FLUSH line 2\nRETURN-FLUSH line 3\nRETURN-FLUSH atexit handler ran\n"
    movc = lambda c, v, got, n: (f"MOVC-PROBE b=5 c={c}\nMOVC-C32 {v} got={got} want=0x5000 called=1\n"
                                 f"MOVC-ICONV {v} n={n}\n")
    INT, ALL = INTCAP_INTEGER, INTCAP_INTEGER + INTCAP_POINTER
    intcap = lambda v, cases: ("".join(f"INTCAP {v} {c} ok\n" for c in cases if c in INT)
                               + f"INTCAP {v} integers done bad=0\n"
                               + "".join(f"INTCAP {v} {c} ok\n" for c in cases if c not in INT))
    tls = "".join(f"TLS-TEST PASS c{i} got=0 want=0\n" for i in range(9)) + "TLS-TEST-DONE failures=0\n"
    cases = [
        # (name, check, record, stdout, stderr, context update, must pass)
        ("init-fini", check_init_fini, ok_exit(0), native, "", {}, True),
        ("init-fini: no constructor ran, exit() faulted", check_init_fini, segv, "MAIN 1\n", "", {}, False),
        ("init-fini: no native reference", check_init_fini, ok_exit(0), native, "",
         {"init-fini-native": ""}, False),
        ("exit-hook", check_exit(42), ok_exit(42), exit_out, "", {}, True),
        ("exit-hook: the hook did not run", check_exit(42), ok_exit(7), exit_out, "", {}, False),
        ("exit-default: halted at the image base", check_exit(7), segv, "", "", {}, False),
        ("exit-default: musl's atexit lost the tag", check_exit(7), segv,
         "EXIT-TEST before exit\n", "", {}, False),
        ("return-flush", check_return_flush, ok_exit(5), rf, "", {}, True),
        ("return-flush: returned straight to the caller", check_return_flush, ok_exit(5),
         "RETURN-FLUSH line 1\n", "", {}, False),
        ("unserved-report", check_unserved, ok_exit(0), "UNSERVED-TEST two brk calls (-1 -1), closing fd 1\n",
         UNSERVED_LINE + "\n", {}, True),
        ("unserved-report: the report was lost", check_unserved, ok_exit(0),
         "UNSERVED-TEST two brk calls (-1 -1), closing fd 1\n", "", {}, False),
        ("large-read", check_large_read, ok_exit(0),
         "".join(f"LARGE-READ PASS {n} x\n" for n in range(5)) + "LARGE-READ-DONE failures=0\n", "", {}, True),
        ("large-read: the read failed", check_large_read, ok_exit(1),
         "LARGE-READ PASS open x\nLARGE-READ FAIL read-whole n=-1 errno=14\nLARGE-READ-DONE failures=1\n",
         "", {}, False),
        ("large-read: written file differs", check_large_read, ok_exit(0),
         "".join(f"LARGE-READ PASS {n} x\n" for n in range(5)) + "LARGE-READ-DONE failures=0\n", "",
         {"large-written": b"x"}, False),
        ("big-stdout", check_big_stdout, ok_exit(0), "", "", {}, True),
        ("big-stdout: the long line never arrived", check_big_stdout, ok_exit(0), "", "",
         {"big-stdout-file": b"BIG-STDOUT-END\n"}, False),
        ("mmap-shm", check_done("MMAP-SHM-DONE"), ok_exit(0), "MMAP-SHM PASS mmap x\nMMAP-SHM-DONE failures=0\n",
         "", {}, True),
        ("mmap-shm: no mmap", check_done("MMAP-SHM-DONE"), ok_exit(1),
         "MMAP-SHM FAIL mmap p=MAP_FAILED errno=38\nMMAP-SHM-DONE failures=1\n", "", {}, False),
        ("mmap-shm-control", check_mmap_control, ok_exit(0), "MMAP-SHM-CONTROL rc=-1 errno=38\n",
         "capstone-domain: UNSERVED syscalls: 222\n", {}, True),
        ("mmap-shm-control: the syscall layer mapped", check_mmap_control, ok_exit(1),
         "MMAP-SHM-CONTROL rc=4096 errno=0\n", "", {}, False),
        ("tls-O2", check_done("TLS-TEST-DONE", 9, r"TLS-TEST PASS "), ok_exit(0), tls, "", {}, True),
        ("tls-O2: the template was not copied", check_done("TLS-TEST-DONE", 9, r"TLS-TEST PASS "),
         ok_exit(1), tls.replace("PASS c0", "FAIL c0").replace("failures=0", "failures=1"), "", {}, False),
        ("tls-O2: stopped part-way", check_done("TLS-TEST-DONE", 9, r"TLS-TEST PASS "), segv,
         "TLS-TEST PASS c0 got=0 want=0\n", "", {}, False),
        ("tls-overrun", check_tls_overrun, segv, "TLS-TEST overrun: writing zeroed[100]\n", "", {}, True),
        ("tls-overrun: not stopped", check_tls_overrun, ok_exit(1),
         "TLS-TEST overrun: writing zeroed[100]\nTLS-TEST overrun: NOT stopped\n", "", {}, False),
        ("cap-atomics: untagged pointer faulted", check_done("CAP-ATOMICS-DONE"), segv,
         "ok   control: long store and fetch_add\n", "", {}, False),
        ("movc-rule off", check_movc("rule"), ok_exit(0), movc("5", "rule", "0x5000", 3), "",
         {"movc-switch": "0", "movc-notices": 0}, True),
        ("movc-keep off", check_movc("keep"), ok_exit(0), movc("5", "keep", "0x5000", 3), "",
         {"movc-switch": "0", "movc-notices": 0}, True),
        ("movc-rule on", check_movc("rule"), ok_exit(0), movc("0", "rule", "0x5000", 3), "",
         {"movc-switch": "1", "movc-notices": 1}, True),
        ("movc-keep on", check_movc("keep"), ok_exit(0), movc("0", "keep", "0x1", 1), "",
         {"movc-switch": "1", "movc-notices": 1}, True),
        ("movc-rule on: the copy rule missed a copy", check_movc("rule"), ok_exit(0),
         movc("0", "rule", "0x1", 1), "", {"movc-switch": "1", "movc-notices": 1}, False),
        ("movc-keep on: the control did not see C-32", check_movc("keep"), ok_exit(0),
         movc("0", "keep", "0x5000", 3), "", {"movc-switch": "1", "movc-notices": 1}, False),
        ("movc-keep off: lost a value with QEMU's default", check_movc("keep"), ok_exit(0),
         movc("5", "keep", "0x1", 1), "", {"movc-switch": "0", "movc-notices": 0}, False),
        ("movc-rule on: a QEMU without the switch", check_movc("rule"), ok_exit(0),
         movc("5", "rule", "0x5000", 3), "", {"movc-switch": "1", "movc-notices": 0}, False),
        ("movc-rule off: a notice with the switch off", check_movc("rule"), ok_exit(0),
         movc("5", "rule", "0x5000", 3), "", {"movc-switch": "0", "movc-notices": 1}, False),
        ("movc-rule: the other variant's lines", check_movc("rule"), ok_exit(0),
         movc("5", "keep", "0x5000", 3), "", {"movc-switch": "0", "movc-notices": 0}, False),
        ("intcap", check_intcap("intcap"), ok_exit(0), intcap("intcap", ALL) + "INTCAP intcap END bad=0\n",
         "", {}, True),
        ("intcap: the pointer lost its tag", check_intcap("intcap"), segv, intcap("intcap", INT), "", {},
         False),
        ("intcap: a case printed BAD", check_intcap("intcap"), ok_exit(1),
         intcap("intcap", ALL).replace("shift ok", "shift BAD got=0x0 want=0x10")
         + "INTCAP intcap END bad=1\n", "", {}, False),
        ("intcap-uptr", check_intcap("uptr"), segv, intcap("uptr", INT), "", {}, True),
        ("intcap-uptr: the pointer cases passed", check_intcap("uptr"), ok_exit(0),
         intcap("uptr", ALL) + "INTCAP uptr END bad=0\n", "", {}, False),
        ("intcap-uptr: faulted among the integers", check_intcap("uptr"), segv,
         "INTCAP uptr int ok\n", "", {}, False),
        ("arith-0", check_arith(0), ok_exit(0), "ARITH-CASE 0 cinc=1 scc=1\n", "", {}, True),
        ("arith-1 trap", check_arith(1), segv, "", "", {}, True),
        ("arith-1 trap: printed a result", check_arith(1), ok_exit(0), "ARITH-CASE 1 result=0x5008\n", "",
         {}, False),
        ("arith-2 cheri", check_arith(2), ok_exit(0), "ARITH-CASE 2 result=0x6000\n", "",
         {"arith-expect": "cheri"}, True),
        ("arith-2 cheri: trapped", check_arith(2), segv, "", "", {"arith-expect": "cheri"}, False),
    ]
    bad = 0
    for name, check, record, out, err, update, must_pass in cases:
        problems = check(record, out, err, {**ctx, **update})
        good = (not problems) == must_pass
        bad += not good
        print(f"{'ok  ' if good else 'BAD '} {name}: {'passes' if not problems else '; '.join(problems)}")
    print(f"self-test: {len(cases) - bad} of {len(cases)} as expected")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdk", type=Path, help="application SDK build directory (capstone-cc, sdk.json)")
    parser.add_argument("--work", type=Path, help="where the images and the references are built")
    parser.add_argument("--state", type=Path, help="capstone_vm state directory of a running guest")
    parser.add_argument("--only", nargs="*", default=[], choices=sorted(VARIANTS), metavar="NAME")
    parser.add_argument("--arith-expect", choices=["trap", "cheri"], default="trap",
                        help="untagged-cap-arith: today's ISA (cause 24) or CHERI's rule")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.sdk or not args.work:
        parser.error("--sdk and --work are required")
    if not args.build_only and not args.state:
        parser.error("--state is required unless --build-only")
    args.work = args.work.resolve()
    config, built = build(args.sdk.resolve(), args.work, set(args.only))
    for name, image in built.items():
        print(f"built {name:<17} {image}")
    if args.build_only:
        return 0
    return run(args, config, built)


if __name__ == "__main__":
    sys.exit(main())
