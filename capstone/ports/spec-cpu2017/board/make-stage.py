#!/usr/bin/env python3
"""Stage one SPEC CPU2017 benchmark at one size for a run without runcpu: the FPGA board, or its QEMU twin.

  make-stage.py --tree SPEC17 --bench 519.lbm_r --size test --arms capstone-spatial,capstone-lifetimes
                --out DIR [--ldecod <static RISC-V ldecod>] [--inband-max BYTES]

The board has no SSH, no 9p, no disk and no network, so runcpu cannot drive it. runcpu still decides every command:
the tree must hold run directories from `runcpu --fake --action=run --size=<size>` for every arm (tools/build.sh,
then the fake run), and this script reads their speccmds.cmd, inputgen.cmd and compare.cmd. It never writes a
command of its own.

DIR/stage gets:
- in/: the run directory's inputs, identical across the arms (checked file by file), less the files no command of
  this size reads (EXCLUDE below; a removed file a command needed fails that command, it is never silent);
- bin/<arm>/: each arm's executables, under the names runcpu gave them;
- ref/: the reference files the target-side validation commands read (imagevalidate's expected images);
- ldecod: 525.x264_r's input generator built for RISC-V Linux, run OUTSIDE a context (tools/build-host-ldecod.sh says
  why the Capstone build cannot generate the input; the kit runs the host build for the same reason);
- run.sh: every arm in turn, each in a fresh copy of in/ under /tmp/sr. The launcher is the batch's own (SPEC_VEXEC:
  the board kit's on the board, the twin kit's on QEMU), never staged here.
DIR/plan.json records the commands, the host-side specdiff commands and the arms.

run.sh prints, per command, @@SPEC-BEGIN and @@SPEC-END with rc, /proc/uptime before and after, the launcher's
CAPSTONE_VM_STATS line and the module's last bench line with the count of bench lines so far (a command that printed
none would otherwise quote the previous one); per arm, sha256 of every file in the run directory; and every output
file named by a command or a specdiff whose size is at most --inband-max (SPEC_INBAND_MAX at run time overrides it:
the twin carries every output, the board's UART only small ones), gzip+base64 between @@SPEC-FILE markers.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Files of a run directory that no command of the size reads. runcpu copies data/all/input into every size.
EXCLUDE = {
    "500.perlbench_r": {"test": ["lib/ham.pl", "lib/spam.pl", "lib/mailcomp.pm"], "train": ["lib/ham.pl", "lib/spam.pl"]},
    "557.xz_r": {"test": ["cld.tar.xz", "input.combined.xz"], "train": ["cld.tar.xz", "cpu2006docs.tar.xz"]},
}
RUNCPU_FILES = re.compile(r"^(speccmds\.cmd|compare\.cmd|inputgen\.cmd|benchmark_run\.(out|err)|compare_run\.(out|err)|"
                          r"input_generation\.(out|err))$")


def parse_cmd(path):
    """specinvoke's command file: (cwd, [dict(input, output, error, argv)]). -E (environment) lines are dropped."""
    cmds, cwd = [], None
    for line in path.read_text().splitlines():
        if line.startswith(("-E ", "-r", "-N ", "-u ")) or not line.strip():
            continue
        if line.startswith("-C "):
            cwd = line[3:].strip()
            continue
        w = shlex.split(line)
        c = dict(input=None, output=None, error=None, keep=False)
        while w and w[0] in ("-i", "-o", "-e", "-k"):
            if w[0] == "-k":
                c["keep"] = True
                w = w[1:]
                continue
            c[{"-i": "input", "-o": "output", "-e": "error"}[w[0]]] = w[1]
            w = w[2:]
        # specinvoke appends the shell redirections it will apply; they repeat -i/-o/-e.
        while len(w) >= 2 and w[-2] in (">", "2>>", "<"):
            w = w[:-2]
        c["argv"] = w
        cmds.append(c)
    return cwd, cmds


def tree_files(root, skip):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if not skip(rel):
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True, help="the SPEC17 directory of a built tree")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--size", required=True, choices=["test", "train", "refrate"])
    ap.add_argument("--arms", required=True, help="labels, in run order")
    ap.add_argument("--ldecod", type=Path, help="static RISC-V Linux ldecod (525.x264_r only)")
    ap.add_argument("--inband-max", type=int, default=65536)
    ap.add_argument("--no-target-validation", action="store_true",
                    help="stage no imagevalidate run and no reference image (the board image of a benchmark whose outputs "
                    "are compared with the twin's by hash; the references alone can exceed the payload cap)")
    ap.add_argument("--strip", type=Path, metavar="LLVM_OBJCOPY",
                    help="strip every staged executable's symbol table with this llvm-objcopy, keeping the .capstone_* "
                    "sections (502.gcc_r's 24 MB image is 8.5 MB of symbols; the code and data are unchanged)")
    ap.add_argument("--native-harness", type=Path, metavar="PERL",
                    help="500.perlbench_r test only: run test.pl under this plain (non-Capstone) static riscv64 perl, its "
                    "$^X set to the Capstone wrapper, so each t/ test runs as its own top-level context. vp2 has one saved-"
                    "context slot, so a Capstone parent waiting on a Capstone child deadlocks (twin, 2026-10-29)")
    ap.add_argument("--keep-lib", type=Path, metavar="LIST",
                    help="stage only the lib/ files this list names, one run-directory path per line (perlbench's lib/ is "
                    "54 MB and its train commands open 70 files of it; the list comes from tracing the native build); "
                    "every other lib/ file joins the excluded ones, so a missing module fails its command, never silently")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    bdir = a.tree / "benchspec/CPU" / a.bench
    if a.out.exists():
        sys.exit(f"make-stage: {a.out} exists")
    runs = {arm: bdir / f"run/run_base_{a.size}_{arm}.0000" for arm in arms}
    for arm, r in runs.items():
        if not (r / "speccmds.cmd").is_file():
            sys.exit(f"make-stage: no fake run for {arm}: {r} (runcpu --fake --action=run --size={a.size})")
    excl = set(EXCLUDE.get(a.bench, {}).get(a.size, []))

    def is_exe(rel, arm):
        return "/" not in rel and rel.endswith("." + arm)

    plan = {}
    for arm, r in runs.items():
        _, cmds = parse_cmd(r / "speccmds.cmd")
        _, gen = parse_cmd(r / "inputgen.cmd") if (r / "inputgen.cmd").exists() else (None, [])
        _, cmp_ = parse_cmd(r / "compare.cmd")
        def norm(cs, arm=arm):
            """The arm's label becomes ARM everywhere (x264's output names carry it), and the program loses its ../run_base_ path."""
            ln = lambda x: x.replace(arm, "ARM") if x else x
            return [dict(c, output=ln(c["output"]), error=ln(c["error"]),
                         argv=[ln(re.sub(rf"^\.\./run_base_{a.size}_{re.escape(arm)}\.0000/", "", x)) for x in c["argv"]]) for c in cs]
        plan[arm] = dict(cmds=norm(cmds), gen=norm(gen), cmp=norm(cmp_))
    first = arms[0]
    for arm in arms[1:]:
        if plan[arm] != plan[first]:
            sys.exit(f"make-stage: {arm}'s commands differ from {first}'s once the label is normalised")
    p = plan[first]
    # Inputs: identical across arms, the executables and runcpu's own files aside.
    files = {arm: tree_files(r, lambda rel, arm=arm: is_exe(rel, arm) or RUNCPU_FILES.match(rel) is not None)
             for arm, r in runs.items()}
    for arm in arms[1:]:
        if files[arm] != files[first]:
            diff = sorted(set(files[arm].items()) ^ set(files[first].items()))[:10]
            sys.exit(f"make-stage: {arm}'s run directory differs from {first}'s: {diff}")
    if a.keep_lib:
        keep = {x.strip() for x in a.keep_lib.read_text().splitlines() if x.strip()}
        absent = sorted(keep - set(files[first]))
        if absent:
            sys.exit(f"make-stage: --keep-lib names files the run directory lacks: {absent[:10]}")
        excl |= {rel for rel in files[first] if rel.startswith("lib/") and rel not in keep}
    missing = sorted(excl - set(files[first]))
    if missing:
        sys.exit(f"make-stage: EXCLUDE names files the run directory lacks: {missing}")
    stage = a.out / "stage"
    (stage / "in").mkdir(parents=True)
    for rel in files[first]:
        if rel in excl:
            continue
        dst = stage / "in" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runs[first] / rel, dst)
    if a.native_harness:
        shutil.copy2(a.native_harness, stage / "natperl")
        (stage / "natperl").chmod(0o755)
    stub = None
    if a.bench == "500.perlbench_r" and (stage / "in/t/op/pack.t").exists():
        stub = json.loads(subprocess.run([sys.executable, str(HERE.parent / "tools/perl-pack-stub.py"), str(stage / "in/t/op/pack.t")],
                                         check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    # Executables, per arm.
    exes = sorted({c["argv"][0] for c in p["cmds"] + p["cmp"] if c["argv"] and c["argv"][0].endswith(".ARM")})
    for arm in arms:
        (stage / "bin" / arm).mkdir(parents=True)
        for e in exes:
            dst = stage / "bin" / arm / e.replace("ARM", arm)
            shutil.copy2(bdir / "exe" / e.replace("ARM", arm), dst)   # a fake run copies none
            if a.strip:
                subprocess.run([str(a.strip), "--strip-all", "--keep-section=.capstone_virtual", "--keep-section=.capstone_domreq",
                                str(dst)], check=True)
    # Target-side validation: commands whose program is one of the benchmark's Capstone executables (imagevalidate).
    # Their reference files live in the SPEC tree's data/; they are staged under ref/ and named there.
    tgt_val, host_val, refs = [], [], {}
    spec_root = str(a.tree.resolve())
    for c in p["cmp"]:
        if c["argv"] and c["argv"][0].endswith(".ARM"):
            if a.no_target_validation:
                continue
            argv = []
            for x in c["argv"]:
                if x.startswith(spec_root + "/"):
                    name = "ref/" + hashlib.sha256(x.encode()).hexdigest()[:8] + "-" + os.path.basename(x)
                    refs[name] = x
                    x = "$S/" + name
                argv.append(x)
            tgt_val.append(dict(c, argv=argv))
        else:
            host_val.append(dict(c, argv=[x.replace(spec_root, "SPEC") for x in c["argv"]]))
    for name, src in refs.items():
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, stage / name)
    if p["gen"]:
        if a.bench != "525.x264_r" or not a.ldecod:
            sys.exit(f"make-stage: {a.bench} has input generation; only 525.x264_r's (ldecod, --ldecod) is supported")
        shutil.copy2(a.ldecod, stage / "ldecod")
        (stage / "ldecod").chmod(0o755)
    # Outputs to carry in-band: every command's -o/-e, and every file a host-side specdiff reads from the run directory.
    outs = []
    for c in p["cmds"] + tgt_val:
        outs += [x for x in (c["output"], c["error"]) if x]
    for c in host_val:
        outs += [x for x in c["argv"] if "/" not in x and not x.startswith("-") and x != c["argv"][0] and not re.match(r"^[0-9.e+-]+$", x)]
    outs = sorted(set(outs))

    q = shlex.quote
    L = ["#!/bin/sh",
         f"# SPEC {a.bench} {a.size}: arms {' '.join(arms)} in that order, each in a fresh copy of in/ under /tmp/sr.",
         "# Generated by capstone/ports/spec-cpu2017/board/make-stage.py from runcpu's own command files.",
         'S=$(cd "$(dirname "$0")" && pwd)', 'V=${SPEC_VEXEC:?set SPEC_VEXEC to the batch launcher}', "export CAPSTONE_VM_STATS=1", "n=0",
         "bench() { dmesg | grep 'capstone_vm: bench' | tail -1 | sed 's/.*capstone_vm: //'; }",
         "nbench() { dmesg | grep -c 'capstone_vm: bench'; }",
         "up() { cut -d' ' -f1 /proc/uptime; }",
         "# the module's cumulative in-context counters (sysfs, 0444): their delta over a command sums every context it ran",
         "sysb() { for f in cycles instret entries; do cat /sys/module/capstone_vm/parameters/bench_$f 2>/dev/null || echo 0; done | tr '\\n' ' '; }",
         f"echo \"@@SPEC-STAGE {a.bench} {a.size} arms={','.join(arms)} commands={len(p['cmds'])} validations={len(tgt_val)}\""]

    def run_line(tag, arm, i, c, wrap):
        argv = [x.replace("ARM", arm) for x in c["argv"]]
        prog = argv[0]
        args = " ".join('"$S/' + x[3:] + '"' if x.startswith("$S/") else q(x) for x in argv[1:])
        c = dict(c, output=c["output"] and c["output"].replace("ARM", arm), error=c["error"] and c["error"].replace("ARM", arm))
        if wrap and tag == "run" and a.bench == "500.perlbench_r" and a.native_harness and argv[-1] == "test.pl":
            hargs = " ".join(q(x) for x in argv[1:-1])
            cmd = f'"$S/natperl" {hargs} -e \'$^X="../sw/perl"; do "./test.pl"; die $@ if $@\''
        elif wrap and tag == "run" and a.bench == "500.perlbench_r":
            # perl's t/TEST starts each test as '../'.$^X through sh, so $^X must be a RELATIVE path that sh can
            # execute: the board kernel has no binfmt_misc to start an image directly. The launcher's
            # --application-argv sets the application's argv[0] (perl's $^X) apart from the image, and ../sw/perl,
            # outside the hashed run directory, is a native script that starts the same image with the same argv[0].
            cmd = f'"$V" --application-argv "$S/bin/{arm}/{prog}" -- ../sw/perl {args}'
        elif wrap:
            cmd = f'"$V" "$S/bin/{arm}/{prog}" {args}'
        else:
            cmd = f'"$S/ldecod" {args}'
        redir = (f" < {q(c['input'])}" if c["input"] else "") + (f" > {q(c['output'])}" if c["output"] else "") + \
                (f" 2>> {q(c['error'])}" if c["error"] else "")
        err = q(c["error"]) if c["error"] else "/dev/null"
        return [f'echo "@@SPEC-BEGIN $n {tag} {arm} {i}"', "s0=$(sysb)", "u0=$(up)",
                f"{cmd}{redir}", "rc=$?", "u1=$(up)", "s1=$(sysb)",
                "sys=$(echo $s0 $s1 | awk '{printf \"%.0f,%.0f,%.0f\", $4-$1, $5-$2, $6-$3}')",
                f'echo "@@SPEC-END $n {tag} {arm} {i} rc=$rc up0=$u0 up1=$u1 sys=$sys nbench=$(nbench) '
                f'$(grep -a CAPSTONE_VM_STATS {err} 2>/dev/null | tail -1) :: $(bench)"', "n=$((n+1))"]

    for arm in arms:
        L += [f'echo "@@SPEC-ARM-BEGIN {arm}"', "cd / && rm -rf /tmp/sr && mkdir -p /tmp/sr && cp -a \"$S/in/.\" /tmp/sr/ && cd /tmp/sr || exit 1"]
        if a.bench == "500.perlbench_r":
            img = f"$S/bin/{arm}/" + p["cmds"][0]["argv"][0].replace("ARM", arm)
            L += ["rm -rf /tmp/sw && mkdir -p /tmp/sw || exit 1",
                  f"printf '#!/bin/sh\\nexec \"%s\" --application-argv \"%s\" -- \"$0\" \"$@\"\\n' \"$V\" \"{img}\" > /tmp/sw/perl",
                  "chmod 755 /tmp/sw/perl || exit 1"]
        for i, c in enumerate(p["gen"]):
            L += run_line("gen", arm, i, dict(c, argv=["ldecod"] + c["argv"][1:]), wrap=False)
        for i, c in enumerate(p["cmds"]):
            L += run_line("run", arm, i, c, wrap=True)
        for i, c in enumerate(tgt_val):
            L += run_line("val", arm, i, c, wrap=True)
        L += [f'echo "@@SPEC-HASH-BEGIN {arm}"', "find . -type f | sort | while read -r f; do sha256sum \"$f\"; done",
              f'echo "@@SPEC-HASH-END {arm}"']
        for f in (x.replace("ARM", arm) for x in outs):
            L += [f'if [ -f {q(f)} ] && [ $(wc -c < {q(f)}) -le ${{SPEC_INBAND_MAX:-{a.inband_max}}} ]; then echo "@@SPEC-FILE {arm} {f}"; '
                  f'gzip -9c {q(f)} | base64; echo "@@SPEC-FILE-END"; fi']
        L += [f'echo "@@SPEC-ARM-END {arm}"']
    L += ["cd / && rm -rf /tmp/sr /tmp/sw", f'echo "@@SPEC-DONE {a.bench} {a.size}"', ""]
    (stage / "run.sh").write_text("\n".join(L))
    (stage / "run.sh").chmod(0o755)
    meta = dict(bench=a.bench, size=a.size, arms=arms, commands=p["cmds"], generation=p["gen"], target_validation=tgt_val,
                host_validation=host_val, outputs=outs, excluded=sorted(excl), perl_pack_stub=stub,
                inputs={k: v for k, v in files[first].items() if k not in excl},
                executables={arm: {e.replace("ARM", arm): hashlib.sha256((stage / "bin" / arm / e.replace("ARM", arm)).read_bytes()).hexdigest()
                                   for e in exes} for arm in arms},
                ldecod=hashlib.sha256(a.ldecod.read_bytes()).hexdigest() if p["gen"] else None, stripped=bool(a.strip),
                refs={k: os.path.relpath(v, spec_root) for k, v in refs.items()})
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    raw = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    gz = int(subprocess.run(f"tar -C {q(str(stage))} -cf - . | gzip -9 | wc -c", shell=True, check=True,
                            capture_output=True, text=True).stdout)
    print(f"stage {a.bench} {a.size}: {len(p['cmds'])} commands, {len(tgt_val)} target validations, {len(host_val)} host "
          f"validations, {len(outs)} outputs; {raw} bytes raw, {gz} gzip -9")


if __name__ == "__main__":
    main()
