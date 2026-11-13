#!/usr/bin/env python3
"""MEASUREMENT ONLY. make-native-stage.py SRC DST NAME=EXE...: a native-route copy of a make-stage.py stage.

SRC is a stage directory (plan.json + stage/), copied with symlinks resolved; it is never modified. In the copy:
  - stage/bin/<arm>/ is replaced by stage/bin/native/<NAME>_base.native (the native-IO SDK build of NAME);
  - run.sh execs each native image directly (dom_binfmt), with no launcher;
  - the workload runs on a private tmpfs at /tmp/nt, because the native grant accepts tmpfs, ext4 and 9p only
    (resource-dom.h dom_disk_inode). The cwd is the grant root and holds .capstone-tmp, for dom_local_tmp;
  - every command is bracketed by `@@NATIVE-CNT <n> begin|end <native_counters>`.
The SPEC markers (@@SPEC-BEGIN/END, HASH, FILE) are kept, so the existing parsers read the stage.
Each rewrite is asserted to match exactly as often as expected."""
import json, re, shutil, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
exes = dict(x.split('=', 1) for x in sys.argv[3:])
assert not dst.exists(), dst
shutil.copytree(src, dst, symlinks=False)
plan = json.loads((dst / 'plan.json').read_text())
arms = plan['arms']
assert len(arms) == 1, arms
arm = arms[0]
st = dst / 'stage'
shutil.rmtree(st / 'bin' / arm)
(st / 'bin' / 'native').mkdir(parents=True)
for name, exe in exes.items():
    shutil.copy2(exe, st / 'bin' / 'native' / f'{name}_base.native')
run = st / 'run.sh'
t = run.read_text()
n = t.count('V=${SPEC_VEXEC:?set SPEC_VEXEC to the batch launcher}\n')
assert n == 1, n
t = t.replace('V=${SPEC_VEXEC:?set SPEC_VEXEC to the batch launcher}\n',
              'cnt() { cat /sys/module/capstone_vm/parameters/native_counters 2>/dev/null || echo absent; }\n'
              'mkdir -p /tmp/nt && { mountpoint -q /tmp/nt || mount -t tmpfs tmpfs /tmp/nt; } || exit 1\n')
k = t.count('cd / && rm -rf /tmp/sr && mkdir -p /tmp/sr && cp -a "$S/in/." /tmp/sr/ && cd /tmp/sr || exit 1')
assert k == 1, k
t = t.replace('cd / && rm -rf /tmp/sr && mkdir -p /tmp/sr && cp -a "$S/in/." /tmp/sr/ && cd /tmp/sr || exit 1',
              'cd / && rm -rf /tmp/nt/sr /tmp/nt/b && mkdir -p /tmp/nt/sr /tmp/nt/b && cp -a "$S/in/." /tmp/nt/sr/ && '
              'cp "$S"/bin/native/* /tmp/nt/b/ && chmod 755 /tmp/nt/b/* && cd /tmp/nt/sr && mkdir -m 0700 .capstone-tmp || exit 1')
t = t.replace('cd / && rm -rf /tmp/sr /tmp/sw', 'cd / && rm -rf /tmp/nt/sr /tmp/nt/b')
# perl's form: a /tmp/sw/perl wrapper names the launcher, and each command runs the launcher with that argv[0]
t, w = re.subn(r'^printf .*> /tmp/sw/perl\n', '', t, flags=re.M)
t = re.sub(r'^chmod 755 /tmp/sw/perl.*\n', '', t, flags=re.M)
t = re.sub(r'^"\$V" --application-argv "\$S/bin/' + re.escape(arm) + r'/([A-Za-z0-9_]+)_base\.' + re.escape(arm) + r'" -- \.\./sw/perl',
           lambda m: f'"$S/bin/{arm}/{m.group(1)}_base.{arm}"'.replace('$S', '$$S'), t, flags=re.M)
t = t.replace('"$$S/bin/', '"$V" "$S/bin/')
# perl's test size: a host perl (natperl) runs test.pl with $^X="../sw/perl", so every sub-test execs ../sw/perl from
# the cwd. Without that wrapper each one fails to start ("can't run '../../sw/perl ...'") and test.out miscompares.
# The cwd is /tmp/nt/sr here, so the wrapper is /tmp/nt/sw/perl, and it execs the native image.
# The image it execs is a copy INSIDE the run directory, /tmp/nt/sr/.perl-native: a sub-test's $^X is the path its perl
# was exec'd by, and t/base/term.t (ok 7) opens $^X. A native task reads only beneath its grant (the run directory), so
# with the image in /tmp/nt/b that open was refused. riscv-base reads its own binary through /tmp/sw/perl; here the
# program under test sits in the run directory instead. The copy is removed before the stage hashes its files, so the
# hashed file set is unchanged.
if '$^X="../sw/perl"' in t:
    assert w == 1 and exes.get('perlbench_r'), (w, list(exes))
    setup = 'cd /tmp/nt/sr && mkdir -m 0700 .capstone-tmp || exit 1\n'
    assert t.count(setup) == 1
    t = t.replace(setup, setup + 'cp /tmp/nt/b/perlbench_r_base.native /tmp/nt/sr/.perl-native && chmod 755 /tmp/nt/sr/.perl-native && '
                  'mkdir -p /tmp/nt/sw && printf \'#!/bin/sh\\nexec /tmp/nt/sr/.perl-native "$@"\\n\' '
                  '> /tmp/nt/sw/perl && chmod 755 /tmp/nt/sw/perl || exit 1\n')
    hb = 'echo "@@SPEC-HASH-BEGIN '
    assert t.count(hb) == 1, t.count(hb)
    t = t.replace(hb, 'rm -f /tmp/nt/sr/.perl-native\n' + hb)
# imagick's form: the validator reads "$S/ref/<file>", which a native task cannot open outside its grant (the guest
# reads "Could not open for read", rc 255). As native01's qt20 stage does, copy ref/ into the cwd and name it there.
nref = len(re.findall(r'"\$S/ref/([^"/]+)"', t))
if nref:
    setup = 'cp -a "$S/in/." /tmp/nt/sr/ && '
    assert t.count(setup) == 1 and plan.get('refs'), (t.count(setup), plan.get('refs'))
    t, r = re.subn(r'"\$S/ref/([^"/]+)"', r'\1', t)   # before the copy is added: "$S/ref/." would match too
    assert r == nref
    t = t.replace(setup, setup + 'cp -a "$S/ref/." /tmp/nt/sr/ && ')
pat = re.compile(r'^"\$V" "\$S/bin/' + re.escape(arm) + r'/([A-Za-z0-9_]+)_base\.' + re.escape(arm) + r'"', re.M)
names = pat.findall(t)
assert names and set(names) <= set(exes), (names, list(exes))
t = pat.sub(lambda m: f'"/tmp/nt/b/{m.group(1)}_base.native"', t)
assert '"$V"' not in t
begins = len(re.findall(r'^echo "@@SPEC-BEGIN \$n ', t, re.M))
t, b = re.subn(r'^(echo "@@SPEC-BEGIN \$n .*)$', r'\1\necho "@@NATIVE-CNT $n begin $(cnt)"', t, flags=re.M)
t, e = re.subn(r'^(echo "@@SPEC-END \$n .*)$', r'echo "@@NATIVE-CNT $n end $(cnt)"\n\1', t, flags=re.M)
assert b == e == begins and b > 0, (b, e, begins)
run.write_text(t)
plan['native'] = {'source_stage': str(src), 'exes': {k: str(v) for k, v in exes.items()}, 'arm_replaced': arm}
(dst / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
print(f'{dst}: {b} commands, exes {sorted(set(names))}')
