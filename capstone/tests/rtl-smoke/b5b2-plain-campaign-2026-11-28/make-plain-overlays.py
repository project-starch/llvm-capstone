#!/usr/bin/env python3
"""make-plain-overlays.py OUT: the Plain arm's campaign overlays (runtime lane, 2026-11-28; PREREG.md).

For each campaign group (g1b, g2i, g3, g4mk, x264b) the overlay is the campaign's own CF/CS overlay
($CAPSTONE_HDD/trapk/fpga/spec/SP-B5TL1-<g>/ovl, byte-identical to its B5TL1 source) with exactly these changes:
- every stage directory B5TL1-<bench> is renamed PLM-<bench> (stages.order and stages.bound follow), and its
  bin/native/<bench>_base.native is replaced by the Plain-matched native named in
  $CAPSTONE_HDD/c4/plain/plain-stages/MANIFEST.json, checked by sha256;
- /smoke.dom (a capability domain, which Plain cannot run) is removed and /plain-smoke (tests/line-b-plain, sha256
  42e1f6d1b289fa93...) added; /init's one control line runs it instead and prints "control plain-smoke rc=".
Every other file is asserted byte-identical to the campaign overlay. Prints a manifest per group and writes
OUT/<group>/ plus OUT/stage-root/PLM/<bench> (links to the grader's B5TL1 references; perlbench's to g4mk's).
Refuses an existing OUT/<group>."""
import hashlib, json, os, re, shutil, sys

H = os.environ['CAPSTONE_HDD']
SRC = f'{H}/trapk/fpga/spec'
MAN = f'{H}/c4/plain/plain-stages/MANIFEST.json'
SMOKE = f'{H}/plaincamp/x/plain-smoke'
SMOKE_SHA = '42e1f6d1b289fa93'
REF = {'default': f'{H}/c4/b5b2spec/board-stages/B5TL1', '500.perlbench_r': f'{H}/b5b2linux/g4mkroot/B5TL1'}
GROUPS = ['g1b', 'g2i', 'g3', 'g4mk', 'x264b']
CONTROL_OLD = 'mkdir -p /tmp/run && cd /tmp/run && cp /smoke.dom . && ./smoke.dom; echo "TRAPTEST: control smoke.dom rc=$?"; cd /\n'
CONTROL_NEW = 'mkdir -p /tmp/run && cd /tmp/run && cp /plain-smoke . && ./plain-smoke; echo "TRAPTEST: control plain-smoke rc=$?"; cd /\n'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def tree(d):
    return {os.path.relpath(os.path.join(r, f), d): sha(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs}


def main():
    out = sys.argv[1]
    natives = {}
    for s in json.load(open(MAN))['stages']:
        if s['arm'] == 'plain-matched':
            im = s['images'][0]
            natives[s['bench']] = (im['name'], im['sha256'], im['source'].replace('<hdd>', H))
    if not sha(SMOKE).startswith(SMOKE_SHA):
        sys.exit(f'plain-smoke {sha(SMOKE)[:16]} is not {SMOKE_SHA}')
    for g in GROUPS:
        src, dst = f'{SRC}/SP-B5TL1-{g}/ovl', f'{out}/{g}'
        if os.path.exists(dst):
            sys.exit(f'{dst} exists')
        shutil.copytree(src, dst, symlinks=True)
        os.remove(f'{dst}/smoke.dom')
        shutil.copy2(SMOKE, f'{dst}/plain-smoke')
        t = open(f'{dst}/init').read()
        if t.count(CONTROL_OLD) != 1:
            sys.exit(f'{g}: control line not found once in init')
        open(f'{dst}/init', 'w').write(t.replace(CONTROL_OLD, CONTROL_NEW))
        replaced = []
        for st in sorted(os.listdir(f'{dst}/stages')):
            m = re.fullmatch(r'B5TL1-(\d{3}\.\w+)', st)
            if not m:
                sys.exit(f'{g}: unexpected stage {st}')
            bench = m[1]
            name, want, srcpath = natives[bench]
            if sha(srcpath) != want:
                sys.exit(f'{bench}: {srcpath} is not the MANIFEST image {want[:16]}')
            new = f'{dst}/stages/PLM-{bench}'
            os.rename(f'{dst}/stages/{st}', new)
            target = f'{new}/stage/bin/native/{name}'
            if not os.path.exists(target):
                sys.exit(f'{bench}: no {name} in the campaign stage')
            os.remove(target)
            shutil.copy2(srcpath, target)
            os.chmod(target, 0o755)
            replaced.append(f'stages/PLM-{bench}/stage/bin/native/{name}')
            ref = REF.get(bench, REF['default'])
            os.makedirs(f'{out}/stage-root/PLM', exist_ok=True)
            link = f'{out}/stage-root/PLM/{bench}'
            if not os.path.exists(link):
                os.symlink(f'{ref}/{bench}', link)
        for f in ('stages.order', 'stages.bound'):
            p = f'{dst}/{f}'
            if os.path.exists(p):
                txt = open(f'{src}/{f}').read()
                with open(p, 'w') as fo:
                    fo.write(txt.replace('B5TL1-', 'PLM-'))
                # the renamed list must keep every line of the campaign's, in order, with only the arm prefix changed
                if open(p).read().replace('PLM-', 'B5TL1-') != txt or (txt.strip() and not open(p).read().strip()):
                    sys.exit(f'{g}: {f} not carried over')
        # every other file byte-identical to the campaign overlay
        a, b = tree(src), tree(dst)
        ren = lambda k: k.replace('stages/B5TL1-', 'stages/PLM-')
        a2 = {ren(k): v for k, v in a.items()}
        expect = set(replaced) | {'init', 'stages.order', 'stages.bound', 'plain-smoke', 'smoke.dom'}
        diff = sorted(k for k in set(a2) | set(b) if a2.get(k) != b.get(k) and k not in expect)
        if diff:
            sys.exit(f'{g}: unexpected differences {diff[:5]}')
        print(f'{g}: {len(b)} files; replaced {len(replaced)} natives; init control -> plain-smoke; '
              f'stages {sorted(os.listdir(f"{dst}/stages"))}')
        with open(f'{dst}.manifest', 'w') as fo:
            for k in sorted(b):
                fo.write(f'{b[k]}  {k}\n')


main()
