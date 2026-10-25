#!/usr/bin/env python3
"""collect.py PROGRAM RUNS_DIR NATIVE_DIR OUT_JSON: every repetition's raw counters for one program,
and the comparison the README tables are read from (printed). RUNS_DIR holds mm-run.py's
<program>-<arm>-<rep>/ directories (rep.json, files/); NATIVE_DIR mm-native.sh's rep1..rep3."""
import hashlib, json, re, sys
from pathlib import Path

prog, runs, native, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ARMS = ('stock', 'protected')


def mc_stats(text):
    sections, cur = {}, None
    for line in text.splitlines():
        if line.startswith('== '):
            cur = sections.setdefault(line[3:].strip(), {})
        elif line.startswith('STAT ') and cur is not None:
            _, k, v = line.split(' ', 2)
            cur[k] = v.strip()
    return sections


def proc_fields(text):
    f = {}
    for item in text.split(';'):
        if ':' in item:
            k, v = item.split(':', 1)
            f[k.strip()] = v.strip()
    return f


def counts(err):
    return {tag: dict(kv.split('=') for kv in rest.split()) for tag, rest in
            re.findall(r'^MM_COUNT (\S+) (.*)$', err, re.M)}


def vmstats(text):
    m = re.search(r'^CAPSTONE_VM_STATS (.*)$', text, re.M)
    return {k: int(v) for k, v in (kv.split('=') for kv in m.group(1).split())} if m else None


def delta(rep, a, b):
    s = rep['vexec_stats']
    if a not in s or b not in s:
        return None
    return {k: s[b][k] - s[a][k] for k in s[a] if isinstance(s[a].get(k), int) and isinstance(s[b].get(k), int)}


record = {'program': prog, 'repetitions': [], 'native': []}
for arm in ARMS:
    for rep in (1, 2, 3):
        d = runs / f'{prog}-{arm}-{rep}'
        r = json.loads((d / 'rep.json').read_text())
        files = d / 'files'
        row = {'arm': arm, 'rep': rep, 'completed': r['completed'], 'boot_wall_seconds': r['boot_wall_seconds'],
               'markers': r['markers'], 'vexec_stats': r['vexec_stats'], 'proc': r['proc'],
               'staged_sha256': r['staged_sha256'], 'gate_sha256': r['gate_sha256'],
               'platform_sha256': r['platform_sha256'], 'files_sha256': {k: v['sha256'] for k, v in r['files'].items()}}
        if prog == 'memcached':
            row['vexec_delta'] = {'run': delta(r, 'before', 'after'), 'to_end_of_workload': delta(r, 'before', 'during'),
                                  'stats_commands': delta(r, 'during', 'during2')}
            st = (files / 'stats.txt').read_text() if (files / 'stats.txt').is_file() else ''
            row['memcached_stats'] = mc_stats(st)
            row['transcript_norm_sha256'] = r['files'].get('h-transcript.norm', {}).get('sha256')
            row['job'] = (files / 'job.json').read_text().strip() if (files / 'job.json').is_file() else None
            row['server_err_bytes'] = r['files'].get('server.err', {}).get('bytes')
            row['launcher_exit_stats'] = vmstats((files / 'server.err').read_text()) if (files / 'server.err').is_file() else None
        else:
            row['vexec_delta'] = {lab: delta(r, f'{lab}:before', f'{lab}:after') for lab in ('ao', 'smoke')}
            row['object_counts'] = {lab: counts((files / f'{lab}.err').read_text()) if (files / f'{lab}.err').is_file() else {}
                                    for lab in ('ao', 'smoke')}
            row['stdout_sha256'] = {lab: r['files'].get(f'{lab}.out', {}).get('sha256') for lab in ('ao', 'smoke')}
            row['launcher_exit_stats'] = {lab: vmstats((files / f'{lab}.err').read_text()) if (files / f'{lab}.err').is_file() else None
                                          for lab in ('ao', 'smoke')}
        record['repetitions'].append(row)
for rep in (1, 2, 3):
    d = native / f'rep{rep}'
    row = {'rep': rep, 'markers': (d / 'markers.txt').read_text().split()}
    row['proc'] = {line.split(' ', 1)[0]: proc_fields(line.split(' ', 1)[1] if ' ' in line else '')
                   for line in (d / 'proc.txt').read_text().splitlines()}
    if prog == 'memcached':
        row['memcached_stats'] = mc_stats((d / 'stats.txt').read_text())
        row['transcript_norm_sha256'] = hashlib.sha256((d / 'h/transcript.norm').read_bytes()).hexdigest()
    else:
        row['object_counts'] = {lab: counts((d / f'{lab}.err').read_text()) for lab in ('ao', 'smoke')}
        row['stdout_sha256'] = {lab: hashlib.sha256((d / f'{lab}.out').read_bytes()).hexdigest() for lab in ('ao', 'smoke')}
    record['native'].append(row)
out.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')


def spread(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return 'not measured'
    return str(vs[0]) if len(set(map(str, vs))) == 1 else f'{min(vs)}..{max(vs)} ({", ".join(map(str, vs))})'


reps = {arm: [r for r in record['repetitions'] if r['arm'] == arm] for arm in ARMS}
print('completed:', {arm: [r['completed'] for r in reps[arm]] for arm in ARMS})
print('boot wall s:', {arm: [r['boot_wall_seconds'] for r in reps[arm]] for arm in ARMS})
kb = lambda s: int(s.split()[0]) if s else None
if prog == 'memcached':
    print('transcript sha:', {arm: [r['transcript_norm_sha256'] for r in reps[arm]] for arm in ARMS},
          'native', [r['transcript_norm_sha256'] for r in record['native']])
    print('markers:', {arm: [r['markers'] for r in reps[arm]] for arm in ARMS})
    print('job:', {arm: [r['job'] for r in reps[arm]] for arm in ARMS})
    for sec, keys in (('stats', ('curr_items', 'total_items', 'cmd_get', 'cmd_set', 'get_hits', 'get_misses', 'bytes',
                                 'evictions', 'curr_connections', 'total_connections', 'threads', 'pointer_size',
                                 'rejected_connections', 'slab_reassign_running', 'slabs_moved', 'lru_crawler_running')),
                      ('stats slabs', ('active_slabs', 'total_malloced'))):
        for k in keys:
            print(f'{sec:11} {k:24}', ' | '.join(f'{arm}: {spread([r["memcached_stats"].get(sec, {}).get(k) for r in reps[arm]])}' for arm in ARMS),
                  '| native:', spread([r['memcached_stats'].get(sec, {}).get(k) for r in record['native']]))
    classes = sorted({int(k.split(':')[0]) for r in record['repetitions'] + record['native']
                      for k in r['memcached_stats'].get('stats slabs', {}) if ':' in k})
    print('per class: chunk_size chunks_per_page total_pages total_chunks used_chunks free_chunks')
    for c in classes:
        cells = []
        for label, rows in (('stock', reps['stock']), ('protected', reps['protected']), ('native', record['native'])):
            cells.append(label + ' ' + ' '.join(spread([r['memcached_stats'].get('stats slabs', {}).get(f'{c}:{f}') for r in rows])
                                                 for f in ('chunk_size', 'chunks_per_page', 'total_pages', 'total_chunks', 'used_chunks', 'free_chunks')))
        print(f'class {c:2}:', ' || '.join(cells))
    print('items per class (stats items number):')
    for c in classes:
        print(f'class {c:2}:', ' | '.join(f'{label}: {spread([r["memcached_stats"].get("stats items", {}).get(f"items:{c}:number") for r in rows])}'
                                     for label, rows in (('stock', reps['stock']), ('protected', reps['protected']), ('native', record['native']))))
    for tag in ('after_workload', 'after_stats'):
        for f in ('VmHWM', 'VmRSS', 'VmPeak'):
            vals = {arm: [kb(next((v.get(f) for k, v in r['proc'].items() if k.startswith(tag)), None)) for r in reps[arm]] for arm in ARMS}
            print(f'proc {tag} {f} kB:', ' | '.join(f'{arm}: {spread(vals[arm])}' for arm in ARMS),
                  '| native:', spread([kb(r['proc'].get(tag, {}).get(f)) for r in record['native']]))
    keys = sorted({k for r in record['repetitions'] for k in (r['launcher_exit_stats'] or {})})
    for k in keys:
        print(f'CAPSTONE_VM_STATS {k:14}', ' | '.join(f'{arm}: {spread([(r["launcher_exit_stats"] or {}).get(k) for r in reps[arm]])}' for arm in ARMS))
    for which in ('run', 'to_end_of_workload', 'stats_commands'):
        keys = sorted({k for r in record['repetitions'] for k in (r['vexec_delta'][which] or {})})
        for k in keys:
            print(f'vexec delta {which:18} {k:22}', ' | '.join(f'{arm}: {spread([(r["vexec_delta"][which] or {}).get(k) for r in reps[arm]])}' for arm in ARMS))
    for tag in ('before', 'during', 'after'):
        keys = sorted({k for r in record['repetitions'] for k in r['vexec_stats'].get(tag, {})})
        for k in keys:
            print(f'vexec {tag:6} {k:22}', ' | '.join(f'{arm}: {spread([r["vexec_stats"].get(tag, {}).get(k) for r in reps[arm]])}' for arm in ARMS))
else:
    for lab in ('ao', 'smoke'):
        print(f'{lab} stdout sha:', {arm: [r['stdout_sha256'][lab] for r in reps[arm]] for arm in ARMS},
              'native', [r['stdout_sha256'][lab] for r in record['native']])
        print(f'{lab} rc:', {arm: [r['markers'].get(f'RC:{lab}') for r in reps[arm]] for arm in ARMS},
              'samples', {arm: [r['markers'].get(f'SAMPLES:{lab}') for r in reps[arm]] for arm in ARMS})
        for tag in ('before', 'after', 'after_gc'):
            keys = sorted({k for r in record['repetitions'] + record['native'] for k in r['object_counts'][lab].get(tag, {})})
            for k in keys:
                print(f'{lab} {tag:8} {k:12}', ' | '.join(f'{arm}: {spread([r["object_counts"][lab].get(tag, {}).get(k) for r in reps[arm]])}' for arm in ARMS),
                      '| native:', spread([r['object_counts'][lab].get(tag, {}).get(k) for r in record['native']]))
        for f in ('VmHWM', 'VmRSS', 'VmPeak'):
            for when in ('done', 'last'):
                print(f'{lab} proc {when} {f} kB:', ' | '.join(f'{arm}: {spread([kb(r["proc"].get(f"{lab}:{when}", {}).get(f)) for r in reps[arm]])}' for arm in ARMS),
                      '| native:', spread([kb(r['proc'].get(f'{lab}:{when}', {}).get(f)) for r in record['native']]))
        keys = sorted({k for r in record['repetitions'] for k in (r['launcher_exit_stats'][lab] or {})})
        for k in keys:
            print(f'{lab} CAPSTONE_VM_STATS {k:14}', ' | '.join(f'{arm}: {spread([(r["launcher_exit_stats"][lab] or {}).get(k) for r in reps[arm]])}' for arm in ARMS))
        keys = sorted({k for r in record['repetitions'] for k in (r['vexec_delta'][lab] or {})})
        for k in keys:
            print(f'{lab} vexec delta {k:22}', ' | '.join(f'{arm}: {spread([(r["vexec_delta"][lab] or {}).get(k) for r in reps[arm]])}' for arm in ARMS))
        for tag in ('before', 'after'):
            keys = sorted({k for r in record['repetitions'] for k in r['vexec_stats'].get(f'{lab}:{tag}', {})})
            for k in keys:
                print(f'{lab} vexec {tag:6} {k:22}', ' | '.join(f'{arm}: {spread([r["vexec_stats"].get(f"{lab}:{tag}", {}).get(k) for r in reps[arm]])}' for arm in ARMS))
