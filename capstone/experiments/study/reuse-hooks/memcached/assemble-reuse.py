#!/usr/bin/env python3
"""Assemble the memcached row of the reuse-distance measurement from two run-reuse.py records.

    assemble-reuse.py STOCK_RESULT_JSON PROTECTED_RESULT_JSON BUILD_LOG OUT_JSON

Each arm's top-level counts are repetition 1 (the pre-registration's one run per arm); every
repetition is kept under "repetitions". Bucket k is the observer's bins[k-1]; empty buckets are
omitted. Also prints, per boundary: the median bucket per arm, the share of reuses whose bucket
differs between the arms (sum over k of |s_k/N_s - p_k/N_p| / 2) for repetition 1 and for every
stock x protected repetition pair, and the per-bucket ranges over the repetitions.
"""
import itertools
import json
import re
import sys

BOUNDARIES = ('slabs', 'cache')
FIELDS = ('issues', 'releases', 'reuses', 'distinct_starts', 'attempts', 'error')


def arm_record(rep):
    out = {}
    for b in BOUNDARIES:
        r = rep['reuse'][b]
        out[b] = dict(buckets={str(k + 1): n for k, n in enumerate(r['bins']) if n},
                      **{f: r[f] for f in FIELDS})
    return out


def median_bucket(buckets):
    n = sum(buckets.values())
    if not n:
        return None
    cum = 0
    for k in sorted(buckets, key=int):
        cum += buckets[k]
        if 2 * cum >= n:
            return int(k)


def reusediff(s, p):
    ns, np_ = sum(s.values()), sum(p.values())
    if not ns or not np_:
        return None
    keys = set(s) | set(p)
    return sum(abs(s.get(k, 0) / ns - p.get(k, 0) / np_) for k in keys) / 2


def main():
    stock, prot = (json.load(open(path)) for path in sys.argv[1:3])
    build_log = open(sys.argv[3]).read()
    for rec, arm in ((stock, 'stock'), (prot, 'protected')):
        assert rec['arm'] == arm and rec['completed'], (arm, rec['completed'])
        for rep in rec['repetitions']:
            assert rep['harness_status'] == 0 and rep['job'] == {'version': 1, 'kind': 'exit', 'value': 0}, (arm, rep)
            for b in BOUNDARIES:
                assert rep['reuse'][b]['arm'] == arm and rep['reuse'][b]['error'] == 0, (arm, b, rep['reuse'][b])
    transcripts = {r['transcript_sha256'] for rec in (stock, prot) for r in rec['repetitions']}
    reps = {arm: [arm_record(r) for r in rec['repetitions']] for arm, rec in (('stock', stock), ('protected', prot))}
    boundaries = {}
    summary = {}
    for b in BOUNDARIES:
        boundaries[b] = {}
        for arm in ('stock', 'protected'):
            boundaries[b][arm] = dict(reps[arm][0][b])
            boundaries[b][arm]['repetitions'] = [dict(rep=i + 1, **r[b]) for i, r in enumerate(reps[arm])]
        s1, p1 = boundaries[b]['stock']['buckets'], boundaries[b]['protected']['buckets']
        keys = sorted({k for arm in reps for r in reps[arm] for k in r[b]['buckets']}, key=int)
        ranges = {arm: {k: [min(r[b]['buckets'].get(k, 0) for r in reps[arm]),
                            max(r[b]['buckets'].get(k, 0) for r in reps[arm])] for k in keys} for arm in reps}
        overlap = all(ranges['stock'][k][0] <= ranges['protected'][k][1] and
                      ranges['protected'][k][0] <= ranges['stock'][k][1] for k in keys)
        summary[b] = dict(
            median_bucket={arm: median_bucket(boundaries[b][arm]['buckets']) for arm in reps},
            median_bucket_per_rep={arm: [median_bucket(r[b]['buckets']) for r in reps[arm]] for arm in reps},
            reusediff_rep1=reusediff(s1, p1),
            reusediff_all_pairs=[reusediff(s[b]['buckets'], p[b]['buckets'])
                                 for s, p in itertools.product(reps['stock'], reps['protected'])],
            reusediff_within_stock=[reusediff(x[b]['buckets'], y[b]['buckets'])
                                    for x, y in itertools.combinations(reps['stock'], 2)],
            reusediff_within_protected=[reusediff(x[b]['buckets'], y[b]['buckets'])
                                        for x, y in itertools.combinations(reps['protected'], 2)],
            equal_rep1=s1 == p1 and all(boundaries[b]['stock'][f] == boundaries[b]['protected'][f] for f in FIELDS),
            identical_across_reps={arm: all(r[b] == reps[arm][0][b] for r in reps[arm]) for arm in reps},
            bucket_ranges=ranges, ranges_overlap=overlap)
    images = {}
    for arm in ('stock', 'protected'):
        m = re.search(rf'^{arm} memcached.dom: sha256 (\w+) bytes (\d+) cderive=(\d+) crevoke=(\d+)$', build_log, re.M)
        images[arm] = dict(sha256=stock['staged_sha256']['memcached.dom'] if arm == 'stock'
                           else prot['staged_sha256']['memcached.dom'],
                           bytes=int(m[2]), cderive=int(m[3]), crevoke=int(m[4]),
                           cppflags='-DEXP_REUSE_GAP' + (' -DMC_CAPSTONE_SUBLET' if arm == 'protected' else ''))
    assert stock['platform_sha256'] == prot['platform_sha256']
    assert all(stock['staged_sha256'][n] == prot['staged_sha256'][n]
               for n in ('capstone-vexec', 'capstone_vm.ko', 'capstone-job', 'mc-harness'))
    out = {
        'program': 'memcached 1.6.45', 'pre_registration': '16d2171a3edc',
        'workload': ('the protocol oracle unchanged (ports/memcached/app/host/mc-harness/mc-harness.c, sha256 '
                     '1e569a8b63b4...: one connection through every command family and values 1 B-900 KB plus one '
                     'too large, then 8 concurrent pipelining connections x 40 set/get/incr, then stats) against '
                     '`memcached ' + stock['flags'] + '` under capstone-job as uid 65534, stopped with TERM; '
                     'three servers in turn per boot, one boot per arm; top-level counts are repetition 1'),
        'boundaries': boundaries,
        'summary': summary,
        'sanity': dict(transcripts_equal_across_arms_and_reps=len(transcripts) == 1,
                       transcript_sha256=sorted(transcripts)),
        'images': images,
        'platform': dict(sha256=stock['platform_sha256'],
                         launcher_sha256=stock['staged_sha256']['capstone-vexec'],
                         module_sha256=stock['staged_sha256']['capstone_vm.ko'],
                         job_sha256=stock['staged_sha256']['capstone-job'],
                         harness_sha256=stock['staged_sha256']['mc-harness'],
                         harts=1, workers=4,
                         boot_wall_seconds={'stock': stock['boot_wall_seconds'], 'protected': prot['boot_wall_seconds']}),
    }
    with open(sys.argv[4], 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    print(json.dumps(summary, indent=1))
    print('transcripts equal:', len(transcripts) == 1)


if __name__ == '__main__':
    raise SystemExit(main())
