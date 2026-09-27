#!/usr/bin/env python3
"""Prepare fetched SQLite sources for event-based, repeated-work memory study.

No benchmark sources are vendored. Inputs are explicit and SHA-256 recorded.
The Capstone input is the unprotected, adapted upstream 3.22.0 amalgamation;
the PoisonCap input is the published 3.22.0 fork before local study patches.
"""
import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = HERE.parents[1] / 'ports/sqlite'
BEGIN = '/************** Begin file mem5.c '
END = '/************** End of mem5.c '


def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new, 1)


def apply(s, patch, directory):
    name = patch.read_text().splitlines()[0].removeprefix('--- a/')
    assert Path(name).name == name
    p = directory / name
    p.write_text(s)
    subprocess.run(['patch', '-s', '-F0', '-p1', '-d', str(directory)],
                   input=patch.read_text(), text=True, check=True)
    return p.read_text()


def mem5_span(s):
    a = s.index(BEGIN)
    b = s.index('\n', s.index(END, a)) + 1
    return a, b


OBSERVER = r'''
/* Integer block-index observer. No stored application capabilities. */
#ifdef CAPSTONE_GP_CAPTABLE_ABI
extern int capstone_printf(const char *, ...);
#define study_printf capstone_printf
#else
extern int printf(const char *, ...);
#define study_printf printf
#endif
#define STUDY_MAX_ATOMS 131072
static unsigned char study_ever[STUDY_MAX_ATOMS/8];
static unsigned char study_window[STUDY_MAX_ATOMS/8];
static unsigned char study_starts[STUDY_MAX_ATOMS/8];
static unsigned long study_ever_atoms, study_window_atoms;
static unsigned long study_allocs, study_reused_starts, study_unit;
static void study_cover(unsigned long first, unsigned long count){
  unsigned long j;
  if(first+count>STUDY_MAX_ATOMS) abort();
  study_allocs++;
  if(study_starts[first/8] & (1u<<(first%8))) study_reused_starts++;
  study_starts[first/8] |= 1u<<(first%8);
  for(j=first;j<first+count;j++){
    unsigned char mask = 1u<<(j%8);
    if(!(study_ever[j/8]&mask)) study_ever_atoms++;
    if(!(study_window[j/8]&mask)) study_window_atoms++;
    study_ever[j/8] |= mask;
    study_window[j/8] |= mask;
  }
}
'''


REUSE_OBSERVER = r'''
/* Release-to-next-allocation gap at the same arena start, in allocation
** events. The table contains integers only and survives workload units. */
static unsigned int study_last_release[STUDY_MAX_ATOMS];
static unsigned long study_all_allocs, study_gap_reuses;
static unsigned long study_gap_bins[32];
static void study_gap_release(unsigned long first){
  if(first>=(unsigned long)mem5.nBlock || !study_all_allocs) abort();
  study_last_release[first]=(unsigned int)study_all_allocs;
}
static void study_gap_allocate(unsigned long first){
  unsigned long gap, bin=0;
  if(study_all_allocs>=0xffffffffUL) abort();
  study_all_allocs++;
  if(study_last_release[first]){
    gap=study_all_allocs-study_last_release[first];
    if(!gap) abort();
    while(gap>1){gap>>=1;bin++;}
    if(bin>=32) abort();
    study_gap_bins[bin]++;
    study_gap_reuses++;
    study_last_release[first]=0;
  }
}
static void study_gap_emit(void){
  unsigned long i;
  study_printf("STUDY-GAP-TOTAL unit=%lu allocs=%lu reuses=%lu\n",
    study_unit,study_all_allocs,study_gap_reuses);
  for(i=0;i<32;i+=2){
    study_printf("STUDY-GAP unit=%lu pair=%lu a=%lu b=%lu\n",
      study_unit,i/2,study_gap_bins[i],study_gap_bins[i+1]);
  }
}
'''


def instrument(s, arm, reuse_gaps=False):
    if arm == 'capstone-sublet':
        # Legacy sharing rounds the region to pages. Keep the allocatable atom
        # count equal to the original heap, and leave the extra tail unused.
        s = replace(s, '  *pEnd = capstone_cap_end(&memsys5Grant);',
                    '  *pEnd = *pBase + 129055UL*64;')
        s = replace(s, '  mem5.nBlock = (int)((capstone_cap_end(&memsys5Grant)-mem5.poolBase)/mem5.szAtom);',
                    '  if(capstone_cap_end(&memsys5Grant)-mem5.poolBase < 129055UL*64) return SQLITE_NOMEM;\n  mem5.nBlock = 129055;')
    observer = OBSERVER
    if reuse_gaps:
        observer += REUSE_OBSERVER
        observer = replace(observer, '  study_allocs++;',
                           '  study_allocs++;\n  study_gap_allocate(first);')
        # A prototype is needed because the existing observer calls the
        # release-gap observer before its definition below.
        observer = observer.replace('static void study_cover(',
                                    'static void study_gap_allocate(unsigned long);\nstatic void study_cover(',1)
    s = replace(s, '#define mem5 GLOBAL(struct Mem5Global, mem5)',
                '#define mem5 GLOBAL(struct Mem5Global, mem5)\n' + observer)
    if reuse_gaps:
        first = ('((uptr)pPrior-mem5.poolBase)/mem5.szAtom' if arm == 'capstone-sublet'
                 else '(cheri_getaddress(pPrior)-mem5_heap_base)/mem5.szAtom' if arm == 'poisoncap-temporal'
                 else '((u8 *)pPrior-mem5.zPool)/mem5.szAtom')
        s = replace(s, 'static void memsys5Free(void *pPrior){\n  assert( pPrior!=0 );',
                    'static void memsys5Free(void *pPrior){\n  assert( pPrior!=0 );\n'
                    f'  study_gap_release((unsigned long)({first}));')
    s = replace(s, '  mem5.aCtrl[i] = iLogsize;',
                '  mem5.aCtrl[i] = iLogsize;\n  study_cover(i, iFullSz/mem5.szAtom);')
    protected = arm == 'poisoncap-temporal'
    if arm.startswith('poisoncap-') and protected:
        live, peak, held, oom = 'study_live', 'study_peak_held', 'study_held', 'study_oom_requests'
        link = 'mem5_link[j].next'
        # mmap rounds the external free-list allocation to pages.
        meta = '((mem5.nBlock*sizeof(Mem5Link)+4095UL)&~4095UL) + mem5.nBlock + sizeof(quarantine) + sizeof(mem5)'
        extra = 'printf("STUDY-REVOKE unit=%lu calls=%lu errors=%lu full=%lu threshold=%lu\\n", study_unit, (unsigned long)study_revokes, (unsigned long)study_revoke_errors, (unsigned long)study_full_drains, (unsigned long)study_threshold_drains);'
    else:
        live, peak, held, oom = 'study_live', 'study_peak', 'study_live', 'study_oom'
        link = 'mem5_link[j].next' if arm == 'capstone-sublet' else 'MEM5LINK(j)->next'
        # Sublet puts all per-atom tables in the configured heap.
        meta = 'sqlite3GlobalConfig.nHeap + sizeof(mem5)' if arm == 'capstone-sublet' else 'mem5.nBlock + sizeof(mem5)'
        extra = ''
    # Identify the actual Sublet link accessor rather than guessing its name.
    if arm == 'capstone-sublet':
        link = 'MEM5LINK(j)->next'
    report = r'''
SQLITE_API void sqlite3_study_begin(unsigned long unit){
  study_unit=unit;
  memset(study_window,0,sizeof(study_window));
  study_window_atoms=study_allocs=study_reused_starts=0;
  @PEAK@=@HELD@;
}
SQLITE_API void sqlite3_study_sample(int phase){
  unsigned long f=0, largest=0, blocks=0, m=(unsigned long)(@META@);
  int k,j;
  for(k=0;k<=LOGMAX;k++){
    for(j=mem5.aiFreelist[k];j>=0;j=@LINK@){
      unsigned long bytes=((unsigned long)mem5.szAtom)<<k;
      if(++blocks>(unsigned long)mem5.nBlock) abort();
      f+=bytes;
      if(bytes>largest) largest=bytes;
    }
  }
  if(f+(unsigned long)@HELD@!=(unsigned long)mem5.nBlock*mem5.szAtom) abort();
  study_printf("STUDY-LEDGER unit=%lu phase=%d live=%lu held=%lu peak_held=%lu",
    study_unit,phase,(unsigned long)@LIVE@,(unsigned long)@HELD@,(unsigned long)@PEAK@);
  study_printf(" quarantine=%lu metadata=%lu free=%lu largest_free=%lu pool=%lu",
    (unsigned long)(@HELD@-@LIVE@),m,f,largest,(unsigned long)mem5.nBlock*mem5.szAtom);
  study_printf(" ever=%lu window=%lu allocs=%lu reused_starts=%lu oom=%lu observer=%lu\n",
    study_ever_atoms*mem5.szAtom,study_window_atoms*mem5.szAtom,study_allocs,
    study_reused_starts,(unsigned long)@OOM@,
    (unsigned long)(sizeof(study_ever)+sizeof(study_window)+sizeof(study_starts)));
  @EXTRA@
  @REUSE_GAPS@
}
'''
    for key, value in dict(PEAK=peak, HELD=held, LIVE=live, OOM=oom, META=meta,
                           LINK=link, EXTRA=extra,
                           REUSE_GAPS='if(phase<0) study_gap_emit();' if reuse_gaps else '').items():
        report = report.replace('@'+key+'@', value)
    if reuse_gaps:
        report = replace(report, 'sizeof(study_starts))',
                         'sizeof(study_starts)+sizeof(study_last_release)'
                         '+sizeof(study_gap_bins)+sizeof(study_all_allocs)'
                         '+sizeof(study_gap_reuses))')
    # Place within ENABLE_MEMSYS5 after all allocator symbols are defined.
    a, b = mem5_span(s)
    section = s[a:b]
    section = replace(section, '#endif /* SQLITE_ENABLE_MEMSYS5 */', report+'\n#endif /* SQLITE_ENABLE_MEMSYS5 */')
    return s[:a]+section+s[b:]


WRAPPER = r'''
/* A unit is an entire official main testset, including normal database close. */
int main(int argc, char **argv){
  int i, rc, units=17, burst=0, cleanargc=1;
  char *clean[32];
  clean[0]=argv[0];
  for(i=1;i<argc;i++){
    if(strcmp(argv[i],"--study-units")==0 && i+1<argc) units=atoi(argv[++i]);
    else if(strcmp(argv[i],"--study-burst")==0) burst=1;
    else { if(cleanargc>=28) return 96; clean[cleanargc++]=argv[i]; }
  }
  if(units<1 || units>1000) return 97;
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  {
    void *pool=malloc(8388608);
    if(!pool) return 98;
    rc=sqlite3_config(SQLITE_CONFIG_HEAP,pool,8388608,64);
    if(rc) return rc;
    rc=sqlite3_config(SQLITE_CONFIG_LOOKASIDE,0,0);
    if(rc) return rc;
    rc=sqlite3_initialize();
    if(rc) return rc;
  }
#endif
  if(burst) units=13;
  for(i=0;i<units;i++){
    int n=cleanargc;
    memset(&g,0,sizeof(g));
    clean[n++]="--size";
    clean[n++]=(burst && i==4)?"4":"1";
    clean[n]=0;
    sqlite3_study_begin(i);
    printf("STUDY-BEGIN unit=%d size=%s\n",i,clean[n-1]);
    rc=study_speedtest1_once(n,clean);
    sqlite3_study_sample(-1);
    printf("STUDY-END unit=%d rc=%d\n",i,rc);
    if(rc) return rc;
  }
  printf("STUDY-COMPLETE units=%d\n",units);
  return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('upstream', 'capstone-source', 'poisoncap-source', 'driver', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--reuse-gaps', action='store_true',
                   help='add logical-release-to-same-start gap accounting')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {k: {'path': str(v), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()}
              for k,v in vars(args).items() if k not in ('out','reuse_gaps')}
    inputs['reuse_gaps'] = args.reuse_gaps
    (args.out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    header = args.upstream.with_name('sqlite3.h')
    (args.out/'sqlite3.h').write_bytes(header.read_bytes())
    cap = args.capstone_source.read_text()
    stock = args.upstream.read_text()
    poison = args.poisoncap_source.read_text()
    temporal = apply(poison, PORT/'study/poisoncap-322-instrumented.patch', args.out)
    # Replacing the allocator preserves every application-side CHERI ABI fix.
    a,b = mem5_span(temporal)
    c,d = mem5_span(stock)
    headers = temporal[temporal.index('#include <sys/param.h>', a):temporal.index('#define NESTED_TEMPORAL 1', a)]
    original = temporal[:a]+headers+stock[c:d]+temporal[b:]
    original = replace(original, 'return (void*)&mem5.zPool[i*mem5.szAtom];',
                       'return cheri_setbounds((void*)&mem5.zPool[i*mem5.szAtom], iFullSz);')
    sources = {'capstone': apply(cap,PORT/'study/capstone-322-spatial-stats.patch',args.out),
               'capstone-sublet': apply(apply(cap,PORT/'sublet/sublet-3220000-memsys5.patch',args.out),PORT/'study/capstone-322-sublet-stats.patch',args.out),
               'poisoncap-spatial': apply(original,PORT/'study/capstone-322-spatial-stats.patch',args.out),
               'poisoncap-temporal': temporal}
    # Correct the published full-queue policy directly in the protected source.
    # Both builds use identical explicit compiler options.
    sources['poisoncap-temporal'] = sources['poisoncap-temporal'].replace('#ifdef SQLITE_STUDY_CORRECT_FULL_QUEUE','#if 1 /* corrected full-queue policy */')
    # Record the unprotected port baselines before adding observation.
    (args.out/'capstone-port-base.c').write_text(cap)
    (args.out/'poisoncap-port-base.c').write_text(original)
    for arm,s in sources.items():
        s = instrument(s,arm,args.reuse_gaps)
        (args.out/(arm+'.c')).write_text(s)
        base = cap if arm.startswith('capstone') else original
        patch = ''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/sqlite3-capstone.c',tofile='b/sqlite3-capstone.c'))
        (args.out/(arm+'.patch')).write_text(patch)
        if arm == 'capstone-sublet':
            subbase = apply(cap, PORT/'sublet/sublet-3220000-memsys5.patch', args.out)
            overlay = ''.join(difflib.unified_diff(subbase.splitlines(True), s.splitlines(True), fromfile='a/sqlite3-capstone.c', tofile='b/sqlite3-capstone.c'))
            (args.out/'capstone-sublet-observer.patch').write_text(overlay)
    # The pinned driver may already carry the oracle patch. Preserve its
    # exact bytes and avoid applying the same patch twice.
    driver = args.driver.read_text()
    if 'STUDY-ORACLE phase=' not in driver:
        driver = apply(driver, PORT/'study/speedtest1-oracle.patch', args.out)
    driver = replace(driver, 'int main(int argc, char **argv){','static int study_speedtest1_once(int argc, char **argv){')
    driver = replace(driver, '  g.iStart = speedtest1_timestamp();','  sqlite3_study_sample(study_phase);\n  g.iStart = speedtest1_timestamp();')
    driver = '#define SQLITE_STUDY_ORACLE 1\n'+driver+WRAPPER
    (args.out/'speedtest1.c').write_text(driver)


if __name__ == '__main__':
    main()
