"""Regression checks for corrupt or duplicate benchmark telemetry."""
import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('sqlite_memory_plot',Path(__file__).with_name('plot-sqlite-normalized.py'))
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
gap_spec=importlib.util.spec_from_file_location('sqlite_reuse_gap',Path(__file__).with_name('plot-sqlite-reuse-gaps.py'))
gap=importlib.util.module_from_spec(gap_spec);gap_spec.loader.exec_module(gap)

SAMPLE='''STUDY-BEGIN unit=0 size=1
STUDY-ORACLE phase=100 rows=0 hash=cbf29ce484222325
STUDY-LEDGER unit=0 phase=-1 live=0 held=64 peak_held=128 quarantine=64 metadata=32 free=192 largest_free=128 pool=256 ever=192 window=192 allocs=4 reused_starts=1 oom=0 observer=49152
STUDY-END unit=0 rc=0
STUDY-COMPLETE units=1
'''


class Telemetry(unittest.TestCase):
    def test_valid_quarantine_is_not_live(self):
        rows,_,ends,_=plot.parse(SAMPLE)
        self.assertEqual(rows[0]['quarantine'],64)
        self.assertEqual(ends,{0:0})

    def test_rejects_stack_argument_corruption(self):
        for field,value in [('pool','192'),('ever','257'),('reused_starts','5'),
                            ('peak_held','63'),('largest_free','193')]:
            import re
            with self.subTest(field=field),self.assertRaises(ValueError):
                plot.parse(re.sub(r'\b'+field+r'=\d+',field+'='+value,SAMPLE))

    def test_rejects_duplicate_output(self):
        with self.assertRaises(ValueError):plot.parse(SAMPLE+SAMPLE)


class ReleaseGaps(unittest.TestCase):
    def sample(self):
        # One setup allocation, then three measured allocations. The first
        # measured allocation reuses a logically released same-start block.
        lines=['STUDY-LEDGER unit=0 phase=-1 live=0 held=0 peak_held=64 '
               'quarantine=0 metadata=0 free=256 largest_free=256 pool=256 '
               'ever=128 window=128 allocs=3 reused_starts=1 oom=0 observer=573712',
               'STUDY-GAP-TOTAL unit=0 allocs=4 reuses=1']
        lines += [f'STUDY-GAP unit=0 pair={i} a={int(i==0)} b=0'
                  for i in range(16)]
        lines += ['STUDY-COMPLETE units=1']
        return '\n'.join(lines)

    def test_excludes_setup_allocation_from_denominator(self):
        parsed=gap.parse(self.sample(),1)
        self.assertEqual((parsed['allocations'],parsed['setup_allocations'],parsed['reuses']),
                         (3,1,1))

    def test_rejects_missing_or_contradictory_bin(self):
        self.assertRaises(ValueError,gap.parse,self.sample().replace('pair=15','pair=14'),1)
        self.assertRaises(ValueError,gap.parse,self.sample().replace('pair=0 a=1','pair=0 a=2'),1)


if __name__=='__main__':unittest.main()
