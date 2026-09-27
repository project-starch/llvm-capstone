"""Check the observer against actual libc allocation lifetimes and failures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from allocation_metrics import allocation_samples, valid_allocations

HERE = Path(__file__).resolve().parent
PROGRAM = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
ssize_t __real_write(int fd, const void *p, size_t n) { return write(fd, p, n); }
void exp_alloc_start(void);
void exp_alloc_report(const char *);
int main(void) {
  exp_alloc_start();
  exp_alloc_report("start");
  char *a = malloc(17), *b = calloc(3, 13);
  if (!a || !b) return 1;
  memset(a, 7, 17);
  for (int i=0; i<39; ++i) if (b[i]) return 2;
  exp_alloc_report("both");
  volatile size_t huge = SIZE_MAX;
  if (realloc(a, huge)) return 3;
  if (calloc(huge, huge)) return 4;
  exp_alloc_report("failed");
  a = realloc(a, 97);
  if (!a || a[0] != 7 || a[16] != 7) return 5;
  exp_alloc_report("grown");
  free(a); free(b);
  exp_alloc_report("empty");
  void *aligned;
  if (posix_memalign(&aligned, 64, 128)) return 6;
  exp_alloc_report("aligned");
  free(aligned);
  exp_alloc_report("end");
  return 0;
}
'''


class AllocationTests(unittest.TestCase):
    def test_full_history_is_rejected(self):
        program = r'''
#include <stdlib.h>
#include <unistd.h>
ssize_t __real_write(int fd, const void *p, size_t n) { return write(fd, p, n); }
void exp_alloc_start(void);
void exp_alloc_report(const char *);
int main(void) {
  void *p[4];
  exp_alloc_start();
  for (int i=0; i<4; ++i) if (!(p[i] = malloc(17))) return 1;
  exp_alloc_report("full");
  for (int i=0; i<4; ++i) free(p[i]);
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'main.c').write_text(program)
            subprocess.run([os.environ.get('CC', 'cc'), '-O1', '-fno-builtin',
                '-DEXP_ADDRESS_BITS=1', str(root/'main.c'), str(HERE/'allocations.c'),
                '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign,--wrap=aligned_alloc',
                '-o', str(root/'check')], check=True)
            result = subprocess.run([str(root/'check')], capture_output=True, text=True, check=True)
        self.assertGreater(allocation_samples(result.stderr)[0]['errors'], 0)
        self.assertFalse(valid_allocations(result.stderr, ['full']))

    def test_libc_lifetimes_and_failed_realloc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'main.c').write_text(PROGRAM)
            subprocess.run([os.environ.get('CC', 'cc'), '-O1', '-fno-builtin',
                str(root/'main.c'), str(HERE/'allocations.c'),
                '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign,--wrap=aligned_alloc',
                '-o', str(root/'check')], check=True)
            result = subprocess.run([str(root/'check')], capture_output=True, text=True, check=True)
        rows = allocation_samples(result.stderr)
        self.assertTrue(valid_allocations(result.stderr,
            ['start', 'both', 'failed', 'grown', 'empty', 'aligned', 'end']))
        self.assertEqual([s['live'] for s in rows], [0, 56, 56, 136, 0, 128, 0])
        self.assertEqual(rows[2]['failures'], 2)
        self.assertEqual(rows[-1]['allocations'], 4)
        self.assertEqual(rows[-1]['frees'], 4)
        for corruption in ('errors=1', 'unknown=1', 'peak=0', 'observer=0', 'reuse1=99'):
            key = corruption.split('=')[0]
            import re
            broken = re.sub(r'\b'+key+r'=\d+', corruption, result.stderr)
            self.assertFalse(valid_allocations(broken, [s['phase'] for s in rows]))


if __name__ == '__main__':
    unittest.main()
