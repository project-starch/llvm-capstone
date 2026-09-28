#!/usr/bin/env python3
"""Apply the pinned whole-interpreter PoisonCap GC-slot adapter in scratch."""
import argparse
import hashlib
import shutil
from pathlib import Path

GC_SHA256 = '532d2dfe5ed52425291251d05e8a6784b3e94ff8a8aeba5222cf486f317b840b'
HERE = Path(__file__).resolve().parent


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'expected one GC adaptation site, found {source.count(before)}: {before[:65]!r}')
    return source.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Copied, pinned mruby source tree')
    args = parser.parse_args()
    gc_file = args.source / 'src/gc.c'
    raw = gc_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GC_SHA256:
        raise ValueError('mruby gc.c differs from the qualified 4.0.0-rc2 source')
    source = raw.decode()
    source = replace_once(source, '#include <string.h>\n',
        '#include <string.h>\n#ifdef MRB_GC_STUDY_POISONCAP\n#include <stdint.h>\n#endif\n')
    source = replace_once(source, '  RVALUE objects[MRB_HEAP_PAGE_SIZE];\n} mrb_heap_page;',
        '''#ifdef MRB_GC_STUDY_POISONCAP
  uint8_t study_quarantine[MRB_HEAP_PAGE_SIZE/8];
  uint8_t study_seen[MRB_HEAP_PAGE_SIZE/8];
  uint64_t study_release_at[MRB_HEAP_PAGE_SIZE];
#endif
  RVALUE objects[MRB_HEAP_PAGE_SIZE];
} mrb_heap_page;''')
    source = replace_once(source, 'mrb_noreturn void mrb_raise_nomemory(mrb_state *mrb);',
        '''mrb_noreturn void mrb_raise_nomemory(mrb_state *mrb);
#ifdef MRB_GC_STUDY_POISONCAP
#include "mruby-poisoncap-gc.inc"
#endif''')
    source = replace_once(source,
        '  mrb_heap_page *page = (mrb_heap_page*)mrb_calloc(mrb, 1, sizeof(mrb_heap_page));\n  init_heap_page(page);',
        '''#ifdef MRB_GC_STUDY_POISONCAP
  mrb_heap_page *page = mrb_gc_study_new_page();
  if (!page) mrb_raise_nomemory(mrb);
#else
  mrb_heap_page *page = (mrb_heap_page*)mrb_calloc(mrb, 1, sizeof(mrb_heap_page));
#endif
  init_heap_page(page);''')
    source = replace_once(source,
        '      if (p->as.free.tt != MRB_TT_FREE)\n        obj_free(mrb, &p->as.basic, TRUE);\n    }\n    if (!tmp->region) {\n      mrb_free(mrb, tmp);',
        '''#ifdef MRB_GC_STUDY_POISONCAP
      if (mrb_gc_study_quarantined(tmp, p)) continue;
#endif
      if (p->as.free.tt != MRB_TT_FREE)
        obj_free(mrb, &p->as.basic, TRUE);
    }
    if (!tmp->region) {
#ifdef MRB_GC_STUDY_POISONCAP
      mrb_gc_study_free_page(tmp);
#else
      mrb_free(mrb, tmp);
#endif''')
    # Empty reusable storage grows the heap normally. Reclamation occurs on
    # the published byte/queue thresholds at release, not on allocation demand.
    source = replace_once(source,
        '  paint_partial_white(gc, &p->as.basic);\n  return &p->as.basic;',
        '''  paint_partial_white(gc, &p->as.basic);
#ifdef MRB_GC_STUDY_POISONCAP
  return mrb_gc_study_issue(gc->free_heaps ? gc->free_heaps : gc->heaps, p);
#else
  return &p->as.basic;
#endif''')
    # The selected free_heap page is not necessarily gc->free_heaps after the
    # last slot is popped; preserve it before mutating the list.
    source = replace_once(source,
        '  RVALUE *p = gc->free_heaps->freelist;\n  gc->free_heaps->freelist = p->as.free.next;',
        '''  mrb_heap_page *issued_page = gc->free_heaps;
  RVALUE *p = issued_page->freelist;
  gc->free_heaps->freelist = p->as.free.next;''')
    source = replace_once(source,
        'return mrb_gc_study_issue(gc->free_heaps ? gc->free_heaps : gc->heaps, p);',
        'return mrb_gc_study_issue(issued_page, p);')
    source = replace_once(source,
        '    while (p<e) {\n      if (is_dead(gc, &p->as.basic)) {',
        '''    while (p<e) {
#ifdef MRB_GC_STUDY_POISONCAP
      if (mrb_gc_study_quarantined(page, p)) { p++; continue; }
#endif
      if (is_dead(gc, &p->as.basic)) {''')
    source = replace_once(source,
        '''          if (p->as.basic.tt == MRB_TT_FREE) {
            p->as.free.next = page->freelist;
            page->freelist = p;
            freed++;
          }''',
        '''          if (p->as.basic.tt == MRB_TT_FREE) {
#ifdef MRB_GC_STUDY_POISONCAP
            mrb_gc_study_release(gc, page, p);
            if (!mrb_gc_study.mode) {
#endif
              p->as.free.next = page->freelist;
              page->freelist = p;
#ifdef MRB_GC_STUDY_POISONCAP
            }
#endif
            freed++;
          }''')
    source = replace_once(source, '    if (dead_slot && !page->region) {',
        '''#ifdef MRB_GC_STUDY_POISONCAP
    if (0) { /* Keep pages in both modes; charge their identical reservation. */
#else
    if (dead_slot && !page->region) {
#endif''')
    source = replace_once(source,
        '      mrb_free(mrb, page);\n      page = next;',
        '''#ifdef MRB_GC_STUDY_POISONCAP
      mrb_gc_study_free_page(page);
#else
      mrb_free(mrb, page);
#endif
      page = next;''')
    shutil.copy2(HERE / 'mruby-poisoncap-gc.inc', args.source / 'src/mruby-poisoncap-gc.inc')
    shutil.copy2(HERE.parents[1] / 'ports/common/include/poisoncap-quarantine-policy.h',
                 args.source / 'src/poisoncap-quarantine-policy.h')
    gc_file.write_text(source)
    print(hashlib.sha256(source.encode()).hexdigest())


if __name__ == '__main__':
    main()
