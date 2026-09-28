#!/usr/bin/env python3
"""Add the same aggregate GC-slot release-gap observer to pinned mruby arms."""
import argparse
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_SHA = '532d2dfe5ed52425291251d05e8a6784b3e94ff8a8aeba5222cf486f317b840b'
SUBLET_SHA = 'd127b83f9a4ff4baab733c1b4084e92440bfe3b630eb81a43a44cea2b7abab03'
FIELDS = '''#ifdef MRB_GC_STUDY_GAPS
  uint8_t study_seen[MRB_HEAP_PAGE_SIZE/8];
  uint64_t study_release_at[MRB_HEAP_PAGE_SIZE];
#endif
'''


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected one adaptation site ({source.count(old)}): {old[:75]!r}')
    return source.replace(old, new)


def prepare_base(source):
    source = once(source, '  RVALUE objects[MRB_HEAP_PAGE_SIZE];\n} mrb_heap_page;',
        FIELDS + '  RVALUE objects[MRB_HEAP_PAGE_SIZE];\n} mrb_heap_page;')
    source = once(source, '} mrb_heap_page;\n\ntypedef struct mrb_heap_region',
        '} mrb_heap_page;\n#ifdef MRB_GC_STUDY_GAPS\n#include "mruby-gc-gap.inc"\n#endif\n\ntypedef struct mrb_heap_region')
    source = once(source, '  init_heap_page(page);\n  link_heap_page(gc, page);',
        '  init_heap_page(page);\n  link_heap_page(gc, page);\n#ifdef MRB_GC_STUDY_GAPS\n  gc_gap_page_new();\n#endif')
    source = once(source,
        '  RVALUE *p = gc->free_heaps->freelist;\n  gc->free_heaps->freelist = p->as.free.next;',
        '''#ifdef MRB_GC_STUDY_GAPS
  mrb_heap_page *issued_page = gc->free_heaps;
#endif
  RVALUE *p = gc->free_heaps->freelist;
  gc->free_heaps->freelist = p->as.free.next;
#ifdef MRB_GC_STUDY_GAPS
  gc_gap_issue(issued_page, (unsigned)(p - issued_page->objects));
#endif''')
    source = once(source,
        '          if (p->as.basic.tt == MRB_TT_FREE) {\n            p->as.free.next = page->freelist;',
        '''          if (p->as.basic.tt == MRB_TT_FREE) {
#ifdef MRB_GC_STUDY_GAPS
            gc_gap_release(page, (unsigned)(p - page->objects));
#endif
            p->as.free.next = page->freelist;''')
    source = once(source, '    if (!tmp->region) {\n      mrb_free(mrb, tmp);',
        '''    if (!tmp->region) {
#ifdef MRB_GC_STUDY_GAPS
      gc_gap_page_free();
#endif
      mrb_free(mrb, tmp);''')
    source = once(source, '      mrb_free(mrb, page);\n      page = next;',
        '''#ifdef MRB_GC_STUDY_GAPS
      gc_gap_page_free();
#endif
      mrb_free(mrb, page);
      page = next;''')
    return source


def prepare_sublet(source):
    source = once(source,
        '  uint16_t next_free[MRB_HEAP_PAGE_SIZE];\n} mrb_heap_page;',
        '  uint16_t next_free[MRB_HEAP_PAGE_SIZE];\n' + FIELDS + '} mrb_heap_page;')
    source = once(source,
        'mrb_static_assert(MRB_HEAP_PAGE_SIZE <= GCS_NIL);',
        '''mrb_static_assert(MRB_HEAP_PAGE_SIZE <= GCS_NIL);
#ifdef MRB_GC_STUDY_GAPS
#include "mruby-gc-gap.inc"
#endif''')
    source = once(source,
        '  gcs_issued++;\n  return p;',
        '''  gcs_issued++;
#ifdef MRB_GC_STUDY_GAPS
  gc_gap_issue(page, i);
#endif
  return p;''')
    source = once(source,
        'gcs_release(mrb_heap_page *page, uint16_t i)\n{\n  page->obj[i] = NULL;',
        '''gcs_release(mrb_heap_page *page, uint16_t i)
{
#ifdef MRB_GC_STUDY_GAPS
  gc_gap_release(page, i);
#endif
  page->obj[i] = NULL;''')
    source = once(source,
        '  link_heap_page(gc, page);\n}\n\nMRB_API int\nmrb_gc_add_region',
        '''  link_heap_page(gc, page);
#ifdef MRB_GC_STUDY_GAPS
  gc_gap_page_new();
#endif
}

MRB_API int
mrb_gc_add_region''')
    source = once(source,
        '    mrb_free(mrb, tmp);\n#else\n    for (p = tmp->objects',
        '''#ifdef MRB_GC_STUDY_GAPS
    gc_gap_page_free();
#endif
    mrb_free(mrb, tmp);
#else
    for (p = tmp->objects''')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=['spatial', 'sublet'], required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    expected = BASE_SHA if args.arm == 'spatial' else SUBLET_SHA
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('GC source is not the pinned ' + args.arm + ' source')
    code = raw.decode()
    code = prepare_base(code) if args.arm == 'spatial' else prepare_sublet(code)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'gc.c').write_text(code)
    shutil.copy2(HERE/'mruby-gc-gap.inc', args.out/'mruby-gc-gap.inc')
    print(hashlib.sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    main()
