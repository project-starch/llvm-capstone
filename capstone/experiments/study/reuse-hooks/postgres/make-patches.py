#!/usr/bin/env python3
"""Generate the PostgreSQL reuse-distance hook patches, one per arm.

  make-patches.py <stock mcxt.c> <protected mcxt.c> <out dir>

Bases: src/backend/utils/mmgr/mcxt.c of 17.5 as build-domain.sh prepares it.
  stock      port patches 0001-0017 (0017's observer is behind
             PG_SPATIAL_GAP_OBSERVER, which these builds do not define)
  protected  port patches 0001-0016 and memory-contexts patch 0003
The same hook text goes into both at the same places: after each manager
alloc call, before each reset/delete method call, and around the free and
realloc method calls. Patch 0003 rewrites the free and realloc call lines
(SubletRelease before, SubletReissue around), so there the arm's own call
sequence is the anchor: the old chunk is read before the sequence (in the
protected arm before SubletRelease revokes the caller's pointer) and the
outcome is recorded after it. Every anchor must match the expected number of
times or the script stops.
"""
import difflib
import re
import sys
from pathlib import Path

INCLUDE = ('#ifdef EXP_REUSE_GAP\n#include "exp-reuse-gap.h"\n#endif\n\n')
ISSUE = ('#ifdef EXP_REUSE_GAP\n'
         '\texp_gap_issue((uintptr_t) context, (uintptr_t) ret, size);\n'
         '#endif\n')
FREE_BEFORE = ('#ifdef EXP_REUSE_GAP\n'
               '\tuintptr_t\texp_gap_old_base = exp_gap_base(pointer);\n'
               '\tMemoryContextMethodID exp_gap_old_method = GetMemoryChunkMethodID(pointer);\n'
               '#endif\n')
FREE_AFTER = ('#ifdef EXP_REUSE_GAP\n'
              '\tif (exp_gap_old_method != MCTX_BUMP_ID &&\n'
              '\t\texp_gap_old_method != MCTX_ALIGNED_REDIRECT_ID)\n'
              '\t\texp_gap_release(exp_gap_old_base);\n'
              '#endif\n')
REALLOC_BEFORE = ('#ifdef EXP_REUSE_GAP\n'
                  '\tuintptr_t\texp_gap_old_base = exp_gap_base(pointer);\n'
                  '\tMemoryContext exp_gap_old_context = GetMemoryChunkContext(pointer);\n'
                  '\tMemoryContextMethodID exp_gap_old_method = GetMemoryChunkMethodID(pointer);\n'
                  '#endif\n')
REALLOC_AFTER = ('#ifdef EXP_REUSE_GAP\n'
                 '\tif (ret != NULL && exp_gap_old_method != MCTX_ALIGNED_REDIRECT_ID)\n'
                 '\t{\n'
                 '\t\tuintptr_t\texp_gap_new_base = exp_gap_base(ret);\n'
                 '\n'
                 '\t\tif (exp_gap_new_base == exp_gap_old_base)\n'
                 '\t\t\texp_gap_resize(exp_gap_old_base, size);\n'
                 '\t\telse\n'
                 '\t\t{\n'
                 '\t\t\texp_gap_release(exp_gap_old_base);\n'
                 '\t\t\texp_gap_issue((uintptr_t) exp_gap_old_context, exp_gap_new_base, size);\n'
                 '\t\t}\n'
                 '\t}\n'
                 '#endif\n')

# The arm's own free and realloc call sequences.
FREE_SEQ = {
    'stock': r'\tMCXT_METHOD\(pointer, free_p\) \(pointer\);\n',
    'protected': r'\tpointer = SubletRelease\(pointer\);\n'
                 r'\tMCXT_METHOD\(pointer, free_p\) \(pointer\);\n',
}
REALLOC_SEQ = {
    'stock': r'\tret = MCXT_METHOD\(pointer, realloc\) \(pointer, size, (?:0|flags)\);\n',
    'protected': r'\tpointer = SubletRelease\(pointer\);\n'
                 r'\tret = SubletReissue\(MCXT_METHOD\(pointer, realloc\) '
                 r'\(pointer, size, (?:0|flags)\), size\);\n',
}


def edit(text, pattern, count, repl):
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != count:
        sys.exit(f'anchor {pattern!r}: {n} matches, expected {count}')
    return new


def instrument(text, arm):
    text = edit(text, r'^/\*\n \* MemoryContextTraverseNext\n', 1,
                lambda m: INCLUDE + m.group(0))
    text = edit(text, r'^\tret = context->methods->alloc\(context, size, \w+\);\n', 7,
                lambda m: m.group(0) + ISSUE)
    text = edit(text, r'^\t\tcontext->methods->reset\(context\);\n', 1,
                lambda m: '#ifdef EXP_REUSE_GAP\n\t\texp_gap_reset((uintptr_t) context);\n'
                          '#endif\n' + m.group(0))
    text = edit(text, r'^\tcontext->methods->delete_context\(context\);\n', 1,
                lambda m: '#ifdef EXP_REUSE_GAP\n\texp_gap_reset((uintptr_t) context);\n'
                          '#endif\n' + m.group(0))
    text = edit(text, '^' + FREE_SEQ[arm], 1,
                lambda m: FREE_BEFORE + m.group(0) + FREE_AFTER)
    text = edit(text, '^' + REALLOC_SEQ[arm], 2,
                lambda m: REALLOC_BEFORE + m.group(0) + REALLOC_AFTER)
    return text


def main():
    stock, protected, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    name = 'src/backend/utils/mmgr/mcxt.c'
    for arm, base in (('stock', stock), ('protected', protected)):
        text = base.read_text()
        diff = difflib.unified_diff(text.splitlines(keepends=True),
                                    instrument(text, arm).splitlines(keepends=True),
                                    'a/' + name, 'b/' + name, n=3)
        (out / f'hooks-{arm}.patch').write_text(''.join(diff))


if __name__ == '__main__':
    main()
