#!/usr/bin/env python3
"""Apply the Sublet lifetime port to an external copy of SPEC CPU2017 502.gcc_r.

TREE is a fresh copy of the benchmark's build directory, as tools/build.sh
leaves it with the kit's `capstone` source alternative. The licensed sources
stay outside the repository. The script reads and changes no benchmark input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
HEADER = 'sublet-port.h'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tree', type=Path)
    parser.add_argument('--record', type=Path,
                        help='write the source identity here, outside the licensed tree')
    args = parser.parse_args()

    root = args.tree.resolve()
    baseline = json.loads((HERE / 'baseline.json').read_text())
    if args.record and args.record.exists():
        raise SystemExit(f'refusing to replace the record {args.record}')
    before = require_baseline(root, baseline)

    port_pools(root)
    port_gc_heap(root)
    port_obstacks(root)
    fix_insn_rescan(root)
    shutil.copyfile(HERE / HEADER, root / HEADER)

    after = require_result(root, baseline)
    if args.record:
        record = {'schema': 1, 'baseline_package_sha256': baseline['baseline_package_sha256'],
                  'before': before, 'after': after,
                  'tools': {name: sha256(HERE / name) for name in ('apply.py', 'baseline.json', HEADER)}}
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    print(f'lifetime port applied to {len(before)} files, {HEADER} added')


def port_pools(root):
    """A child per element at its first hand-out, a renewal at every return."""
    edit(root, 'alloc-pool.c', [
        ('#include "hashtab.h"\n',
         '#include "hashtab.h"\n#include "sublet-port.h"\n'),
        ('      header = (alloc_pool_list) USER_PTR_FROM_ALLOCATION_OBJECT_PTR (block);\n'
         '      header->next = NULL;\n',
         '      /* Sublet: the element leaves the block as a child lifetime of it.  */\n'
         '      header = (alloc_pool_list) sublet_derive (USER_PTR_FROM_ALLOCATION_OBJECT_PTR (block),\n'
         '\t\t\t\t\t\tpool->elt_size);\n'
         '      header->next = NULL;\n'),
        ('  header = (alloc_pool_list) ptr;\n'
         '  header->next = pool->returned_free_list;\n',
         '  /* Sublet: the client\'s lifetime of the element ends here. The free list\n'
         '     keeps the reference to the next one.  */\n'
         '  ptr = sublet_renew (ptr);\n'
         '  header = (alloc_pool_list) ptr;\n'
         '  header->next = pool->returned_free_list;\n'),
    ])


def port_gc_heap(root):
    """A child per object, bounded to its size-class slot. ggc_free ends it."""
    edit(root, 'ggc-page.c', [
        ('#include "tree-flow.h"\n',
         '#include "tree-flow.h"\n#include "sublet-port.h"\n'),
        ('  result = entry->page + object_offset;\n',
         '  result = entry->page + object_offset;\n'
         '  /* Sublet: the object leaves the page as a child lifetime of it.  */\n'
         '  if (object_size < SUBLET_EXACT_LIMIT)\n'
         '    result = sublet_derive (result, object_size);\n'),
        ('  size_t size = OBJECT_SIZE (order);\n'
         '\n'
         '#ifdef GATHER_STATISTICS\n'
         '  ggc_free_overhead (p);\n',
         '  size_t size = OBJECT_SIZE (order);\n'
         '\n'
         '  /* Sublet: the object\'s lifetime ends here.  */\n'
         '  if (size < SUBLET_EXACT_LIMIT)\n'
         '    sublet_release (p);\n'
         '\n'
         '#ifdef GATHER_STATISTICS\n'
         '  ggc_free_overhead (p);\n'),
    ])


def port_obstacks(root):
    """A child per finished object. A free goes by address and ends what it cuts off."""
    edit(root, 'include/obstack.h', [
        ('  char\t*temp_ptr;\t\t/* research build: obstack_finish\'s object, as a pointer */\n',
         '  char\t*temp_ptr;\t\t/* research build: obstack_finish\'s object, as a pointer */\n'
         '  void\t**sublet_issued;\t/* Sublet: the reference of each live object, oldest first */\n'
         '  unsigned sublet_live;\t\t/* Sublet: references in use */\n'
         '  unsigned sublet_room;\t\t/* Sublet: references allocated */\n'),
        ('extern int _obstack_memory_used (struct obstack *);\n',
         'extern int _obstack_memory_used (struct obstack *);\n'
         'extern void *_obstack_sublet_issue (struct obstack *, void *, unsigned long);\n'
         'extern void _obstack_sublet_cut (struct obstack *, void *);\n'
         '#ifndef SPEC\n'
         '# error "the lifetime port covers the obstack macros of a SPEC build only"\n'
         '#endif\n'),
        ('( ((h)->next_free == (h)->object_base\t\t\t\t\t\\\n'
         '   ? (((h)->maybe_empty_object = 1), 0)\t\t\t\t\t\\\n'
         '   : 0),\t\t\t\t\t\t\t\t\\\n'
         '  (h)->temp_ptr = (h)->object_base,\t\t\t\t\t\\\n',
         '( ((h)->next_free == (h)->object_base\t\t\t\t\t\\\n'
         '   ? (((h)->maybe_empty_object = 1),\t\t\t\t\t\\\n'
         '      (h)->temp_ptr = (h)->object_base, 0)\t\t\t\t\\\n'
         '   : ((h)->temp_ptr\t\t\t\t\t\t\t\\\n'
         '      = (char *) _obstack_sublet_issue ((h), (h)->object_base,\t\t\\\n'
         '\t\t\t\t       (h)->next_free - (h)->object_base), 0)), \\\n'),
        ('   ? (ptrdiff_t) ((h)->next_free = (h)->object_base = (h)->temp_ptr)\t\\\n'
         '   : ((obstack_free) ((h), (h)->temp_ptr), 0)))\n',
         '   ? (_obstack_sublet_cut ((h), (h)->temp_ptr), (ptrdiff_t) 0)\t\t\\\n'
         '   : ((obstack_free) ((h), (h)->temp_ptr), (ptrdiff_t) 0)))\n'),
    ])

    # obstack.c carries its free function twice, under two names. Both get the same edits.
    edit(root, 'obstack.c', [
        ('  h->maybe_empty_object = 0;\n'
         '  h->alloc_failed = 0;\n'
         '  return 1;\n',
         '  h->maybe_empty_object = 0;\n'
         '  h->alloc_failed = 0;\n'
         '  h->sublet_issued = 0;\n'
         '  h->sublet_live = h->sublet_room = 0;\n'
         '  return 1;\n', 2),
        ('#undef obstack_free\n',
         '#undef obstack_free\n' + OBSTACK_FUNCTIONS),
        ('      plp = lp->prev;\n'
         '      CALL_FREEFUN (h, lp);\n',
         '      plp = lp->prev;\n'
         '      sublet_forget_chunk (h, lp);\n'
         '      CALL_FREEFUN (h, lp);\n', 2),
        ('      h->object_base = h->next_free = (char *) (obj);\n'
         '      h->chunk_limit = lp->limit;\n'
         '      h->chunk = lp;\n',
         '      h->chunk_limit = lp->limit;\n'
         '      h->chunk = lp;\n'
         '      _obstack_sublet_cut (h, obj);\n', 2),
        ('  else if (obj != 0)\n'
         '    /* obj is not in any of the chunks! */\n'
         '    abort ();\n',
         '  else if (obj != 0)\n'
         '    /* obj is not in any of the chunks! */\n'
         '    abort ();\n'
         '  else\n'
         '    sublet_forget_all (h);\n', 2),
    ])


def fix_insn_rescan(root):
    """gcc's own defect: df_insn_rescan keeps the record that its collected refs name.

    Stock gcc frees the record to its pool and allocates it again. The refs collected in
    between still name the freed record, which works only because the pool hands the same
    element out again at once. GCC keeps the record from release 6.1 on
    (df_insn_info_free_fields, df_insn_info_init_fields)."""
    edit(root, 'df-scan.c', [
        ('      luid = DF_INSN_LUID (insn);\n'
         '      df_insn_delete (NULL, uid);\n'
         '      df_insn_create_insn_record (insn);\n'
         '      DF_INSN_LUID (insn) = luid;\n',
         '      luid = DF_INSN_LUID (insn);\n'
         '      /* Lifetime port: keep the record that the collected refs name.  */\n'
         '      if (insn_info->defs)\n'
         '\t{\n'
         '\t  df_mw_hardreg_chain_delete (insn_info->mw_hardregs);\n'
         '\t  if (df_chain)\n'
         '\t    {\n'
         '\t      df_ref_chain_delete_du_chain (insn_info->defs);\n'
         '\t      df_ref_chain_delete_du_chain (insn_info->uses);\n'
         '\t      df_ref_chain_delete_du_chain (insn_info->eq_uses);\n'
         '\t    }\n'
         '\t  df_ref_chain_delete (insn_info->defs);\n'
         '\t  df_ref_chain_delete (insn_info->uses);\n'
         '\t  df_ref_chain_delete (insn_info->eq_uses);\n'
         '\t}\n'
         '      memset (insn_info, 0, sizeof (struct df_insn_info));\n'
         '      insn_info->insn = insn;\n'
         '      DF_INSN_LUID (insn) = luid;\n'),
    ])


OBSTACK_FUNCTIONS = r'''
/* Sublet. Every finished object leaves the obstack as a child lifetime of its
   chunk. A client frees by address, and the address may come from a reference
   that is already dead, so the obstack keeps the references it issued, oldest
   first, and ends the ones a free cuts off. Freeing a chunk ends its objects
   without a further instruction.  */

#include "sublet-port.h"

POINTER
_obstack_sublet_issue (struct obstack *h, POINTER object, unsigned long size)
{
  POINTER child;

  if (size >= SUBLET_EXACT_LIMIT)
    return object;

  child = sublet_derive (object, size);
  if (h->sublet_live == h->sublet_room)
    {
      h->sublet_room = h->sublet_room ? 2 * h->sublet_room : 32;
      h->sublet_issued = (POINTER *) realloc (h->sublet_issued,
					      h->sublet_room * sizeof (POINTER));
      if (!h->sublet_issued)
	(*obstack_alloc_failed_handler) ();
    }
  h->sublet_issued[h->sublet_live++] = child;
  return child;
}

/* End the objects of the current chunk that begin at OBJ's address or after
   it, and continue allocation at that address.  */

void
_obstack_sublet_cut (struct obstack *h, POINTER obj)
{
  unsigned long address = (unsigned long) obj;
  unsigned long limit = (unsigned long) h->chunk_limit;

  while (h->sublet_live > 0)
    {
      unsigned long top = (unsigned long) h->sublet_issued[h->sublet_live - 1];

      if (top < address || top >= limit)
	break;
      sublet_release (h->sublet_issued[--h->sublet_live]);
    }
  h->object_base = h->next_free
    = (char *) h->chunk + (address - (unsigned long) h->chunk);
}

/* Forget the objects of CHUNK, which the caller frees.  */

static void
sublet_forget_chunk (struct obstack *h, struct _obstack_chunk *chunk)
{
  unsigned long start = (unsigned long) chunk;
  unsigned long limit = (unsigned long) chunk->limit;

  while (h->sublet_live > 0)
    {
      unsigned long top = (unsigned long) h->sublet_issued[h->sublet_live - 1];

      if (top <= start || top >= limit)
	break;
      h->sublet_live--;
    }
}

/* Every chunk is gone: give the reference stack back.  */

static void
sublet_forget_all (struct obstack *h)
{
  free (h->sublet_issued);
  h->sublet_issued = 0;
  h->sublet_live = h->sublet_room = 0;
}
'''


def require_baseline(root, baseline):
    """The hashes of the files the port changes, which must be the pinned ones."""
    if (root / HEADER).exists():
        raise SystemExit(f'{root} already has {HEADER}: use a fresh copy')
    before = {name: sha256(root / name) for name in baseline['before']}
    wrong = [name for name in before if before[name] != baseline['before'][name]]
    if wrong:
        raise SystemExit('unsupported source baseline: ' + ', '.join(wrong))
    return before


def require_result(root, baseline):
    """The hashes of the ported files, which must be the pinned ones."""
    after = {name: sha256(root / name) for name in baseline['after']}
    wrong = [name for name in after if after[name] != baseline['after'][name]]
    if wrong:
        raise SystemExit('the port did not produce the pinned files: ' + ', '.join(wrong))
    return after


def edit(root, name, replacements):
    """Replace each (old, new[, count]) in the file, and fail unless old occurs count times."""
    path = root / name
    text = path.read_text()
    for old, new, *rest in replacements:
        count = rest[0] if rest else 1
        if text.count(old) != count:
            raise SystemExit(f'{name}: expected {count} matches, found {text.count(old)}:\n{old}')
        text = text.replace(old, new)
    path.write_text(text)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
