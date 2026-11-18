#!/usr/bin/env python3
"""port-alloc-pool.py DIR: SPEC 502.gcc_r's alloc-pool.c with pool elements as Sublet child lifetimes of their block,
under GCC_CAPSTONE_POOL (nested lane, 2026-11-18). Without the define the preprocessed source is unchanged.

- Each element slot grows by one 16-byte header (one pointer) in front of the caller's bytes.
- pool_alloc hands the slot out as a CDERIVE child of its block (the block's own pointer, which is what the free lists
  hold), bounded to the slot; the header keeps the block's pointer; the caller gets the child, offset past the header.
- pool_free reads the block's pointer from the header through the child, ends the child (CREVOKE(block, child)) and
  threads the slot onto the free list through the block's own pointer.
- empty_alloc_pool and free_alloc_pool free the blocks, as upstream; the heap's free ends each block's lifetime and
  with it every element still live in it (CREVOKE ends a subtree).
The free lists, block geometry and order of reuse are upstream's. Needs capstone/runtime/include (capstone/capability.h).
Every edit is asserted to match exactly once."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'alloc-pool.c'
s = p.read_text()
orig = s
def edit(old, new):
    """Replace old by new, then restore upstream's numbering with #line, so __LINE__ (gcc_assert) is unchanged."""
    global s
    assert s.count(old) == 1 and orig.count(old) == 1, (old[:70], s.count(old))
    assert old.endswith('\n') and new.endswith('\n')
    nxt = orig[:orig.index(old) + len(old)].count('\n') + 1
    s = s.replace(old, new + '#line %d\n' % nxt)

edit('#include "hashtab.h"\n', '''#include "hashtab.h"

#ifdef GCC_CAPSTONE_POOL
#include <capstone/capability.h>
/* Capstone Sublet: every element a child lifetime of its block. The header in
   front of each element keeps the block's own pointer (the parent) at the
   element's address; the caller's pointer is the child, past the header. */
#define POOL_LIFE_HEADER 16	/* one pointer; elements stay 16-byte aligned (align_eight) */
#endif
''')
edit('''  /* Now align the size to a multiple of 4.  */
  size = align_eight (size);
''', '''  /* Now align the size to a multiple of 4.  */
  size = align_eight (size);
#ifdef GCC_CAPSTONE_POOL
  size += POOL_LIFE_HEADER;
#endif
''')
edit('''  /* Pull the first free element from the free list, and return it.  */
  header = pool->returned_free_list;
  pool->returned_free_list = header->next;
  pool->elts_free--;
''', '''  /* Pull the first free element from the free list, and return it.  */
  header = pool->returned_free_list;
  pool->returned_free_list = header->next;
  pool->elts_free--;

#ifdef GCC_CAPSTONE_POOL
  {
    /* header is the block's own pointer to the slot: hand the slot out as its
       child, and keep the parent in the slot's header.  capstone_cap_derive
       takes the child's start as an offset from the parent's base.  */
    char *child = (char *) capstone_cap_derive (header,
	__builtin_capstone_cap_get_cursor (header)
	  - __builtin_capstone_cap_get_base (header), pool->elt_size);
    *(void **) child = (void *) header;
    return (void *) (child + POOL_LIFE_HEADER);
  }
#endif
''')
edit('''  header = (alloc_pool_list) ptr;
  header->next = pool->returned_free_list;
  pool->returned_free_list = header;
  pool->elts_free++;
''', '''#ifdef GCC_CAPSTONE_POOL
  {
    /* End the element's lifetime; the slot goes back through the block's
       own pointer, kept in its header.  */
    char *child = (char *) ptr - POOL_LIFE_HEADER;
    void *block_ptr = *(void **) child;
    capstone_cap_revoke_child (block_ptr, child);
    header = (alloc_pool_list) block_ptr;
  }
#else
  header = (alloc_pool_list) ptr;
#endif
  header->next = pool->returned_free_list;
  pool->returned_free_list = header;
  pool->elts_free++;
''')
p.write_text(s)
print('ported')
