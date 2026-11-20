#!/usr/bin/env python3
"""fix-stale.py DIR: fixes for SPEC 502.gcc_r (gcc 4.5) sources in DIR, for every place found where gcc uses an
alloc-pool element after pool_free (nested lane, 2026-11-18; found natively with AddressSanitizer and a quarantining
pool_free, see nested/gcc/asan). Each is a real stale use that stock gcc survives only because pool_free followed by
pool_alloc hands the same element out again. The fixes do not change what gcc computes. Every edit is asserted to
match exactly once and is followed by a #line that restores upstream's line numbers (__LINE__ in gcc_assert)."""
import sys
from pathlib import Path
D = Path(sys.argv[1])
def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    assert old.endswith('\n') and new.endswith('\n')
    nxt = s[:s.index(old) + len(old)].count('\n') + 1
    # line numbers are computed on the file as it is now; edits to one file must be listed top to bottom is NOT
    # required, because every earlier edit already restored the numbering below it with its own #line.
    p.write_text(s.replace(old, new + '#line %d\n' % nxt))

# 1. df-scan.c, df_insn_rescan. df_insn_refs_verify collects the insn's new refs with the insn's CURRENT record
#    (DF_INSN_INFO_GET) as their insn_info. When the refs changed, df_insn_rescan then frees that record
#    (df_insn_delete -> pool_free, "df_scan insn" pool) and creates a new one (df_insn_create_insn_record ->
#    pool_alloc), and installs the collected refs, which still point to the freed record. Stock works because the
#    free list is LIFO, so the new record is the freed element. Fix: point the collected refs at the new record.
edit('df-scan.c', '''      luid = DF_INSN_LUID (insn);
      df_insn_delete (NULL, uid);
      df_insn_create_insn_record (insn);
      DF_INSN_LUID (insn) = luid;
''', '''      luid = DF_INSN_LUID (insn);
      df_insn_delete (NULL, uid);
      df_insn_create_insn_record (insn);
      DF_INSN_LUID (insn) = luid;
      {
	/* The refs collected by df_insn_refs_verify name the record that
	   df_insn_delete just freed; name the new one.  */
	struct df_insn_info *new_info = DF_INSN_INFO_GET (insn);
	unsigned int stale_ix;
	df_ref stale_ref;
	for (stale_ix = 0; VEC_iterate (df_ref, collection_rec.def_vec, stale_ix, stale_ref); stale_ix++)
	  DF_REF_INSN_INFO (stale_ref) = new_info;
	for (stale_ix = 0; VEC_iterate (df_ref, collection_rec.use_vec, stale_ix, stale_ref); stale_ix++)
	  DF_REF_INSN_INFO (stale_ref) = new_info;
	for (stale_ix = 0; VEC_iterate (df_ref, collection_rec.eq_use_vec, stale_ix, stale_ref); stale_ix++)
	  DF_REF_INSN_INFO (stale_ref) = new_info;
      }
''')
print('stale uses fixed')
