#!/usr/bin/env python3
"""cats.py CAPMIX MAP [PAGES]: kernel instructions of mark window 0 in categories: memcg/cgroup accounting, debug checks
(DEBUG_VM/LIST/ATOMIC_SLEEP), native pin path (GUP + the module), page zeroing, other; per page if PAGES given.
Category membership is by symbol NAME (patterns below), so a symbol that does both is counted once, under the first match."""
import sys, re, importlib.util
spec = importlib.util.spec_from_file_location('sd', __file__.replace('cats.py', 'symdiff.py'))
src = open(spec.origin).read().split('n, b = load(')[0]
g = {}; exec(src, g)
ker = g['load'](sys.argv[1], sys.argv[2])
pages = int(sys.argv[3]) if len(sys.argv) > 3 else 0
C = [('memcg', r'memcg|mem_cgroup|cgroup_rstat|lruvec|charge|folio_memcg|unlock_page_memcg'),
     ('debug', r'^(__list_add_valid|__list_del_entry_valid|bad_range|__might_resched|__might_sleep|__might_fault|check_new_pages|free_pcp_prepare|do_raw_spin_lock|do_raw_spin_unlock|do_raw_spin_trylock|debug_.*|__debug_.*|page_expected_state|free_page_is_bad|check_free_page)$'),
     ('native-pin', r'^(retire|resolve|frame_insert|frame_slots|pin_ahead|follow_page_mask|__get_user_pages|try_grab_page|gup_put_folio.*|sanity_check_pinned_pages|unpin_user_pages_dirty_lock|pin_user_pages_remote|__gup_longterm_locked|follow_page_pte|vm_normal_page|PageHuge|pud_huge|pmd_huge|folio_mark_dirty|faultin_page|is_valid_gup_args|stream_window|cv_.*|unpin_user_page|__get_user_pages_remote|follow_pmd_mask.*|follow_pud_mask.*|follow_p4d_mask.*|gup_.*)$'),
     ('zeroing', r'^(memset|clear_page)$')]
tot = {k: 0 for k, _ in C}; tot['other'] = 0; top = {k: [] for k in tot}
for s, n in ker.items():
    for k, p in C:
        if re.search(p, s): tot[k] += n; top[k].append((n, s)); break
    else: tot['other'] += n; top['other'].append((n, s))
T = sum(tot.values())
for k, v in tot.items():
    per = f' {v/pages:8.0f}/page' if pages else ''
    print(f'{k:>11} {v:>14,} {100*v/T:5.1f}%{per}   ' + ', '.join(f'{s}:{n/1e6:.1f}M' for n, s in sorted(top[k], reverse=True)[:5]))
print(f'{"total":>11} {T:>14,}' + (f'        {T/pages:8.0f}/page' if pages else ''))
