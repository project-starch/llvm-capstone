# memcached allocator 06/07: the carve inside a Sublet-issued slab item (`sublet-carve`), 2026-10-09

Under the slab Sublet port alone (column 3, mode 1) cases 6 and 7 complete: the suffix write and
the key read stay inside one item, and the port's region is the item. Built with MC_CARVE_BOUNDS,
`mc_carve` narrows ITEM_suffix and ITEM_key from the item's Sublet alias -- the carve inside the
item, ported -- and run in mode 1: **both fault at the labelled probe, as pre-registered
(a28d6c33fd97)** -- 6 with cause 7 at the write probe (the empty suffix region; Capstone takes the
byte before it), 7 with cause 5 at the read probe (the 9-byte key region). The catch is that bound;
revocation stays the item's as a whole.
