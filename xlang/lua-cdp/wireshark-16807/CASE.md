# Wireshark #16807 — Lua `TvbRange` ⟷ C `tvbuff` use-after-free

**One line.** A Lua dissector caches a `TvbRange` (`buffer(0,16)`) in a global
table; the C dissection engine frees the underlying `tvbuff` when the packet is
re-dissected, but the surviving Lua handle is reused one crossing later and its
dangling `tvbuff` is fed into `proto_tree_add_item_new` → UAF read.

## Identity

| | |
|---|---|
| Library | Wireshark / `wslua` (the built-in Lua dissector API) |
| Language pair | **C ⟷ Lua** (Wireshark's `libwireshark` embeds Lua; here Lua 5.4). Genuine cross-language: the coupled resource is a real C engine object, not FFI plumbing. |
| Upstream | https://gitlab.com/wireshark/wireshark/-/issues/16807 |
| Reported against | Wireshark 3.2.6 (Lua 5.2.4); reproduced here on stock **apt tshark 4.6.4** (Lua 5.4) |
| Native dep | `libwireshark` (apt); no from-source build |
| Detection | valgrind memcheck (apt binary is not ASan-built) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the `TvbRange` userdata from `buffer(0,16)`, stored in the
   global `ProtocolState[packet_id][field]` table (survives GC).
2. **Separate native resource:** the C `tvbuff` struct (valgrind: a 72-byte
   `g_malloc` block) created by `tvb_new_subset_remaining` for the TCP payload,
   owned and freed by the dissection engine.

Two distinct allocations — the Lua userdata *points at* a separately-allocated C
buffer. **Not** the borrowed-pointer-into-a-Lua-string shape.

**Direction:** native-frees. The C engine frees the `tvbuff`
(`epan_dissect_reset` → `tvb_free_chain`); the Lua `TvbRange` outlives it and is
dereferenced on the next dissection.

## Why it needs re-dissection (and how we drive it headlessly)

On first dissection `handle_packet()` stashes the range and uses it in the *same*
call, while the `tvbuff` is still live — no fault. The fault needs the *same
packet dissected again*: the GUI does this when you switch between packets; we do
it headlessly with two-pass analysis (`tshark -2`), which re-runs the dissector
on each packet after pass 1 has freed that pass's tvbuffs. Same re-dissection
path, no GUI required.

## Reproduction status

**REPRODUCED (2026-08-03), with two controls.**

- **A** `trigger.lua` + `-2`: valgrind **Invalid read** (size 1 and 4) in
  `tvb_ensure_bytes_exist` ← `proto_tree_add_item_new` (the wslua
  `TreeItem:add(field,range)` path), the block freed by `tvb_free_chain` ←
  `epan_dissect_reset`, allocated by `tvb_new_subset_remaining` ←
  `dissect_tcp_payload`.
- **B** `trigger.lua` single-pass: **clean** (no re-dissection → no stale reuse).
- **C** `trigger_fixed.lua` + `-2`: **clean** (fills the field from the live
  `buffer` each time; nothing outlives its `tvbuff`).
- Full trace + controls in `evidence.txt`.

The issue's reported crash site `tvb_offset_from_real_beginning` is reached from
the *same* `proto_tree_add_item_new` use-site; 4.6.4's leaf within that add path
is `tvb_ensure_bytes_exist`. Same coupled object, free-site and use-site.

## PASS signature

`run.sh` passes iff **A** produces the tvb-UAF stack signature (`Invalid read` +
`tvb_ensure_bytes_exist` + freed-by `tvb_free_chain`/`epan_dissect_reset`) AND
both controls **B** and **C** do **not**. Any one wrong = FAIL. The check keys on
the stack signature, not valgrind's exit code (unrelated plugin-init warnings
pollute the code in every run).
