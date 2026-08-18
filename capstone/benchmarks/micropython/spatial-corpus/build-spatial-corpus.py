#!/usr/bin/env python3
"""Build the MicroPython SPATIAL-allocator corpus CSV.

Sibling of ../temporal-corpus, same discipline and deliberately NOT merged into
it: that corpus is a finished, published artifact about lifetime defects, and
rewriting it to hold a second question would invalidate every number already
quoted from it.

Title, state, date and URL are COPIED from a verified GitHub GraphQL response
(github-issues.json), never typed here. Only the classification columns are
authored, and `is_spatial` was set by READING each report body
(github-bodies.json), not its title -- reading the temporal corpus's class column
off titles is a mistake that audit had to undo for fourteen rows.

WHY A SPATIAL CORPUS AT ALL. The temporal corpus shows Capstone catching nothing,
because gc_free never reaches the hardware. The prediction here is sharper and
falsifiable per row, because scope decides it:

  gc-heap       the block is one offset into the single heap array, so an
                overflow into a neighbouring block is INSIDE the capability
                -> predicted NOT trapped, or trapped only at the heap boundary
                   after everything in between is corrupted
  stack         the domain stack carries its own capability
                -> predicted TRAPPED
  static-global under -capstone-gp-captable each global is carved separately
                -> predicted TRAPPED

So this corpus has its positive controls built in: if the stack and static rows
also came back untrapped, the instrument would be broken, not the target.
"""
import csv, json, os

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "spatial-allocator-corpus.csv")
gql = json.load(open(os.path.join(S, "github-issues.json")))["data"]["repository"]
issues = {v["number"]: v for v in gql.values() if v}

# id, ref, class, cwe, component, scope, trigger, is_spatial, spatial_evidence, reach, notes
ROWS = [
 ("MPY-S01", 19314, "heap-overflow", "CWE-787,CWE-190",
  "py/objstr.c,py/objlist.c,py/objtuple.c,py/sequence.c", "gc-heap",
  "seq * n sized in unchecked size_t; 8 * 2**61 wraps to 0, so the allocation is empty and the write is not",
  "yes", "body: 'the allocation is tiny/zero while the write is enormous -> heap buffer overflow'",
  "measured", "OPEN at the pin, so no parent build was needed. Reproduces on stock 2e3304a for str, bytes, list and tuple."),
 ("MPY-S02", 13041, "heap-overflow", "CWE-787", "py/objint.c:int_to_bytes", "gc-heap",
  "int.to_bytes(0, 'big'): the guard is `len < 0`, so length 0 passes and the write is unbounded",
  "yes", "body: 'the len is 0, thus it pass through if (len < 0)' plus a segfault against CPython's OverflowError",
  "parent-build", "Pure py/, three lines of Python, no OS. The cheapest remaining spatial reproduction."),
 ("MPY-S03", 13007, "heap-overread", "CWE-125", "py/objslice.c:slice_indices", "gc-heap",
  "slice.indices(0.0): a float object is read as an integer, so the read runs 8 bytes past a 16-byte object",
  "yes", "body gives the object extent [0x7fffef005760,0x7fffef005770) and the access at +0x8 past it",
  "parent-build", "Needs float objects, which this domain now has."),
 ("MPY-S04", 13220, "heap-overread", "CWE-125", "lib/re1.5/compilecode.c:68", "gc-heap",
  "an unterminated character class: the `for (cnt = 0; *re != ']'; ...)` loop walks off the pattern",
  "yes", "body quotes the loop and marks the two unchecked advances",
  "parent-build", "modre is compiled into this domain, so no new module is needed."),
 ("MPY-S05", 15271, "heap-overflow", "CWE-787", "py/objarray.c", "gc-heap",
  "array.append after a grow that raised MemoryError: the free-slot count was incremented before the allocation",
  "yes", "PR body: 'the next attempt to grow the buffer will cause a buffer overflow'",
  "parent-build", "Reachable without any module: fill the heap, catch MemoryError, append again."),
 ("MPY-S06", 13005, "heap-overflow", "CWE-787", "py/moductypes.c", "gc-heap",
  "uctypes.struct laid over a bytearray shorter than the descriptor",
  "yes", "body walks the 1-byte bytearray into a descriptor that reads more",
  "blocked-uctypes", "uctypes is architecturally unusable in this domain, measured: 15 of its 16 tests fault."),
 ("MPY-S07", 13428, "heap-overflow", "CWE-787", "py/binary.c:155", "gc-heap",
  "a block device whose writeblocks reads buf[len(buf)]",
  "yes", "reporter's ASan trace pins py/binary.c:155", "blocked-vfs", "Needs VfsLfs2."),
 ("MPY-S08", 13046, "heap-overflow", "CWE-787,CWE-190", "py/stream.c:121", "gc-heap",
  "mp_stream_rw checks an unsigned size with > 0, so the integer overflow reaches the copy",
  "yes", "body names the check and the resulting overflow", "blocked-vfs", "Needs a VFS and a block device."),
 ("MPY-S09", 13006, "heap-overflow", "CWE-787", "extmod/vfs.c:276", "gc-heap",
  "umount compares mount strings using the length of the unmount string",
  "yes", "body distinguishes the global-buffer-overflow and heap-buffer-overflow cases",
  "blocked-vfs", "Reported as producing BOTH a global and a heap overflow depending on mount order."),
 ("MPY-S10", 13004, "heap-overread", "CWE-125", "py/objfun.c:135", "gc-heap",
  ".mpy with an out-of-range qstr reference; qstr_table is indexed without a length check",
  "yes", "body: 'there is no length check on accessing qstr_table'", "blocked-mpy",
  "MICROPY_PERSISTENT_CODE_LOAD is 0 here."),
 ("MPY-S11", 13003, "heap-overread", "CWE-125", "py/persistentcode.c:134", "gc-heap",
  "native rodata relocation dereferences 8 bytes at an unvalidated offset",
  "yes", "body: 'it tries to dereference(load) addr_to_adjust by 8 bytes'", "blocked-mpy", ""),
 ("MPY-S12", 13002, "heap-overread", "CWE-125", "py/persistentcode.c", "gc-heap",
  "malformed .mpy: 15 bytes read from a 3-byte code object",
  "yes", "body: 'the valid code length is just 3 bytes, but it tries to read 15 bytes'", "blocked-mpy", ""),
 ("MPY-S13", 18637, "stack-overflow", "CWE-787", "py/vm.c:373", "stack",
  "malformed bytecode sizes the VLA value stack; the write lands past it",
  "yes", "ASan dynamic-stack-buffer-overflow WRITE, shadow bytes ca/cb quoted in the body",
  "blocked-mpy", "CONTRAST ROW. The domain stack carries its own capability, so this is predicted TRAPPED."),
 ("MPY-S14", 17852, "static-overread", "CWE-125", "py/stream.c:102", "static-global",
  "an object with no stream protocol passed where one is assumed; the read walks a static table",
  "yes", "ASan global-buffer-overflow READ of size 8, py/stream.c:102, quoted in the body",
  "blocked-modules", "CONTRAST ROW. Needs ssl and machine. Predicted TRAPPED: gp-captable carves each global."),
 ("MPY-S15", 12532, "static-overread", "CWE-125", "py/stream.c", "static-global",
  "same defect as MPY-S14, reported separately and still open",
  "yes", "title and body both name mp_get_stream_raise and a global buffer overflow",
  "blocked-modules", "CONTRAST ROW. Kept because it is the independently reported instance of MPY-S14."),
 ("MPY-S16", 12587, "static-overflow", "CWE-787", "extmod/vfs.c", "static-global",
  "os.umount('WjeePMR1E'): a non-path string parsed as a path",
  "yes", "body: 'buffer out-of-bound crash', ASan global-buffer-overflow",
  "blocked-vfs", "CONTRAST ROW."),
 ("MPY-S17", 7246, "static-overread", "CWE-125", "lib/oofatfs/ff.c:2824", "static-global",
  "create_name reads one byte past a static table",
  "yes", "ASan: 'global-buffer-overflow ... READ of size 1 ... ff.c:2824 in create_name'",
  "blocked-vfs", "CONTRAST ROW, and note the scope: ASan says GLOBAL, not heap."),
 ("MPY-S18", 12528, "heap-overflow", "CWE-787", "py/mpz.c:mpz_as_bytes", "gc-heap",
  "a negative size reaches memset",
  "yes", "body: 'manifested as heap-buffer-overflow in an older commit'",
  "blocked-mpz", "MICROPY_LONGINT_IMPL is NONE here, so there is no mpz."),
 ("MPY-S19", 7860, "heap-overflow", "CWE-787", "extmod/modframebuf.c", "gc-heap",
  "a stride larger than the buffer justifies", "yes",
  "title and body: the constructor does not validate stride against the buffer length",
  "not-reproducible-at-pin", "framebuf IS in this domain, but the pin already raises on every stride form tried."),
 ("MPY-S20", 19431, "heap-overflow", "CWE-787,CWE-190", "extmod/modframebuf.c", "gc-heap",
  "x + y * stride computed in 16-bit int", "yes",
  "body: fourteen index expressions, out-of-bounds from y >= 128 on a 256-wide buffer",
  "not-applicable", "16-bit-int targets only. capstone64 has 32-bit int, so the overflow cannot occur."),
 ("MPY-S21", 599, "heap-overflow", "CWE-787", "py/emit*.c:emit_write_bytecode_byte_int", "gc-heap",
  "a 10-byte buffer written past its end on 64-bit", "yes",
  "body: 'has a buf of 10 bytes (9 of which are used in this case) but emit[s more]'",
  "too-old", "2014. A parent build of that tree is not worth the toolchain archaeology."),
 # --- audited and NOT certain. Kept visible rather than dropped, with the reason. ---
 ("MPY-S22", 3090, "bounds-arithmetic", "CWE-125", "py/sequence.c", "gc-heap",
  "negative-step slice indices computed with an inclusive stop", "uncertain",
  "the PR fixes the index arithmetic; neither it nor the thread states that an out-of-bounds ACCESS occurred",
  "excluded", "Correctness fix that MAY be spatial. Not counted."),
 ("MPY-S23", 12702, "mixed", "CWE-125,CWE-476", "py/moductypes.c", "gc-heap", 
  "uctypes.sizeof() on various descriptors", "uncertain",
  "body: 'exhibited in global-buffer-overflow sometimes or [null dereference]' -- two different defects in one report",
  "excluded", "Two outcomes, one report. Not counted."),
 ("MPY-S24", 12660, "unknown", "", "py/mpz.c:mpz_hash", "gc-heap",
  "uctypes.struct reaching mpz_hash", "uncertain",
  "body reports a crash; it does not establish an out-of-bounds access",
  "excluded", "Not counted."),
 ("MPY-S25", 12871, "mixed", "CWE-125,CWE-476", "py/pairheap.c", "static-global",
  "asyncio.Event() paths into mp_pairheap_delete", "uncertain",
  "body: 'exhibited in either global-buffer-overflow or null-dereference'",
  "excluded", "Two outcomes, one report, and asyncio is absent here. Not counted."),
 ("MPY-S26", 12776, "mixed", "CWE-125,CWE-476", "extmod/modasyncio.c", "static-global",
  "pushing a module object to asyncio.TaskQueue()", "uncertain",
  "body: 'null-dereference when the passed object was a value and global-buffer-overflow when it was a module'",
  "excluded", "Not counted."),
 ("MPY-S27", 18752, "not-spatial", "", "py/objstr.c:str.center", "n/a",
  "str.center() counts bytes rather than characters", "no",
  "PR body describes INCORRECT PADDING, not an out-of-bounds access",
  "excluded", "Matched the search on the word overflow only. Not a spatial defect."),
 ("MPY-S28", 6365, "not-a-defect", "", "extmod/modframebuf.c", "n/a",
  "a feature request for blit source rectangles", "no",
  "labelled enhancement; no defect is reported", "excluded", "Search artefact."),
 ("MPY-S29", 533, "not-a-defect", "", "py/gc.c", "n/a",
  "a proposal to add stray-pointer checking to the collector", "no",
  "labelled enhancement; no defect is reported", "excluded", "Search artefact."),
 ("MPY-S30", 9045, "not-a-defect", "", "extmod/modframebuf.c", "n/a",
  "a PR adding ellipse and poly methods", "no",
  "a feature PR; no defect is reported", "excluded", "Search artefact."),
]

# What an UNMODIFIED MicroPython gets from Capstone, per scope. gc-heap follows from
# a measurement (../temporal-corpus/evidence/spatial-gap-same-mechanism-2026-08-18.txt);
# the other two are predictions this corpus exists to test.
PREDICT = {"gc-heap": "no-or-at-region-boundary", "stack": "yes", "static-global": "yes", "n/a": ""}

# Measured in the domain. domain_behaviour vocabulary is deliberately the temporal
# corpus's, plus fault-cause7, which is the bounds fault and did not occur there.
MEASURED = {
    # 2026-08-18, one boot, resumable suite, EXTRA feature level, MPY_FLOAT_CORE=1.
    # Sanity PASSED in the same image (0x00077724), so the three faults are results
    # and not a domain that failed to start -- an earlier attempt at this run used the
    # WRONG guest runner, domain creation returned -1, and all three "faults" were the
    # monitor's, at pc 0x8001b10e with cause 5. Check `Created domain ID = 0`.
    19314: ("measured", "crash-sigsegv", "fault-cause7"),
}

def main():
    out = [["id","source","ref","url","title","state","first_seen","class","cwe","component",
            "scope","trigger","is_spatial","spatial_evidence","predicted_trap","reach",
            "repro_status","stock_behaviour","domain_behaviour","notes"]]
    for rid, ref, cls, cwe, comp, scope, trig, isp, ev, reach, notes in ROWS:
        iss = issues[ref]
        repro, stock, dom = MEASURED.get(ref, ("none", "not-run", "not-run"))
        out.append([rid, "github", f"#{ref}", iss["url"], iss["title"], iss["state"],
                    iss["createdAt"][:10], cls, cwe, comp, scope, trig, isp, ev,
                    PREDICT[scope], reach, repro, stock, dom, notes])
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"{len(out)-1} Zeilen -> {OUT}")
    for col in ("is_spatial", "scope", "reach", "domain_behaviour"):
        i = out[0].index(col)
        tally = {}
        for r in out[1:]:
            tally[r[i]] = tally.get(r[i], 0) + 1
        print(f"  {col:18} {tally}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
