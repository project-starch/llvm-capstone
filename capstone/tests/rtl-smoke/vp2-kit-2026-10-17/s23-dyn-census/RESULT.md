# S-23: the CAPSTONE_DYN ops in the images sql0-drt and pb10-drt execute inside a context (static census, 2026-10-19)

**Question** (the runtime lane, for S-23's surviving route): an application lifetime op, issued speculatively inside a
context and flushed by an escape, whose rev-node walk outlives the flush (ex_stage.sv:1084, rev_owed_q survives a flush)
and clears the caller's CPMP tracker by index. That route needs the field programs to contain such ops.

**Answer:** sql0's image contains node-invalidating ops: **4 CREVOKE**, all in the virtual runtime's allocator (malloc,
realloc, __capstone_malloc_prepare_free, __capstone_malloc_nested_release). This census does not exclude the route.
It is static presence, not execution, and says nothing of timing.

## Method

- **Op list:** taken from decoder.sv's own cases at capstone-ariane **8522d72fe**, every encoding routed to fu
  CAPSTONE_DYN. Opcode custom-2 0x5b:
  - funct3 1, by funct7: REVOKE 0x00, TIGHTEN 0x02, DELIN 0x03, LCC 0x04, SPLIT 0x06, MREV 0x08, DROP 0x0b, CALL 0x20,
    RETURN 0x21, CSSUPERVISE 0x22, CSRUNV 0x24, CDERIVE 0x51, CREVOKE 0x52, CSRETIRE 0x53, CSNODE 0x54;
  - funct3 3: LDC;
  - funct3 4: STC.
- **Counting:** by raw 32-bit word and field decode, never by mnemonic. The ops are .insn-encoded and disassemble as
  `<unknown>`. Instruction boundaries come from llvm-objdump; every image here is all 32-bit (no RVC). Tool:
  `dyn-census.py <llvm-objdump> <image>...`, beside this file. It prints "N instructions parsed" and exits 2 on a no-parse.
- **Channels:** each op's rev-node channels, read from its own handler in capstone_dyn_unit.anvil at 8522d72fe
  (`send rev_node_ep.<ch>_req`, comments excluded):
  - REVOKE: query :67, rev :72;
  - DROP: drop :41;
  - CREVOKE: query :319 and :323, kill :330;
  - CSRETIRE: retire :625;
  - CDERIVE: query :268, derive :281;
  - MREV: deepen :132 and :146, mrev :153;
  - SPLIT: query :198, init :220;
  - DELIN: delin :808.
  - **Node-invalidating: REVOKE, DROP, CREVOKE, CSRETIRE. CDERIVE is NOT one.** A first version of this table, sent as
    a message, marked CDERIVE invalidating: a regex had matched the CREVOKE comment at :303-304, which sits inside
    CDERIVE's text range. The RTL lane's reading of the handlers corrected it; re-read here from the source.

## Result (sha256/16; instructions parsed; DYN ops present, all others 0)

| image | parsed | LDC | STC | LCC | CDERIVE | CREVOKE | DELIN | node-invalidating |
|---|---|---|---|---|---|---|---|---|
| CONTROL contract.dom 125705371837ea92 (v9; v8 f2017064 identical counts) | 33,337 | 1,807 | 1,546 | 157 | 22 | 5 | 5 | CREVOKE 5 |
| sql0-drt postgres.dom 64010720f1a6b3bc (the only staged image; runtime + musl static) | 2,949,460 | 222,670 | 177,437 | 154 | 12 | 4 | 4 | CREVOKE 4 |
| pb10-drt defect-00 df5eea02db56843a | 39,847 | 2,444 | 1,893 | 158 | 18 | 7 | 4 | CREVOKE 7 |
| pb10-drt defect-01 0f1bfd544ecc559d | 39,617 | 2,421 | 1,876 | 158 | 18 | 7 | 4 | CREVOKE 7 |
| pb10-drt defect-02 84fbc193c1709b0a | 39,576 | 2,420 | 1,871 | 158 | 18 | 7 | 4 | CREVOKE 7 |
| pb10-drt defect-03 d6666b4618df1d6e | 39,674 | 2,420 | 1,877 | 158 | 18 | 7 | 4 | CREVOKE 7 |
| pb10-drt defect-90 f7ffffa49e2e53fe (controls bin) | 39,447 | 2,409 | 1,868 | 158 | 18 | 7 | 4 | CREVOKE 7 |
| pb10-drt controls.dom 491655742d378fcd | 27,868 | 1,587 | 1,329 | 127 | 8 | 3 | 3 | CREVOKE 3 |

The control reproduces the 2026-10-18 contract.dom decode (22 CDERIVE, 5 CREVOKE). REVOKE, DROP, CSRETIRE, CSNODE, MREV,
SPLIT, TIGHTEN, CALL, RETURN, CSSUPERVISE and CSRUNV are 0 in every image.

- **sql0:** every CDERIVE and CREVOKE site sits in the virtual runtime's allocator. Three DELIN sites do too, and the
  fourth is in mmap. The image's .L labels were discarded, so the names below are exact.
  Read by `dyn-sites.py`, which uses the same word decode and attributes each site to the enclosing symbol:
  - CDERIVE: malloc 2, realloc 2, __capstone_malloc_group_{attach,map,nested,remap} 2 each;
  - CREVOKE: malloc 1, realloc 1, __capstone_malloc_prepare_free 1, __capstone_malloc_nested_release 1;
  - DELIN: __capstone_malloc_group_{attach,map,remap} 1 each, mmap 1.
- **pb10:** besides the runtime's sites, the Sublet pool API itself is present. In defect-00, capstone_cap_derive holds 4 CDERIVE
  and capstone_cap_revoke_child holds 2 CREVOKE. The rest attribute to unresolved .L labels, since this image keeps them.
