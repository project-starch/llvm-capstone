# Sublet reference model

This document specifies the Sublet processor extension for the RTL
implementation in the CVA6-based core, rule by rule, with the source location
of each rule in the executable reference model.

## 1. Purpose and source of truth

The reference model is capstone-qemu at commit `9289114e06` (branch
`address-space-root`), which the submodule `capstone/capstone-qemu` of this
branch pins. Every `file:line` reference is relative to the capstone-qemu
repository root at that commit. The code is normative. Where this text and the
code differ, the code wins.

The evaluation runs `-M virt -smp 1` with `x-capstone-trap-route=true`,
`x-capstone-supervisor-refs=true` and `x-capstone-raw-moves=true`
(`capstone/runtime/virtual/trap_platform.py:88-96`, `:241` in this repository).
`x-capstone-exact-bounds` keeps its default, false (`target/riscv/cpu.c:2021`).
This document describes that configuration. The CPU realizes only with one
possible hart (`target/riscv/cpu.c:1685-1700`).

A *reference* is a 128-bit value with its tag set. The code calls it a
capability. A *lifetime* is one generation of a record in the lifetime table.
*Protected U* is user mode while `srevroot` is nonzero (Section 6).

## 2. Reference format

### 2.1 The word

```
 127          96 95          65 64          38 37           0
+--------------+--------------+--------------+--------------+
|  generation  |     slot     |    bounds    |   address    |
+--------------+--------------+--------------+--------------+
lo (bytes 0-7)  = bounds[25:0] << 38 | address[37:0]
hi (bytes 8-15) = generation << 32 | slot << 1 | bounds[26]
```

(`target/riscv/cap_compress.c:39-47`, `:67-90`.) The format stores no type and
no rights. The decoder gives every reference type NONLIN and rights RWX, so
every type and rights check passes and only bounds and lifetime decide an
access (`target/riscv/cap_compress.c:59-65`, `target/riscv/cap.h:187-196`).

The cursor is the zero extension of `address[37:0]`
(`target/riscv/cap_compress.c:70`, `target/riscv/cap.h:199-205`). A register
write of a reference whose cursor is at or above 2^38 writes the cursor as an
untagged integer with a zero upper doubleword instead
(`target/riscv/cap.h:207-233`). CSMINT and CBNZ raise 29 for such a cursor. satp
accepts no mode above Sv39 (`target/riscv/cpu.c:1383-1400`). Bounds have no
such limit (`target/riscv/cap.h:192-193`).

### 2.2 Bounds field

| Subfield | bounds bits | word bits |
|---|---|---|
| bE | 2:0 | 40:38 |
| b | 13:3 | 51:41 |
| tE | 16:14 | 54:52 |
| t | 25:17 | 63:55 |
| iE | 26 | 64 |

(`target/riscv/cap_compress.c:31-35`.) Decode with cursor `a`, in wrapping
64-bit arithmetic (`target/riscv/cap_compress.c:171-222`):

```
if iE: E = tE << 3 | bE   B = b << 3        T = t << 3        L = 1
else:  E = 0              B = b << 3 | bE   T = t << 3 | tE   L = 0
T[13:12] = B[13:12] + (T[11:0] < B[11:0]) + L            (mod 4)
s = 2^(E+14), or 0 when E + 14 >= 64
base = (a & ~(s-1)) | B << E          end = (a & ~(s-1)) | T << E
A3 = a[E+13:E+11] (0 when E + 11 >= 64)   B3 = B[13:11]   T3 = T[13:11]
R = (B3 - 1) mod 8
end  += s if A3 >= R and T3 < R       end  -= s if A3 < R and T3 >= R
base += s if A3 >= R and B3 < R       base -= s if A3 < R and B3 >= R
```

Encode (`target/riscv/cap_compress.c:92-145`):

```
len = end - base;  E = max(msb(len), 12) - 12     (msb: index of the top set bit)
if E == 0 and len < 4096:  iE = 0, B = base[13:0], T = end[11:0]
else:
  iE = 1; g = 2^(E+3); q = len >> (E+3); sr = base mod g; lr = len mod g
  if (q == 1023 and (sr | lr) != 0) or (q == 1022 and sr + lr > g): E = E + 1
  B = (base >> E) with bits 2:0 cleared
  T = (end >> E) with bits 2:0 cleared, plus 8 if end mod 2^(E+3) != 0
  B[2:0] = E[2:0];  T[2:0] = E[5:3]
bE = B[2:0]  b = B[13:3]  tE = T[2:0]  t = T[11:3]
```

The encoder rounds the base down and the end up. The comment at
`target/riscv/cap_compress.c:107-112` requires the RTL's `compress_bounds` to
take the same exponent step.

### 2.3 Representability

*Exact bounds.* CDERIVE and CSMINT encode the new bounds, decode them and raise
29 when base or end differs (`target/riscv/op_helper.c:1139-1147`,
`target/riscv/capstone_table.c:455-461`).

*Cursor moves.* CINCOFFSET, CINCOFFSETIMM and SCC keep the tag only if the
source's encoded bounds, decoded at the new cursor, equal those decoded at the
old cursor (`target/riscv/cap_compress.c:147-168`). Otherwise they write the
new cursor as an untagged integer and do not trap
(`target/riscv/op_helper.c:784-796`). A cursor at or above 2^38 never qualifies
(`target/riscv/cap_compress.c:157-160`).

SHRINK and SHRINKTO apply neither rule (Section 9).

The code states neither rule in closed form. We derived two equivalent forms
from the codec and checked them against `cap_compress`, `cap_uncompress` and
`capstone_cursor_representable` on random cases below 2^38 (3.9 * 10^6 and
2.4 * 10^7) with a host harness outside both repositories. Bounds of length L
at base b are exact when L < 4096, or when b and L are multiples of 8
(L < 8192) or of 2^(msb(L) - 9) (L >= 8192). Inexact bounds always decode to a
superset. With E, B and R as in the decoder, s = 2^(E+14) and
w = base - ((base - R * 2^(E+11)) mod s), a cursor c keeps the tag exactly when
(c - w) mod 2^64 < s and c < 2^38.

### 2.4 Lifetime id

The id is `generation << 31 | slot`, 63 bits (`target/riscv/cap.h:187-197`,
`target/riscv/cap_compress.c:75-76`). No instruction takes an id as an operand
(`target/riscv/cap_rev_table_abi.h:14-17`).

### 2.5 Registers and memory

Each integer register carries a tag (`target/riscv/cap.h:82-120`). Integer
instructions read a reference as its cursor (`target/riscv/translate.c:1608-1613`,
`target/riscv/cap.h:62-86`), and every integer write clears the tag
(`target/riscv/translate.c:647-668`).

The model keeps one tag per 16-byte granule of main RAM, keyed by physical
address. Other memory holds no tag (`target/riscv/cap_mem_map.h:125-131`,
`target/riscv/cpu.c:1521-1538`). Every write except an STC of a reference
clears the tags it overlaps: plain stores (`target/riscv/op_helper.c:3353-3361`),
SC and AMOs (`target/riscv/op_helper.c:3206-3218`), other bus masters
(`target/riscv/op_helper.c:3250-3261`) and table writes
(`target/riscv/capstone_table.c:107-115`). A granule under a writable device
mapping holds no tag (`target/riscv/cap_mem_map.h:104-123`).

An integer or FP load of bytes 0 to 7 of a tagged granule reads the cursor, not
the raw bits (`target/riscv/op_helper.c:3044-3132`,
`target/riscv/insn_trans/trans_rvi.c.inc:229-237`). Bytes 8 to 15 read raw, so
they reveal the id (`target/riscv/op_helper.c:1016-1022`).

## 3. Lifetime table

### 3.1 Locating the table

The S-level CSR `srevroot` (0x5c1) holds the physical address of the root page
(`target/riscv/cpu_bits.h:536`). A write clears bits 11:0
(`target/riscv/csr.c:4194-4204`). Reset writes 0 (`target/riscv/cpu.c:927`),
which names no table. The model accesses the table by physical address through
`address_space_ldq_le` and `address_space_stq_le`, without the page table or
PMP (`target/riscv/capstone_table.c:79-115`), and requires writable RAM
(`target/riscv/capstone_table.c:121-169`). The kernel builds the table.

Protected U, and S while `srevroot` is nonzero, use the table at `srevroot`.
Elsewhere the model uses an empty host tree, so CDERIVE, CREVOKE and CRENEW
raise 25 there (`target/riscv/capstone_table.c:171-213`,
`target/riscv/cap_rev_tree.h:89-93`).

### 3.2 Record

```
 127          96 95          65 64          34 33           3 2   0
+--------------+--------------+--------------+--------------+-----+
|  generation  |   previous   |     next     |    depth     |  0  |
+--------------+--------------+--------------+--------------+-----+
word 0 (bytes 0-7)  = next[29:0] << 34 | depth << 3
word 1 (bytes 8-15) = generation << 32 | previous << 1 | next[30]
```

(`target/riscv/cap_rev_table_abi.h:17-22`, `target/riscv/cap_rev_table.c:4-39`.)
previous and next hold slots, 0 for none. There is no parent field and no
valid, linear or free flag: a record is live for exactly the id {its
generation, its slot} (`target/riscv/cap_rev_table_abi.h:21-22`). A free record
holds its next generation, previous 0, depth 0 and its free-list link in next
(`target/riscv/cap_rev_table.c:54-65`).

### 3.3 Paged layout and header

The table is a tree of 4 KiB pages, and a slot splits 5:9:9:8
(`target/riscv/cap_rev_table_abi.h:37-53`, `target/riscv/cap_rev_table.c:73-106`):

```
root + 64 + 8*slot[30:26] -> directory 1;  dir1 + 8*slot[25:17] -> directory 2
dir2 + 8*slot[16:8] -> record page;        page + 16*slot[7:0]  -> record
```

A walk fails when the slot is at or above capacity, an entry is 0 or not 4 KiB
aligned, a page is not writable RAM, or a page repeats the root or a directory
above it (`target/riscv/cap_rev_table.c:75-104`).

| Offset | Field | Content (`target/riscv/cap_rev_table_abi.h:43-53`) |
|---|---|---|
| +0 | capacity, flags | bits 61:0 backed slots, a multiple of 256, at most 2^31. Bits 63 RECYCLE and 62 PAGED, both set |
| +8 | next | next never-used slot, 2 <= next <= capacity |
| +16 | free | free head in bits 31:0, free count in bits 63:32 |
| +24 | allocations | allocations and in-place CRENEWs, cumulative |
| +32 | magic | `0x3245474150444f4e` |
| +40 | retired | slots retired at the last generation |
| +48 | growth | bit 0 NOGROW, other bits zero |
| +56 | reserved | zero |
| +64 | root entries | 32 physical page addresses, 0 for none |

The processor writes only next, free, allocations, retired and records
(`target/riscv/cap_rev_table.c:234-245`, `:356-384`, `:420-440`). It never
issues slots 0 and 1 (`target/riscv/cap_rev_table_abi.h:41`). Tests also use a flat
table (PAGED clear), which never grows (`target/riscv/cap_rev_table.c:78-81`).

### 3.4 List, slots and generations

The records under one root form a doubly linked list in depth-first order: the
subtree of a record is the run that directly follows it with a greater depth
(`target/riscv/cap_rev_tree.c:153-189`). CSMINT makes a root at depth 0
(`target/riscv/cap_rev_tree.c:71-89`). CDERIVE inserts the child directly after
its parent, one level deeper (`target/riscv/cap_rev_tree.c:110-120`).

An allocation takes the free head, else the next never-used slot, and keeps the
generation it finds. A never-used slot starts at generation 0. The allocation
updates free or next and increments allocations
(`target/riscv/cap_rev_table.c:329-348`, `:234-245`).

Generations advance at release, never at allocation
(`target/riscv/cap_rev_table_abi.h:24-28`). Releasing slot s at generation g
(`target/riscv/cap_rev_table.c:356-384`):

- g + 1 < 2^32 - 1: the record becomes {g + 1, previous 0, depth 0, next = old
  free head}, and free becomes {head s, count + 1}.
- g + 1 = 2^32 - 1: the slot retires. The record becomes {2^32 - 1, zeros}, it
  never returns to the free list, and retired increments.

The processor never issues generation 2^32 - 1 and never issues an id twice
(`target/riscv/cap_rev_table_abi.h:29-32`).

### 3.5 Full table and growth

An allocation decides its resources after every other condition of its
instruction (`target/riscv/cap_rev_table.c:465-482`,
`target/riscv/cap_rev_table_abi.h:55-66`):

1. Table failed validation: 29.
2. A free or never-used slot exists: proceed.
3. No slot, and the table can grow (paged, NOGROW clear, capacity < 2^31): 30,
   before any record changes.
4. Otherwise no fault: CDERIVE and CRENEW write an untagged 0 to rd, CSMINT
   writes 0 to rd and stores nothing, no record changes.

The kernel answers 30 by publishing record pages and directory entries, then
the larger capacity, or by setting NOGROW, and resumes the instruction
(`target/riscv/cap_rev_table_abi.h:67-75`). The model reads the growth word at
every operation (`target/riscv/cap_rev_table.c:135-140`).

### 3.6 Validation and liveness

Every operation checks the header (`target/riscv/cap_rev_table.c:108-199`): a
nonzero root, capacity in [1, 2^31], next in [1, capacity], and for a paged
table RECYCLE, capacity a multiple of 256, a writable root page, the magic,
growth and +56. Then next >= 2, free count <= next - 2,
retired <= next - 2 - free count, head 0 exactly when count is 0, the first two
free entries in [2, next) with a free shape and consistent links, and the
never-used slots the operation may take reachable. After each `srevroot` write
and reset the first open also walks the whole free list
(`target/riscv/cap_rev_table.c:206-229`, `target/riscv/capstone_table.c:186-213`).
While a table fails, every lifetime instruction raises 29 and every reference
counts as stale (`target/riscv/cap_rev_table_abi.h:81-92`).

A reference {g, s} is live when the table is valid, 2 <= s < next, the walk
reaches the record, the record holds g, and g != 2^32 - 1
(`target/riscv/cap_rev_table.c:281-308`, `:388-401`, `:460-463`).

## 4. Instructions

### 4.1 Encodings and common rules

Opcode 0x5b, R-type, funct3 = 1 (`target/riscv/insn32.decode:972-978`):

| Instruction | funct7 | rd | rs1 | rs2 |
|---|---|---|---|---|
| CSMINT | 0x50 | root id (integer) | slot address | descriptor address |
| CDERIVE | 0x51 | child | parent, cursor at the start | size (integer) |
| CREVOKE | 0x52 | must be x0 | object | must be x0 |
| CRENEW | 0x54 | renewed reference | object | must be x0 |

A nonzero must-be-x0 field makes the instruction illegal, cause 2
(`target/riscv/insn_trans/trans_capstone.c.inc:100-118`). CDERIVE has no field
restriction, and none of the three checks privilege
(`target/riscv/insn_trans/trans_capstone.c.inc:94-98`). Each reads its sources
before it writes rd, so rd may equal a source
(`target/riscv/op_helper.c:1114`, `:1204`). Their causes 24 to 30 carry tval 0
(`target/riscv/cpu_helper.c:1655-1660`). The list code reads every record it
will write before the first write (`target/riscv/cap_rev_tree.h:19-21`). A
record that becomes unreachable during the instruction gives 29
(`target/riscv/cap_rev_table.c:266-279`, `target/riscv/op_helper.c:1088-1107`).

### 4.2 CDERIVE rd, rs1, rs2

```
p = x[rs1]; n = x[rs2]
if !p.tag or n.tag:                                       raise 24
if table invalid:                                         raise 29
if !live(p.id):                                           raise 25
if n == 0 or !(p.base <= p.cursor <= p.end and n <= p.end - p.cursor): raise 29
c = {cursor = p.cursor, base = p.cursor, end = p.cursor + n}
if decode(encode(c)) != (c.base, c.end):                  raise 29
resource decision (3.5): 30, or x[rd] = untagged 0 and stop
q = record of p.id
if q.next != 0 and record q.next does not link back:      raise 29
if q.depth == 2^31 - 1:                                   raise 30
s = allocate; record s = {prev = slot(p), next = q.next, depth = q.depth + 1}
record q.next: prev = s;  q.next = s
if rd != x0: x[rd] = c with id {generation(s), s}
```

(`target/riscv/op_helper.c:1111-1158`, `target/riscv/cap_rev_tree.c:91-123`.)
With rd = x0 CDERIVE still makes the record.

### 4.3 CREVOKE rs1

```
o = x[rs1]
if !o.tag:                                                raise 24
if table invalid:                                         raise 29
if !live(o.id):                                           raise 25
r = record of o.id
read r.prev, the run after r with depth > r.depth and the record after the
run; a missing record or more than next visits:           raise 29
release each record of the run in list order (3.4)
r.prev.next = after; after.prev = r.prev; release r
```

(`target/riscv/op_helper.c:1169-1181`, `target/riscv/cap_rev_tree.c:130-226`.)
CREVOKE needs no parent and no right, and it ends a CSMINT root too
(`target/riscv/cap_rev_table.c:295-298`).

### 4.4 CRENEW rd, rs1

```
o = x[rs1]
if !o.tag:                                                raise 24
if table invalid:                                         raise 29
if !live(o.id):                                           raise 25
r = record of o.id
in_place = generation(r) + 1 < 2^32 - 1
if !in_place: resource decision (3.5): 30, or x[rd] = untagged 0 and stop
check the run as CREVOKE does:                            raise 29
if in_place:
    release the run; r.next = after; after.prev = r
    generation(r) += 1; allocations += 1
else:
    s = allocate, before the run's release; release the run
    record s = {prev = r.prev, next = after, depth = r.depth}
    r.prev.next = s; after.prev = s; release r (it retires)
if rd != x0: x[rd] = o with the new id
```

(`target/riscv/op_helper.c:1202-1229`, `target/riscv/cap_rev_tree.c:238-308`,
`target/riscv/cap_rev_table.c:420-440`.) The result keeps the operand's cursor
and bounds. With rd = rs1 a failure overwrites the register's reference, and the
old lifetime stays live (`target/riscv/op_helper.c:1196-1201`).

### 4.5 CSMINT rd, rs1, rs2

CSMINT makes the kernel's root reference. It is illegal in U and with x2 as an
operand (`target/riscv/insn_trans/trans_capstone.c.inc:387-404`). rs2 points to
six doublewords: base, end, `0x10f`, cursor, a second slot or 0, and 0. In order
(`target/riscv/capstone_table.c:387-500`): descriptor misaligned (4), the
descriptor loads, slot misaligned (6) or not RAM (7), the same for the second
slot, then 29 with tval 0 for base >= end, a third word other than `0x10f`, a
nonzero last word, a cursor outside [base, end] or at or above 2^38, or inexact
bounds. It then takes the resource decision on the table at `srevroot` in any
mode, makes one record at depth 0, stores the tagged reference in each slot and
writes the id to rd.

### 4.6 Other instructions on references

| Instruction | Encoding (0x5b) | Behavior | Source (`target/riscv/`) |
|---|---|---|---|
| MOVC rd, rs1 | R, f3 1, f7 0x0a | copies value and tag, the source stays | `op_helper.c:721-760` |
| CINCOFFSET rd, rs1, rs2 | R, f3 1, f7 0x0c | cursor += x[rs2]. Tagged rs2: 24. Untagged rs1: integer sum | `op_helper.c:807-848` |
| CINCOFFSETIMM rd, rs1, imm | I, f3 2 | as CINCOFFSET with the immediate | `op_helper.c:850-867` |
| SCC rd, rs1, rs2 | R, f3 1, f7 0x05 | cursor = x[rs2], same rules | `op_helper.c:869-905` |
| LCC rd, rs1, sel | R, f3 1, f7 0x04, sel in rs2 | 0: 1. 1: 1, or 7 if untagged. 2 cursor, 3 base, 4 end, 5: 7. 6, 7: 26. 8: id in S with supervisor refs, else 0. Untagged: 24 except sel 1 | `op_helper.c:916-1040` |
| SHRINK rd, rs1, rs2 | R, f3 1, f7 0x01 | x[rd] bounds = [x[rs1], x[rs2]), cursor clamped. 24, 29 | `op_helper.c:1231-1320` |
| SHRINKTO rd, rs1, imm | I, f3 0 | x[rd] = x[rs1] with bounds [cursor, cursor + imm). 26 | `op_helper.c:1322-1342` |
| LDC rd, imm(rs1) | I, f3 3 | 128 bits and tag (Section 5) | `op_helper.c:1738-1797`, `:1863-1992` |
| STC rs2, imm(rs1) | S, f3 4 | 128 bits, tag set when x[rs2] is a reference and the granule takes tags | `op_helper.c:2930-3030`, `:3149-3165` |
| CBNZ rd, imm(rs1) | I, f3 6 | if x[rs1] != 0 jump to x[rd] + imm. Untagged x[rd]: 24. Target >= 2^38: 29 | `insn_trans/trans_capstone.c.inc:359-382`, `op_helper.c:3391-3399` |

(`target/riscv/insn32.decode:957-970`.) Instructions that would write a type or
a right, and CSRETIRE, do not decode
(`target/riscv/insn_trans/trans_capstone.c.inc:26-29`).

## 5. The access check

In protected U every integer and FP load and store, LR, SC, AMO, LDC and STC
uses its base register as a reference (`target/riscv/translate.c:130-139`,
`target/riscv/insn_trans/trans_rvi.c.inc:291-310`, `:401-419`,
`target/riscv/insn_trans/trans_rva.c.inc:24-37`,
`target/riscv/insn_trans/trans_rvf.c.inc:51`, `:82`). Vector loads and stores,
cbo, Zcmp and the T-Head memory decoders are illegal there
(`target/riscv/translate.c:1431-1450`,
`target/riscv/insn_trans/trans_rvzicbo.c.inc:31-70`,
`target/riscv/insn_trans/trans_rvzce.c.inc:170-176`, `:219-225`).

```
a = x[rs1] + sext(imm)            (the tval of 24, 25 and 28)
if !x[rs1].tag:                                   raise 24
if !live(x[rs1].id):                              raise 25
if LDC or STC and a mod 16 != 0:                  raise 4 (load) or 6 (store)
if !(base <= a and a <= end and size <= end - a): raise 28
translate a with the page table and PMP:          13, 15, 5 or 7
```

(`target/riscv/op_helper.c:1436-1611`, `target/riscv/cap.h:138-143`.) The type
and rights checks of `target/riscv/op_helper.c:1495-1522` never fire. LR, SC and
AMO check tag, lifetime and bounds, then SC and AMO probe the store
translation, then all three check alignment, 4 for LR and 6 otherwise
(`target/riscv/op_helper.c:1638-1706`). The inline path at
`target/riscv/translate.c:160-261` performs the same checks.

The translation uses U's page table in protected U and S's with SUM for S
through a reference (`target/riscv/op_helper.c:1378-1384`). The page table
alone grants read, write and execute (`target/riscv/cap.h:192-195`).

LDC returns the granule's tag. For a tagged granule it takes bounds and cursor
from the copy the tag map keeps beside the tag, not from the stored word
(`target/riscv/op_helper.c:1750-1754`, `target/riscv/cap_mem_map.h:17-22`). A
stale reference loads tagged, and only its use faults
(`target/riscv/op_helper.c:1755-1768`). An untagged granule loads raw
(`target/riscv/op_helper.c:1774-1781`), and LDC into x0 discards the value
(`target/riscv/op_helper.c:1742-1744`). STC writes both doublewords and the tag
in one step, and its source keeps the reference
(`target/riscv/op_helper.c:2982-3030`, `:1998-2005`).

The model checks no instruction fetch against a reference. JAL, JALR and
branches use plain addresses (`target/riscv/insn_trans/trans_rvi.c.inc:53-87`).
Commits `35de150cef` and `1d7995a5b4` deleted the last fetch checks. The
comments at `target/riscv/cap_rev_table_abi.h:89-90` and
`target/riscv/op_helper.c:1078-1079` still mention fetches, but no code checks
one.

## 6. Privilege, CSRs and traps

| CSR | Number | Rule |
|---|---|---|
| srevroot | 0x5c1 | S and M read and write, U gets 2 (`target/riscv/csr.c:3945-3951`). A write clears bits 11:0, drops cached verdicts, and makes U protected exactly when the value is nonzero (`target/riscv/csr.c:4194-4204`) |
| scapctl | 0x5c0 | illegal under the trap route (`target/riscv/csr.c:4135-4143`) |
| urevavail, stagtest | 0xcc0, 0x5c2 | test only, illegal without `x-capstone-test-csrs` (`target/riscv/csr.c:4121-4133`, `:4164-4178`) |

medeleg can delegate causes 24 to 30 (`target/riscv/csr.c:1139-1147`).

While `srevroot` is nonzero (`target/riscv/cpu.h:624-631`):

- Rule 1. In S, a load, store, AMO, LR, SC, LDC or STC with a tagged base takes
  the checks of Section 5 against the table at `srevroot`, translates with SUM
  and may reach user pages. An STC through it keeps the tag
  (`target/riscv/translate.c:276-385`, `target/riscv/op_helper.c:1378-1384`).
- Rule 2. A plain S access reaches no user page, whatever SUM says
  (`target/riscv/cpu_helper.c:61-68`).
- HLV and HSV are illegal (`target/riscv/op_helper.c:562-576`).

While `srevroot` is 0, S treats a tagged base as a plain address
(`target/riscv/cpu_helper.c:155-159`). Commit `500905eab2` leaves vector loads
and stores of S outside rule 1. The trap route refuses to start a CPU with Zcmp
or the T-Head memory extensions (`target/riscv/cpu.c:1703-1713`).

With supervisor references, LDC and STC in S and M take an untagged base, use
the mode's own translation without a Sublet check and keep the tag. Without the
property S raises 24 and M raises 2
(`target/riscv/insn_trans/trans_capstone.c.inc:134-216`, `:326-348`). LCC
selector 8 then returns the id in S (`target/riscv/op_helper.c:1026-1032`). Raw
moves (CLDRAW, CSTRAW, CPTAG, opcode 0x0b) carry a granule and its tag as
opaque data in S and M with V = 0, and grant no authority
(`target/riscv/insn_trans/trans_capstone.c.inc:452-505`,
`target/riscv/op_helper.c:3514-3571`).

Trap entry, SRET and MRET write no integer register, so registers keep their
tags across traps (`target/riscv/cpu_helper.c:1628-1836`,
`target/riscv/op_helper.c:362-459`). Causes 24 to 30 report the tval their
raising site passed, else 0 (`target/riscv/cpu_helper.c:1655-1660`,
`:1714-1715`). `csrrw` on sscratch or mscratch moves a reference with its tag
both ways, and the CSR's integer view is the cursor
(`target/riscv/op_helper.c:89-128`,
`target/riscv/insn_trans/trans_rvi.c.inc:951-980`). csrrs and csrrc with rs1
not x0 clear the tag (`target/riscv/op_helper.c:130-152`), as do integer writes
(`target/riscv/csr.c:1917-1922`, `:2584-2589`), reset
(`target/riscv/cpu.c:928-929`), a virtualization switch
(`target/riscv/cpu_helper.c:543`, `:567`) and a state load
(`target/riscv/machine.c:274-275`).

## 7. Fault causes

| Cause | Name | Condition | Source (`target/riscv/`) |
|---|---|---|---|
| 2 | `RISCV_EXCP_ILLEGAL_INST` | must-be-x0 field nonzero, CSMINT in U or with x2, deleted encodings, protected-U vector, cbo, Zcmp, HLV/HSV with `srevroot` != 0, test CSRs, `srevroot` from U | `insn_trans/trans_capstone.c.inc:26-29`, `:100-118`, `:387-404`, `translate.c:1431-1450`, `op_helper.c:562-576`, `csr.c:3945-3951`, `:4121-4143` |
| 4, 6 | `RISCV_EXCP_LOAD_ADDR_MIS`, `RISCV_EXCP_STORE_AMO_ADDR_MIS` | LDC or STC not 16-aligned, misaligned LR, SC, AMO, CSMINT descriptor or slot | `op_helper.c:1530-1571`, `:1698-1704`, `capstone_table.c:400-414` |
| 24 | `RISCV_EXCP_UNEXP_OP_TYPE` | untagged base, CDERIVE parent untagged or size tagged, CREVOKE or CRENEW untagged, tagged CINCOFFSET or SCC rs2, LCC or CBNZ on an integer, SHRINK operands, S LDC/STC with integer base without supervisor refs | `cpu_bits.h:694`, `op_helper.c:1448-1462`, `:1120-1122`, `:1175-1177`, `:1209-1211`, `:830-832`, `:879-881`, `:948-965`, `:3394-3397`, `:1277-1283`, `:1368-1371` |
| 25 | `RISCV_EXCP_INVALID_CAP` | stale base, CDERIVE parent or CREVOKE/CRENEW object not live | `cpu_bits.h:695`, `op_helper.c:1464-1473`, `:1126-1128`, `:1100-1101`, `:1213-1215` |
| 26 | `RISCV_EXCP_UNEXP_CAP_TYPE` | LCC selectors 6 and 7, SHRINKTO untagged or past the bounds | `cpu_bits.h:696`, `op_helper.c:975-980`, `:1326-1337` |
| 27 | `RISCV_EXCP_INSUF_CAP_PERMS` | never raised for a Sublet reference | `cpu_bits.h:697`, `cap_compress.c:59-65` |
| 28 | `RISCV_EXCP_CAP_OOB` | access outside the bounds | `cpu_bits.h:698`, `op_helper.c:1573-1597` |
| 29 | `RISCV_EXCP_ILLEGAL_OP_VAL` | table invalid or record unreachable, CDERIVE size 0, range outside the parent or inexact bounds, CSMINT refusals, CBNZ target >= 2^38 | `cpu_bits.h:699`, `op_helper.c:1080-1107`, `:1130-1147`, `capstone_table.c:442-461`, `capstone_helper.h:39-45` |
| 30 | `RISCV_EXCP_INSUF_RESOURCES` | allocation without a slot in a growable table, CDERIVE parent at depth 2^31 - 1 | `cpu_bits.h:700`, `op_helper.c:1062-1069`, `cap_rev_tree.c:103-105` |

Page faults (13, 15) and access faults (5, 7) keep their RISC-V meaning and
follow the Sublet checks.

## 8. Tests

capstone-qemu `tests/`:

- `sublet-format`: host checks of codec, record and walk, a model check of all
  six-step lifetime sequences, guest images, fourteen source mutants.
- `sublet-lifetimes`: CDERIVE, CREVOKE and CRENEW per case in protected U.
- `trap-route`: `srevroot`-protected U, tagged scratch CSRs, rules 1 and 2.
- `address-space-root`: one CSMINT, then CDERIVE and CREVOKE per mapping.
- `machine-trap-tags`: register tags across a trap into M.
- `capstone-tag-writers`: every non-STC writer clears the tags it overlaps.
- `capstone-addr-policy`: the 38-bit cursor and the Sv39 ceiling.
- `capstone-untagged-arith`, `movc-untagged-source`: integer rules of
  CINCOFFSET, SCC and MOVC.
- `trusted-linux-u-access`, `virtual-capstone-m1`, `capstone-raw-moves`:
  protected-U accesses, the instruction gate, raw transfers.

This repository:

- `capstone/kernel/tests/sublet-format/checks.c`: kernel format and table
  headers against QEMU's codec and table code on the host, with mutants.
- `capstone/kernel/tests/kernel-opcodes/run.sh`: the kernel uses only allowed
  Sublet encodings.
- `capstone/kernel/tests/trap-route/regs.S`: references survive traps and
  system calls.
- `capstone/kernel/tests/first-program/program.S`: loader references, a stale
  access (25), an access past the bounds (28), a mapping round trip.
- `capstone/kernel/tests/syscall-boot/s5.S`: references through system calls,
  with a rule 1 probe.
- `capstone/runtime/virtual/tests/trap-platform/`: the launcher's fault line
  and the batch runner.
- `capstone/ports/common/tests/lifetime/`, `lifetime-failures/`: allocators
  handle the failure results of CDERIVE and CRENEW.

## 9. Freedoms and simplifications in the model

- The processor reads and writes each record once. The model may read one
  more often with the same result (`target/riscv/cap_rev_table_abi.h:26-28`).
- The model caches liveness verdicts per root and id and drops them on every
  table write, `srevroot` write and watched-page write
  (`target/riscv/capstone_table.c:220-239`, `:312-352`). The result equals a
  record read at every access.
- The model checks two free entries per operation and the whole free list once
  per `srevroot` write. The code states that this cannot catch every corruption
  and relies on the trusted kernel (`target/riscv/cap_rev_table_abi.h:81-92`).
- Each lifetime instruction runs to completion in one helper on the only hart
  (`target/riscv/cpu.c:1685-1700`). The model defines no interleaving of a long
  CREVOKE with another hart, a device or an interrupt.
- Registers and the tag map hold exact base and end
  (`target/riscv/cap_mem_map.h:21`). For values from CDERIVE, CSMINT and the
  cursor moves they equal the decoded word. After SHRINK or SHRINKTO to an
  inexact range they are narrower than the stored word decodes to.
- Comments record differences from the existing RTL: the RTL rounds CDERIVE
  bounds outward instead of raising 29
  (`target/riscv/insn_trans/trans_capstone.c.inc:87-88`), raises 24 for an
  untagged CINCOFFSET or SCC operand (`target/riscv/op_helper.c:762-766`) and
  has no scapctl or urevavail (`target/riscv/csr.c:4121-4126`). On silicon a
  monitor emulates CSMINT, which is why the model refuses x2
  (`target/riscv/insn_trans/trans_capstone.c.inc:392-395`).
- funct7 0x43 to 0x48 under opcode 0x5b are model debug instructions
  (`target/riscv/insn32.decode:980-986`).
