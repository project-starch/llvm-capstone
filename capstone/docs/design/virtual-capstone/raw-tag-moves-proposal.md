# Proposal: trusted raw storage transport and physical tag summaries

Status: proposal on `isa/raw-tag-moves` and `runtime/unpinned-domains`.
No change to the rulings in isa.md is implied. No RTL implementation is supplied.

CLDRAW rd, imm(rs1) and CSTRAW rs2, imm(rs1) transfer one aligned 16-byte
capability granule. Experimental encodings use custom-0 opcode 0x0b, funct3 4
(I format) and 5 (S format), with signed 12-bit byte displacement. They require
the virtual profile and the default-off `x-capstone-raw-moves` option. Only
trusted S/M may execute them; ordinary U, protected U and supervised capability
contexts take an illegal-instruction exception.

The effective address uses the scalar base and current privilege's ordinary
translation, including MPRV. Alignment, translation, access permissions,
ordinary RAM and watchpoints are checked before mutation. Misalignment and
translation/access failures use the corresponding load/store exceptions.
Loading x0 still performs the checks; storing x0 writes two zero words and no
tag. Ordering is ordinary load/store ordering, with no implicit fence.

CLDRAW preserves all stored bits, the storage tag and the model's exact-bounds
side metadata, without decompression, address projection, liveness validation
or linear consumption. The resulting register cargo is opaque: its usable
capability tag is clear, and only CSTRAW can republish the retained storage tag.
Ordinary register producers and debugger scalar edits destroy the retained tag.
CSTRAW preserves stale, linear and sealed representations and does not consume
its source. It may also encode a decoded capability register without applying a
liveness check. Model side metadata is not an additional architectural payload.

This does not relax Q-23: S gains no capability memory-access authority. In
particular, enabling raw transport does not enable scalar-addressed S LDC/STC.
The unruled scalar-load projection item stays unruled. Raw transport explicitly
bypasses projection, so a raw page loop replaces the byte copy rather than
reattaching tags after a scalar copy may already have projected stored bits.

Rule 1 continues to clear every tag overlapped by an ordinary writer. Raw
stores publish bytes and tags under the same leaf lock as normal capability
stores. Translation, fault handling, watchpoints, code invalidation and dirty
bookkeeping occur outside that lock. A tagged CSTRAW into a writable DMA
mapping or memory outside tagged main RAM faults before changing bytes. It
cannot silently weaken its exact-tag contract. A write-mapped source reads
untagged; this is not a coherent DMA-data snapshot. Software must establish
page ownership and exclude concurrent writers.

CPTAG rd, rs1, mode uses custom-0 funct3 6, I format. Mode 0 queries the sticky
physical-page may-hold-tags bit; mode 1 acknowledges verified emptiness. Other
modes are illegal. It has the same privilege/property predicate as raw moves.
rs1 must be a 4096-byte-aligned physical address in tagged system RAM, otherwise
a load-access fault reports that operand. x0 suppresses only the result.

Every tag publication sets the physical summary. Scalar writes, linear loads
and DMA may clear actual tags but leave the summary conservatively set.
Acknowledgement checks all actual tags under the leaf lock and returns one
without clearing the flag while any remain. Once empty it clears the flag and
returns zero. It modifies neither data nor actual tags. Both result modes write
a scalar, destroying any prior raw or usable tag in rd. Physical indexing covers
aliases; publication during a COW/migration copy sets the destination summary.
False positives are allowed; false negatives are not.

Acknowledgement does not establish ownership or prevent a later publication.
Linux must hold the page exclusively from clearing through allocator reuse.
A hardware implementation needs a physical summary plus a coherent verified-empty
operation. The model's OR of its existing tag bitmap and one sticky bit does
not establish hardware latency, cache organization or a multicore protocol.

The platform advertises the enabled configuration in a `capstone,unpinned-v1`
device-tree node: u32 `id-bits`, `exact-bounds`, `page-size` and `granule-size`.
There is no node when raw moves are disabled. This is an experimental platform
binding for explicit discovery, not a new test CSR or a silicon discovery ruling.
The four test page CSRs are absent from this branch.

The Linux raw-copy loop masks local interrupts while opaque cargo is live.
An ordinary integer interrupt frame cannot preserve the retained tag. A direct
trap/register ABI, SMP coherence, protected fork and complete DMA ownership
remain separate work. Tagged swap is refused. Qualification and instruction
counts must not be described as silicon or cycle results.
