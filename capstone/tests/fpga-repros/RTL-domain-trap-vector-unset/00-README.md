# Any exception taken inside a domain jumps to address 0: the core vectors via `mtvec`, the monitor installs `ctvec`

**Status: ROOT-CAUSED, both sides quoted, with a positive control on silicon. The fix is
firmware-only — no bitstream.**

Sibling issues, so a reader arriving with the wrong symptom is redirected immediately:
`../S12-wherecode-notcap-operand-vs-memory/` is where this was found and is the separate question of
*why a capability fault occurs* at one particular instruction; this folder is *why any such fault
kills the board*. The historical `gp-free` domain wedge has the identical signature and is very
likely the same defect. `../RTL-cap-mcause-off-by-one/` is unrelated except that it also concerns
capability exception reporting.

## The chain

    1  RTL     core/csr_regfile.sv:2785    trap_vector_base_o = {mtvec_q[VLEN-1:2], 2'b0};
                                           The ONLY trap vector. `ctvec` appears nowhere in the
                                           trap-vector path -- grepped across all of core/; its
                                           only uses are the CCSR read/write pair
                                           (:2378-2381, capstone_csr_unit.anvil:3,57), the
                                           domain-switch bundle (:407, :1903-1906) and reset
                                           (:2995). Nothing consumes it to vector a trap.

    2  MONITOR opensbi .../sbi_capstone_dom.c:30
                                           C_WRITE_CCSR(ctvec, _cap_trap_entry);
                                           The handler is installed in ctvec and NOWHERE else.
                                           `mtvec` does not appear in sbi_capstone.c,
                                           sbi_capstone.S or sbi_capstone.h at all.

    3  RTL     core/csr_regfile.sv:407     mtvec is part of the PER-DOMAIN switch bundle
                        :2995              and resets to 0.

    4  BOARD                               mtvec reads 0x0 at every wedge.

So a domain runs with `mtvec = 0`. Any exception it takes vectors to address 0, fetches zeros,
raises an illegal instruction, advances `mepc` to 2, and traps to 0 again — forever.

## Why the working path was never affected

Domain-to-monitor calls are not traps. They go through `__domcallsaves(d, CAPSTONE_DPI_CALL, ...)`
(`sbi_capstone.c:936,949,1137,1241,1256`), an explicit domain call, so they never read `mtvec` and
work perfectly. **Only the exception path is dead**, which is exactly why this survived a month of
board sessions in which domains demonstrably created regions, shared them and returned results.

## The silicon evidence, including a positive control

The control is deliberately trivial: a domain materialises a plain integer `0xBEEF` and executes
`cincoffsetimm` on it — the most boring capability fault available, and one the monitor is written
to handle and return from.

    probe                    hardware trap latch                architectural CSRs (gdb, halted)
    0xBEEF   tvnh-1          mcause 25  tval 0xbeef             mcause=2  mepc=2  mtval=0  mtvec=0
    0xBEEF   tvnh-2          mcause 25  tval 0xbeef             mcause=2  mepc=2  mtval=0  mtvec=0
    real S-12 s12t-1         mcause 25  tval 0 @0x828f4814      mcause=2  mepc=2  mtval=0  mtvec=0

`EXCX` is 0 in all three. The monitor's unhandled-exception arm reports EXCX/MCAU/MEPC/MTVL and then
calls `fault_return_from_domain` to terminate the faulting domain and return a code to the caller;
`nm` confirms both are in the deployed firmware (`T fault_return_from_domain` at 0x800239ac). It
does not run, because the trap never arrives.

**A benign, recoverable fault kills the board exactly as the real bug does.** That is the whole
finding: the wedge is not a property of the fault, it is a property of the trap path.

## Why it went unseen

The hardware trap latch filters `cause != 2` (`core/cva6.sv:1126-1137`), so it faithfully preserves
the FIRST nontrivial trap — the capability fault — and hides the cause-2 storm the core actually
dies in. Every investigation read the latch. The architectural CSRs, which carry the other half,
were being printed on the very next line of the same dump. `mtvec` was read into the driver's CSR
dictionary and never printed at all, on every run for weeks — the value that closes this was
collected and discarded each time.

## The fix

Give the domain context a trap vector. `mtvec` is per-domain switch state, so it has to be set for
the domain, not only for the monitor. One write, firmware-only, no bitstream.

**Acceptance criterion, and it must FAIL if the diagnosis is wrong:** with `mtvec` pointing at the
monitor's handler, the `0xBEEF` control must stop wedging the board and must instead produce
`EXCX`/`MCAU` on the UART and a returned `CAPSTONE_DOMAIN_FAULT_RETVAL`. If the board still wedges
with `mtvec` set, this diagnosis is wrong and the trap is being lost somewhere else.

## What this does NOT explain

Nothing here says why a capability fault occurs at `cincoffsetimm a4,a4,0xb0` in the first place.
At that wedge the register file holds the correct cursor from the slot the `ldc` read while `tval`
— the operand as the FLU ingested it — is 0, i.e. the load delivered the right value and the
consumer received something else. That is a separate defect and it stays in the S-12 folder.
