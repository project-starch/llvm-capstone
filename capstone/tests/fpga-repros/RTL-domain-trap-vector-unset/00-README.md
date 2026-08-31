# A domain enters with NO trap vector: `create_domain` never writes the trap-vector context slot

**Status: ROOT-CAUSED on both sides in source, with a silicon positive control. The FIX IS NOT
firmware-only — the RTL text predicts a firmware-only fix converts one storm into another. See
"The fix" below.**

Sibling issues, so a reader arriving with the wrong symptom is redirected immediately:
`../S12-wherecode-notcap-operand-vs-memory/` is where this was found and is the separate question of
*why a capability fault occurs* at one particular instruction; this folder is *why any such fault
kills the board*. The historical `gp-free` domain wedge has the identical signature and is very
likely the same defect. `../RTL-cap-mcause-off-by-one/` is unrelated except that it also concerns
capability exception reporting.

## The defect

`create_domain` zeroes the domain's saved register context and then writes three of its slots. It
never writes slot 1 — the trap-vector slot.

`caplifive-system/.../capstone-sbi/sbi_capstone.c:874-900`:

    for(i = 0; i < DOMAIN_DATA_N; i += 1) { dom_seal[i] = 0; }   /* :875-877 */
    ...
    dom_seal[0] = dom_code;                                       /* :898 */
    dom_seal[2] = dom_data;                                       /* :899 */
    dom_seal[3] = (3 << 38) | (2 << 34);                          /* :900 */
                                                                  /* slot 1: never written */

Slot 1 carries **both** trap vectors. Three independent confirmations of the layout:

    core/csr_regfile.sv:407    7'd1: dom_switch_reg_resp_o = {ctvec_tag_q, ctvec_q, mtvec_q};
    core/csr_regfile.sv:1903-1906   the restore side of the same slot
    verif/tests/custom/capstone/interrupt.S:67   STC(a1, a7, 16)  # ctvec

and the neighbouring slots match slot-for-slot: `dom_seal[2]`→cscratch (`:1907-1911`),
`dom_seal[3]`→mstatus (`:1912`).

The domain switch is an **exchange** (`core/anvil_build/capstone_dom_switcher.anvil:6-60`): it parks
the monitor's live vector in the domain's slot and loads the domain's zeroed word. So a domain runs
with `mtvec = 0` **and** `ctvec = 0` — no trap vector of any kind. Any exception it takes vectors to
address 0.

Board confirmation: `mtvec` reads `0x0` at every wedge, through the same GDB path that returns
non-zero for `$5..$9` in the same dump.

## The `mtvec`-versus-`ctvec` framing is a RED HERRING — corrected 2026-08-31

An earlier version of this file said the RTL vectors on `mtvec` while the monitor installs `ctvec`.
That is true and irrelevant: **the domain's `ctvec` is zero too**, from the same unwritten slot.
`C_WRITE_CCSR(ctvec, _cap_trap_entry)` (`sbi_capstone_dom.c:30`) installs the handler in the
*monitor's* CSR, which the swap then parks in the domain's slot — it never reaches the domain. The
bug would exist unchanged if the RTL vectored on `ctvec`.

Two further corrections to that version, both wrong on the facts:

* It cited `csr_regfile.sv:2995` for `mtvec` resetting to 0. `:2995` is `ctvec_q <= '0`; `mtvec_q
  <= '0` is `:2961`. **And `mtvec` does not stay 0 after reset** — `:2960` sets
  `mtvec_rst_load_q`, and `:1070-1071` then loads `boot_addr_i + 'h40`. The domain's zero comes
  from `create_domain`, not from reset. Pointing at reset invites the fix in the wrong place.
* It said the monitor's handler "does not run because the trap never arrives". True, but narrower
  than the truth: `_cap_trap_entry` is **dead code in every context**. Its only reference in the
  compiled monitor is the `ccsrrw` that stores it into `ctvec`
  (`sbi_capstone_dom.c.S:6765-6766`), nothing jumps to it, and stock OpenSBI points `mtvec` at its
  own `_trap_handler` (`firmware/fw_base.S:480`). The capability trap handler has never executed on
  this hardware, in any context.

The same gap affects the interrupt-handler domain built at `sbi_capstone_dom.c:57-72`. **No domain
anywhere gets a vector.**

## The silicon evidence, including a positive control

The control is deliberately trivial: a domain materialises a plain integer `0xBEEF` and executes
`cincoffsetimm` on it — the most boring capability fault available, and one the monitor is written
to handle and return from.

    probe                    hardware trap latch                architectural CSRs (gdb, halted)
    0xBEEF   tvnh-1          mcause 25  tval 0xbeef             mcause=2  mepc=2  mtval=0  mtvec=0
    0xBEEF   tvnh-2          mcause 25  tval 0xbeef             mcause=2  mepc=2  mtval=0  mtvec=0
    real S-12 s12t-1         mcause 25  tval 0 @0x828f4814      mcause=2  mepc=2  mtval=0  mtvec=0

`EXCX` is 0 in all three. **A benign, recoverable fault kills the board exactly as the real bug
does.** The wedge is a property of the trap path, not of the fault.

## The fix — NOT firmware-only, and the reason matters

*Writable*: yes. `dom_seal[1] = <trap-vector capability>` in `create_domain` sets the domain's own
copy — low 64 bits → `mtvec`, high 64 → `ctvec`, memory tag → bit [128] (`csr_regfile.sv:1905`).
`interrupt.S:67` does exactly this shape. Mind the store width; `sbi_capstone.c:733-760, 866-871`
records two separate bugs caused by getting capability-versus-word stride wrong here.

*Predicted insufficient*, from three lines in the RTL — **this is a prediction from the source text,
not an observation:**

    frontend.sv:425-427   on exception, npc_d = trap_vector_base_i ... but
              :443-444    npc_metadata_d = npc_metadata_q  -- the PC capability is NOT updated
                          (contrast :454-460, where the domain-switch path DOES set it)
    csr_regfile.sv:295    capmode_d = capmode_q | capmode_set_i   -- sticky, cleared only by reset
    commit_stage.sv:222-223   with capmode on, in M-mode, PC outside the PC-capability's bounds:
                              pc_cap_ex_cause = 64'd28  // OUT_OF_BOUNDS

Set the domain's `mtvec` to `_cap_trap_entry` (~`0x8002xxxx`) and the core vectors there still
holding the *domain's* PC capability, whose bounds sit around `0x828xxxxx` — out of bounds, cause
28, re-trap, storm again. There is no exception-triggered domain switch to restore the monitor's PC
capability: the only producer of `dom_switch_en` is `commit_stage.sv:352`, inside the `else` of
`if (commit_instr_i[0].ex.valid)` at `:305`, so **a trap cannot initiate a domain switch.**

The design intent visible in `interrupt.S:67` is that slot 1 holds a trap-vector **capability** —
cursor into `mtvec`, metadata into `ctvec`. The RTL saves and restores `ctvec` but never installs it
as the PC capability on a trap. That is half a feature missing in hardware, so the complete fix is
likely firmware **and** RTL.

## Acceptance criterion — written to FAIL if this is wrong

With the domain's trap vector set, the `0xBEEF` control must stop wedging and must instead produce
`EXCX`/`MCAU` on the UART and a returned `CAPSTONE_DOMAIN_FAULT_RETVAL`.

**One-line discriminator at the next wedge: read `$mcause`.** `28` means the vector took and the
PC-capability path is the remaining blocker — the firmware half is right and the RTL half is
needed. `2` means the vector still is not taking. Reading it costs one GDB command.

## Open

* **`mepc = 2` is NOT explained.** An earlier version said the core "fetches zeros, raises an
  illegal instruction, advances `mepc` to 2". That is mechanically wrong: a trap *sets* `mepc` to
  the faulting PC rather than advancing it, and `16'h0000` is an illegal compressed encoding, so
  zeros at address 0 would give `mepc = 0`. `mepc = 2` requires the halfword at 0 to decode as a
  LEGAL compressed instruction and the one at 2 to fault. What address 0 returns on this SoC is
  not established. **Discriminator: `x/4hx 0x0` at the next wedge**, one GDB command.
* **No healthy-halt control for the CSR dump.** `tvnh-1.log:2567` records the control being skipped
  because the boot wedged. Read `$mtvec/$mcause/$mepc` at a clean teardown halt on any successful
  boot; `mtvec` must come back as `_trap_handler`, non-zero.
* Why a capability fault occurs at `cincoffsetimm a4,a4,0xb0` in the first place. Separate defect,
  stays in the S-12 folder.
