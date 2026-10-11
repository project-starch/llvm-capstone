"""Decode the R-43 redesign's REFUSAL RECORD off the debug-switch mux.

WHAT IT IS. A sticky latch, in the LSU's revocation-probe block, of the FIRST cause-25 verdict of the
LSU revocation check since the core's reset. The core's async reset clears it, so a JTAG reload holds
nothing from an earlier boot. It reads through bank 3'b110, regs 01100..10000, i.e. switch values
204..208 (RTL lane, capstone-ariane branch r43-query-on-miss):

    204  {2'b00, arm[3:0], ~v, v}  bits[1:0]: 01 latched, 10 empty, 11 VOID (contaminated), 00 unreadable
                                   arm one-hot in bits[5:2]: bit2 hit-dead, bit3 same-cycle invalidation,
                                   bit4 probe resolved DEAD, bit5 timeout
    205  id[7:0]    206 id[15:8]    207 id[23:16]    (id[0] is bit 0 of 205)
    208  {p_hi, p_lo, id[29:24]}    p_lo = XOR of id[14:0], p_hi = XOR of id[29:15]

WHY THE ENCODING IS SELF-CHECKING. On a RUNNING core the LED pulse stretcher ORs every aperture the
switch walk visits (see run_sqlite_stages_fpga._read_sw). OR can only SET bits, so {v, ~v} turns a
contaminated read into 11 and a contaminated arm into multi-hot, both detectable. Any such reading is
VOID and is reported with its reason, never as a value. Parity catches an odd number of flipped id
bits per half. An even number within one half is not caught, and the lines say so.

WHY THIS IS KEYED OFF A FLAG. Before this build, the driver treated 204..209 as the S-07 recorder. That
recorder is absent from the R-42 lineage, where those apertures read 0x00. On a refusal-record
bitstream the S-07 interpretation would misread the record, so REFUSAL_RECORD=1 switches the map. It
is cross-checked against the resident name, which follows caplifive_r43_<commit>.bit or, for the
supervised-CALL build descended from it, caplifive_supcall_<commit>.bit.

The record latches the first denial AS PRESENTED. A wrong-path access that is later flushed can
occupy it, so compare a latched id against the wedge dump's mepc/tval before reading it as the
denial that stopped the domain.
"""
import os

REFUSAL_RECORD = os.environ.get("REFUSAL_RECORD") == "1"
REFUSAL_SW = (204, 205, 206, 207, 208)
ARMS = {2: "hit-dead", 3: "same-cycle invalidation", 4: "probe resolved DEAD", 5: "timeout"}
_LABELS = {204: "REFUSAL {0,0,arm[3:0] one-hot,~v,v}", 205: "REFUSAL id[7:0]",
           206: "REFUSAL id[15:8]", 207: "REFUSAL id[23:16]", 208: "REFUSAL {p_hi,p_lo,id[29:24]}",
           209: "(no refusal field; S-07 map not applicable)"}


class RefusalSkip(Exception):
    """Raised to skip S-07-only procedures on a refusal-record bitstream."""


def rr_label(sw, default):
    return _LABELS.get(sw, default)


def bitstream_note():
    """One line if the flag and the resident name disagree, else None. It warns and never blocks,
    because the name is only a convention. The flag is what the operator asserted."""
    bs = os.environ.get("FPGA_BITSTREAM", "")
    # Every bitstream descended from R-43 v2 (8f6a0af98) carries the record: the R-43 builds and the
    # supervised-CALL build caplifive_supcall_<commit>.bit (capstone-ariane 36a641e0b, resident 2026-10-02).
    named = bs.startswith(("caplifive_r43_", "caplifive_supcall_"))
    if REFUSAL_RECORD and not named:
        return (f"  [refusal] WARNING: REFUSAL_RECORD=1 but FPGA_BITSTREAM={bs!r} does not follow "
                f"caplifive_r43_ or caplifive_supcall_<commit>.bit -- if this silicon has no refusal record, 204..208 "
                f"read 0x00 and decode as UNREADABLE (00), never as a verdict")
    if named and not REFUSAL_RECORD:
        return (f"  [refusal] WARNING: resident {bs!r} looks like a refusal-record build but "
                f"REFUSAL_RECORD is not 1 -- apertures 204..208 are being read with the S-07 map")
    return None


def _parity(x):
    return bin(x).count("1") & 1


def decode(b):
    """b maps switch value -> byte (int) or None (not read). Returns (status, message).
    status is one of LATCHED, EMPTY, VOID, UNREAD, PARTIAL."""
    hx = " ".join(f"{s}:" + ("--" if b.get(s) is None else f"{b[s]:02x}") for s in REFUSAL_SW)
    c = b.get(204)
    if c is None:
        return "UNREAD", f"204 not read -- no verdict either way  [{hx}]"
    if (c >> 6) & 3:
        return "VOID", f"204 bits[7:6] must be 0 (read 0x{c:02x}): not this encoding, or contaminated  [{hx}]"
    pair = c & 3
    arm = (c >> 2) & 0xF
    if pair == 3:
        return "VOID", f"{{~v,v}} = 11: contaminated read (the stretcher ORed another aperture in)  [{hx}]"
    if pair == 0:
        return "VOID", (f"{{~v,v}} = 00: unreadable -- this silicon may carry no refusal record, "
                        f"or the read never landed  [{hx}]")
    if pair == 2:
        if arm:
            return "VOID", f"empty ({{~v,v}}=10) but arm bits set (0x{arm:x}): contaminated  [{hx}]"
        return "EMPTY", f"no cause-25 LSU denial since reset  [{hx}]"
    # pair == 1: latched
    if bin(arm).count("1") != 1:
        return "VOID", f"latched but arm field 0x{arm:x} is not one-hot: contaminated  [{hx}]"
    arm_name = ARMS[2 + (arm.bit_length() - 1)]
    missing = [s for s in (205, 206, 207, 208) if b.get(s) is None]
    if missing:
        return "PARTIAL", (f"LATCHED, arm = {arm_name}; id NOT recovered (unread: "
                           f"{', '.join(map(str, missing))})  [{hx}]")
    rid = b[205] | (b[206] << 8) | (b[207] << 16) | ((b[208] & 0x3F) << 24)
    p_lo, p_hi = (b[208] >> 6) & 1, (b[208] >> 7) & 1
    if p_lo != _parity(rid & 0x7FFF) or p_hi != _parity(rid >> 15):
        return "VOID", (f"latched, arm = {arm_name}, but id parity FAILS (id 0x{rid:08x}, "
                        f"p_lo {p_lo} p_hi {p_hi}): contaminated or misread  [{hx}]")
    return "LATCHED", (f"arm = {arm_name}, revnode id = 0x{rid:08x} ({rid}), parity ok "
                       f"(an even number of flips in one half would pass)  [{hx}]")


def decode_lines(b, context):
    st, msg = decode(b)
    return [f"  [refusal] {context}: {st} -- {msg}"]


def _selftest():
    """Offline positive and negative controls. Every VOID path must fire, and LATCHED must round-trip."""
    def enc(rid, arm_bit):
        p_lo, p_hi = _parity(rid & 0x7FFF), _parity(rid >> 15)
        return {204: (1 << arm_bit) | 0b01, 205: rid & 0xFF, 206: (rid >> 8) & 0xFF,
                207: (rid >> 16) & 0xFF, 208: (p_hi << 7) | (p_lo << 6) | ((rid >> 24) & 0x3F)}
    cases = []
    rid = 0x2ABCDEF1 & 0x3FFFFFFF
    good = enc(rid, 4)
    cases.append(("latched probe-DEAD round-trips", decode(good), "LATCHED", f"0x{rid:08x}"))
    cases.append(("latched hit-dead", decode(enc(0x155, 2)), "LATCHED", "hit-dead"))
    cases.append(("empty", decode({204: 0b10, 205: 0, 206: 0, 207: 0, 208: 0}), "EMPTY", None))
    cases.append(("contaminated pair 11", decode({**good, 204: good[204] | 0b10}), "VOID", "11"))
    cases.append(("unreadable 00 (no record on this silicon)", decode({204: 0, 205: 0, 206: 0, 207: 0, 208: 0}), "VOID", "00"))
    cases.append(("empty + stray arm bit", decode({204: 0b10 | (1 << 3)}), "VOID", "arm bits set"))
    cases.append(("multi-hot arm", decode({**good, 204: good[204] | (1 << 2)}), "VOID", "one-hot"))
    cases.append(("bits 7:6 set", decode({**good, 204: good[204] | 0x40}), "VOID", "bits[7:6]"))
    cases.append(("parity: one id bit ORed in", decode({**good, 205: good[205] | 0x02}
                                                      if not good[205] & 0x02 else {**good, 205: good[205] ^ 0x02}), "VOID", "parity"))
    cases.append(("204 unread", decode({}), "UNREAD", None))
    cases.append(("latched, id bytes unread", decode({204: good[204]}), "PARTIAL", "NOT recovered"))
    bad = 0
    for name, (st, msg), want_st, want_sub in cases:
        ok = st == want_st and (want_sub is None or want_sub in msg)
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {st} -- {msg}")
    print("refusal_record selftest:", "ALL PASS" if bad == 0 else f"{bad} FAILED")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
