#!/usr/bin/env python3
"""Replay the FPGA monitor's region table through a recorded R-43 a2 carve sequence, under a chosen bounds encoder
and carve policy, and report the slot id every region receives plus every live slot's decoded bounds.

Encoder: check-repr.py's literal port of the RTL compress/decompress (--format full --renorm = caplifive_fpb_afcd19bea,
--format cursorless = 0568f93a9). A capability is modelled as what the core holds after writing [start, end) with a
cursor: writeback(). SPLIT (capstone_dyn_unit.anvil func SPLIT): lower = [decoded start, mid) at cursor = decoded start,
upper = [mid, decoded end) at cursor = mid. SHRINK (capstone_flu_unit.anvil): [b, e), cursor clamped into it.

Policy "current" is split_out_cap at capstone-sbi 7361e1c / 93440d5: find the first live slot whose DECODED bounds
cover the request, decide head/tail on those decoded bounds, append the right-hand remainder on a middle carve, SHRINK
the piece (FPGA, R-55), append the piece itself when it is non-linear (and create_region appends it after).

The controls: the replay must reproduce each recorded run's region ids exactly (0568f93a9 cursorless and the respin
full-scheme), or nothing it predicts is evidence."""
import argparse, importlib.util, json, sys

import os
spec = importlib.util.spec_from_file_location(
    "cr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "check-repr.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

TOP = 1 << 40


def wb(s, e, c):
    lo, hi = cr.writeback(s, e, c)
    return (lo, hi, c)


def granule(length):
    """The full scheme's exactness granule (check-repr full_granule): 1 below 4 KiB, else 2^(hb(len) - 9)."""
    return 1 if length < 4096 else 1 << (length.bit_length() - 1 - 9)


def exact_rule(b, e):
    """The rule the monitor can evaluate without the encoder: [b, e) is exact iff b and e are granule multiples."""
    g = granule(e - b)
    return b % g == 0 and e % g == 0


def greedy_end(nb, ne):
    """The largest exact piece from nb: its end is the most aligned point q <= min(ne, nb + 1023 * lowbit(nb)) with
    [nb, q) exact (a longer piece has a coarser granule than nb's alignment). Used only inside a head or tail piece,
    which is shorter than its parent's granule (<= 1 MiB under a DRAM-bounded region)."""
    if exact_rule(nb, ne):
        return ne
    a = nb & -nb if nb else 1 << 62
    lim = min(ne, nb + 1023 * a)
    for t in range(62, -1, -1):
        q = (lim >> t) << t
        if q > nb and exact_rule(nb, q):
            return q
    raise RuntimeError("no exact piece in [%#x, %#x)" % (nb, ne))


def piece_end(nb, ne):
    """The monitor's rule (sbi_capstone.c, FPGA): the end of the next exact piece of [nb, ne), recomputed from the
    current start each time: ne if exact; else with G = granule(ne - nb), h = align_up(nb, G), t = align_down(ne, G):
    greedy_end(nb, h) while h > nb, t when nb is already G-aligned and h < t, greedy_end(nb, ne) otherwise."""
    if exact_rule(nb, ne):
        return ne
    g = granule(ne - nb)
    h = -(-nb // g) * g
    t = ne // g * g
    if h < t:
        if h > nb:
            return greedy_end(nb, h)
        return t
    return greedy_end(nb, ne)


def pieces(nb, ne):
    """The exact decomposition of a remainder [nb, ne): itself if exact; else, with G = granule(ne - nb),
    h = align_up(nb, G), t = align_down(ne, G): a head [nb, h), the middle [h, t) (exact: both ends are G multiples
    and its granule is at most G) and a tail [t, ne), the head and tail cut greedily from the left. Every cut is a
    multiple of G or lies inside a piece shorter than G, so a later request aligned to its size (or to 1 MiB, CMA's
    cap, which is >= G while every remainder is <= 1 GiB) does not straddle one."""
    def greedy(b, e):
        out = []
        while b < e:
            q = greedy_end(b, e)
            out.append((b, q))
            b = q
        return out
    if exact_rule(nb, ne):
        return [(nb, ne)]
    g = granule(ne - nb)
    h = -(-nb // g) * g
    t = ne // g * g
    if h < t:
        return greedy(nb, h) + [(h, t)] + greedy(t, ne)
    return greedy(nb, ne)


class Table:
    def __init__(self, policy, region2_start, region2_end=TOP, nominal=False):
        self.policy = policy
        self.slots = []            # dicts: cap (lo, hi, cursor), nom (b, e) or None, live
        self.log = []
        # genesis (sbi_capstone_init.S): CAPENTER(_cap_text_start, _cap_text_end) gives a0 = [0, cap_text_start) and
        # a1 = [cap_text_end, 2^40), each written to a register; SPLIT(a2, a1, dom_stack) and SPLIT(a3, a2, dom_stack_end)
        # then give region 1 = a1's LOWER piece (its decoded start, not cap_text_end) and region 2 = a3. dom_stack is the
        # boot stack (8 KiB below region2_start's genesis on the board line). region2_end bounds region 2 when a policy
        # splits it off at DRAM's top.
        cap_text_start, cap_text_end = 0x80014000, 0x80030000
        dom_stack, dom_stack_end = region2_start - 0x2000, region2_start
        a1 = wb(cap_text_end, TOP, cap_text_end)
        region1, a2 = self.split(a1, dom_stack)
        _stack, a3 = self.split(a2, dom_stack_end)
        self.slots.append({"cap": wb(0, cap_text_start, 0), "nom": (0, cap_text_start), "live": True})
        self.slots.append({"cap": region1, "nom": (cap_text_end, dom_stack), "live": True})
        self.slots.append({"cap": a3, "nom": (dom_stack_end, TOP), "live": True})
        self.above = None
        self.region2_end = region2_end
        self.inexact = []
        self.protected = []        # (kind, base, end): domain blocks and module regions
        if self.above is not None:
            self.slots.append(self.above)
        self.booted = False

    def covers(self, i, base, end):
        s = self.slots[i]
        if not s["live"]:
            return False
        if self.policy == "current":
            lo, hi, _ = s["cap"]
        else:
            lo, hi = s["nom"]
        return lo <= base and end <= hi

    def split(self, cap, mid):
        lo, hi, _ = cap
        assert lo <= mid < hi, ("split outside decoded bounds", hex(lo), hex(mid), hex(hi))
        return wb(lo, mid, lo), wb(mid, hi, mid)

    def append(self, cap, nom):
        self.slots.append({"cap": cap, "nom": nom, "live": True})
        return len(self.slots) - 1

    def split_out_cap(self, base, length, linear):
        end = base + length
        i = next((k for k in range(len(self.slots)) if self.covers(k, base, end)), None)
        if i is None:
            raise RuntimeError("SPLA: no region covers [%#x, %#x)" % (base, end))
        s = self.slots[i]
        cap = s["cap"]
        if self.policy == "current":
            rb, re = cap[0], cap[1]
        else:
            rb, re = s["nom"]
        if base == rb:
            region, mem_l = cap, None
        else:
            mem_l, region = self.split(cap, base)
        if end == re:
            if base == rb:
                s["live"] = False              # exact fit: hole
            else:
                s["cap"], s["nom"] = mem_l, (rb, base)
        else:
            region, mem_r = self.split(region, end)
            if base == rb:
                s["cap"], s["nom"] = mem_r, (end, re)
            else:
                s["cap"], s["nom"] = mem_l, (rb, base)
                self.append(mem_r, (end, re))
        region = wb(base, end, min(max(region[2], base), end))    # SHRINK, R-55 FPGA
        if self.policy == "exact" and self.booted:
            self.make_exact(i)
            if end != re:
                j = next(k for k in range(len(self.slots)) if self.slots[k]["live"] and self.slots[k]["nom"] == (end, re)) \
                    if base != rb else i
                self.make_exact(j)
        if not linear:
            k = self.append(region, (base, end))
            self.slots[k]["linear"] = False
        return region

    def make_exact(self, i):
        """Policy "exact": cut slot i's LINEAR capability into pieces whose decoded bounds equal their nominal ones,
        with the monitor's iterative piece_end(); the first piece stays in slot i, the rest are appended. Each cut is
        a SPLIT, then a SHRINK of the lower piece to its nominal bounds."""
        s = self.slots[i]
        if not s["live"] or not s.get("linear", True):
            return
        nb, ne = s["nom"]
        cap = s["cap"]
        first = True
        while True:
            q = piece_end(nb, ne)
            if q < ne:
                piece, cap = self.split(cap, q)
            else:
                piece = cap
            piece = wb(nb, q, nb)                          # SHRINK to the nominal piece
            if (piece[0], piece[1]) != (nb, q):
                self.inexact.append((nb, q, piece[0], piece[1]))
            if first:
                s["cap"], s["nom"] = piece, (nb, q)
                first = False
            else:
                self.append(piece, (nb, q))
            if q == ne:
                break
            nb = q

    def create_region(self, base, length):
        r = self.split_out_cap(base, length, 1)
        self.protected.append(("region", base, base + length))
        return self.append(wb(r[0], r[1], base), (base, base + length))

    def create_domain(self, base, tot):
        self.split_out_cap(base, tot, 1)
        self.protected.append(("domain", base, base + tot))

    def overlaps(self):
        """Every (protected range, live slot) pair where the slot's DECODED bounds intersect the range, other than the
        range's own slot (a module region's own slot is that region)."""
        out = []
        for kind, b, e in self.protected:
            for k, s in enumerate(self.slots):
                if not s["live"]:
                    continue
                lo, hi, _ = s["cap"]
                if s["nom"] == (b, e):
                    continue
                if lo < e and b < hi:
                    out.append((kind, b, e, k, lo, hi))
        return out


INT_STACK = 0x8000
INT_SEAL = 0x400


def boot(t, region2_genesis):
    t.split_out_cap(0x10000000, 0x100, 0)            # UART (FPGA)
    t.split_out_cap(0x200bff8, 8, 0)                 # mtime
    t.split_out_cap(0x2004000, 8, 0)                 # mtimecmp
    t.split_out_cap(region2_genesis, INT_STACK, 1)   # the interrupt handler's stack (linear, not appended)
    t.split_out_cap(region2_genesis + INT_STACK, INT_SEAL, 1)   # and its seal region
    t.split_out_cap(0x80030000, 0x10000, 0)          # the interrupt handler's code (region 1's head)


def replay(rec, policy, fmt, region2_start, region2_end, hold=None):
    cr.FORMAT, cr.RENORM = fmt, fmt == "full"
    genesis = region2_start - INT_STACK - INT_SEAL
    t = Table(policy, genesis, region2_end)
    boot(t, genesis)
    if region2_end < TOP:
        piece = t.split_out_cap(region2_end, TOP - region2_end, 1)      # cap_env_init (FPGA): Linux's region ends at DRAM
        t.append(piece, (region2_end, TOP))
    t.booted = True
    if policy == "exact":
        for k in range(len(t.slots)):
            t.make_exact(k)
    if hold is not None:
        # S-30 (2026-10-19): cap_env_init carves [hold) LINEAR out of Linux's region after the exact cut and keeps it
        # privately, never appended, so no live slot may cover the tag store and the revocation-node pool. Counted as a
        # protected range, so OVERLAP counts every live slot whose DECODED bounds still meet it.
        t.held = t.split_out_cap(hold[0], hold[1] - hold[0], 1)
        t.protected.append(("held", hold[0], hold[1]))
    t.boot_slots = len(t.slots)
    got = []
    for x in rec:
        tot = 0x20000 if x["test"] == "k800" else 0x80000
        t.create_domain(x["dbas"], tot)
        ids = []
        order = sorted({sid: (b, l) for sid, b, l in x["shares"]}.items())
        for sid, (b, l) in order:
            ids.append(t.create_region(b, l))
        got.append((x["test"], ids, [sid for sid, _ in order]))
    return t, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--format", choices=["full", "cursorless"], required=True)
    ap.add_argument("--policy", default="current")
    ap.add_argument("--region2", type=lambda v: int(v, 0), default=0x8008a780)
    ap.add_argument("--region2-end", type=lambda v: int(v, 0), default=TOP)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--hold", default=None, help="S-30: BASE:END carved linear at boot and held privately")
    a = ap.parse_args()
    rec = json.load(open(a.record))
    hold = tuple(int(v, 0) for v in a.hold.split(":")) if a.hold else None
    t, got = replay(rec, a.policy, a.format, a.region2, a.region2_end, hold)
    bad = 0
    for test, ids, want in got:
        ok = ids == want
        bad += not ok
        print("%-20s replay %-28s record %-28s %s" % (test, ids, want, "OK" if ok else "DIFFERS"))
    print("slots used: %d (live %d; %d after boot); %d of %d tests differ from the record" %
          (len(t.slots), sum(s["live"] for s in t.slots), t.boot_slots, bad, len(got)))
    ov = t.overlaps()
    print("OVERLAP: %d (protected range, other live slot) pairs; %d protected ranges touched; %d inexact pieces" %
          (len(ov), len({(k, b, e) for k, b, e, *_ in ov}), len(t.inexact)))
    if hold is not None:
        lin = [(k, s["nom"], s["cap"][:2]) for k, s in enumerate(t.slots) if s["live"] and s["nom"][1] == hold[0]]
        print("HOLD: held piece decoded [%#x, %#x); slots ending at %#x: %s" % (t.held[0], t.held[1], hold[0],
              ", ".join("%d nominal [%#x, %#x) decoded [%#x, %#x)" % (k, n[0], n[1], c[0], c[1]) for k, n, c in lin) or "NONE"))
    for kind, b, e, k, lo, hi in ov[:a.show]:
        print("  %-6s [%#x, %#x) covered by slot %d decoded [%#x, %#x) nominal [%#x, %#x)" %
              (kind, b, e, k, lo, hi, *t.slots[k]["nom"]))


if __name__ == "__main__":
    main()
