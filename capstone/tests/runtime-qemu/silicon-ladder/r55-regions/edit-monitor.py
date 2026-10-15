#!/usr/bin/env python3
"""R-55 (regions): the FPGA monitor's nominal-bounds region table and exact remainders. Edits sbi_capstone.c and
capstone_target.h in the worktree given as argv[1]. Every replacement must match exactly once."""
import sys

root = sys.argv[1]


def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (path, n, old[:80])
        s = s.replace(old, new)
    open(path, "w").write(s)


GLOBALS = r'''unsigned region_shared_base[CAPSTONE_MAX_REGION_N];

#ifdef CAPSTONE_TARGET_FPGA
/* R-55 (regions, 2026-10-15): every slot's NOMINAL bounds [region_nbase, region_nend), the range it was carved as.
 * A full-scheme bounds encoder (caplifive_fpb_afcd19bea, the Sublet bitstream) rounds a long region's DECODED bounds
 * outward, and decisions taken on cap_base/cap_end then went wrong twice over:
 *  - split_out_cap took nearly every carve for a middle carve and kept a fragment slot each time: about 11 slots per
 *    R-43 a2 invocation against 6, so the table filled at invocation 8 of 12;
 *  - a remainder's decoded bounds covered the piece just carved, and every live slot can be loaded into the CPMP for
 *    S-mode (swap_cpmp), so Linux could reach a domain's memory.
 * With exact_regions set (cap_env_init sets it only when the encoder rounds: region 1's decoded base is not
 * _cap_text_end), coverage and head/tail are decided on these bounds, and region_make_exact cuts every remainder into
 * pieces whose decoded bounds are exactly their nominal ones. region_nend == 0 means "not known": the decoded bounds
 * stand. The model, which reproduces the slot ids of 0568f93a9's and the respin's a2 runs exactly, and its predictions
 * are in llvm-capstone tests/runtime-qemu/silicon-ladder/r55-regions/. */
unsigned region_nbase[CAPSTONE_MAX_REGION_N];
unsigned region_nend[CAPSTONE_MAX_REGION_N];
unsigned exact_regions;
#endif
'''

HELPERS = r'''#ifdef CAPSTONE_TARGET_FPGA
#define CAPSTONE_TAG_EXCT 0x45584354 /* "EXCT" region_make_exact: a piece's decoded bounds are not its nominal ones; value = nominal base, BASE/ALEN follow */
#define CAPSTONE_TAG_EXCS 0x45584353 /* "EXCS" region_make_exact: a cut the capability's decoded bounds do not allow, not executed; value = the cut */

/* R-55 (regions): the full scheme's exactness granule of a length, as the kernel module's capstone_repr_granule and
 * check-repr's full_granule: 1 below 4 KiB, else 2^(hb(len) - 9). Built by doubling and halving: capstone-c evaluates
 * a shift at 32 bits. */
static unsigned repr_granule(unsigned len) {
    unsigned hb;
    unsigned v;
    unsigned g;
    unsigned k;
    if(len < 4096) {
        return 1;
    }
    hb = 0;
    v = len;
    while(v >= 2) {
        v = v / 2;
        hb += 1;
    }
    g = 1;
    for(k = 9; k < hb; k += 1) {
        g = g + g;
    }
    return g;
}

/* [b, e) is exact iff both ends are granule multiples (for such ends the renormalisation step cannot fire). */
static unsigned repr_exact(unsigned b, unsigned e) {
    unsigned g;
    g = repr_granule(e - b);
    if((b & (g - 1)) != 0) {
        return 0;
    }
    if((e & (g - 1)) != 0) {
        return 0;
    }
    return 1;
}

/* The largest exact piece from nb: its end is the most aligned point q <= min(ne, nb + 1023 * lowbit(nb)) with
 * [nb, q) exact (a longer piece has a coarser granule than nb's alignment). [nb, nb + lowbit(nb)) is always exact,
 * so the scan finds one. */
static unsigned repr_greedy_end(unsigned nb, unsigned ne) {
    unsigned a;
    unsigned lim;
    unsigned p;
    unsigned q;
    unsigned best;
    if(repr_exact(nb, ne)) {
        return ne;
    }
    lim = ne;
    if(nb != 0) {
        a = nb & (0 - nb);
        if(nb + 1023 * a < ne) {
            lim = nb + 1023 * a;
        }
    }
    p = 1;
    while(p <= lim / 2) {
        p = p + p;
    }
    best = 0;
    while(p != 0) {
        q = lim - (lim & (p - 1));
        if(best == 0) {
            if(q > nb) {
                if(repr_exact(nb, q)) {
                    best = q;
                }
            }
        }
        p = p / 2;
    }
    return best;
}

/* The end of the next exact piece of [nb, ne): ne if exact; else, with G = granule(ne - nb), h = align_up(nb, G),
 * t = align_down(ne, G): the greedy head while nb is below h, the middle [h, t) once nb is G-aligned, the greedy rest
 * when the range has no G-aligned middle. Cuts fall on multiples of the remainder's own granule or inside a piece
 * shorter than it, so a later request aligned to its size (or to 1 MiB, CMA's cap, which is >= G while Linux's region
 * ends at CAPSTONE_DRAM_TOP) never straddles one. */
static unsigned repr_piece_end(unsigned nb, unsigned ne) {
    unsigned g;
    unsigned h;
    unsigned t;
    if(repr_exact(nb, ne)) {
        return ne;
    }
    g = repr_granule(ne - nb);
    h = nb + ((g - (nb & (g - 1))) & (g - 1));
    t = ne - (ne & (g - 1));
    if(h < t) {
        if(h > nb) {
            return repr_greedy_end(nb, h);
        }
        return t;
    }
    return repr_greedy_end(nb, ne);
}

/* Append a live, non-resident slot with known nominal bounds; the caller has checked the table's room. */
static void region_append(__linear void *cap, unsigned nb, unsigned ne) {
    regions[region_n] = cap;
    region_live[region_n] = 1;
    region_cpmp[region_n] = -1;
    region_shared_base[region_n] = 0;
    region_nbase[region_n] = nb;
    region_nend[region_n] = ne;
    region_n += 1;
}

/* R-55 (regions): cut slot i's LINEAR capability into pieces whose decoded bounds are their nominal ones. Each cut is a
 * SPLIT at the piece's end, then a SHRINK of the lower piece to [nb, q); the bounds are read back and a piece that is
 * still not exact is reported (EXCT). The first piece stays in slot i, the rest are appended; the pieces are counted
 * first, and if the table cannot hold them nothing is cut (RGNF). A cut or SHRINK the capability's decoded bounds do
 * not allow (a SHORT region) is reported (EXCS) and never executed: what is left stays in one inexact piece. */
static void region_make_exact(unsigned i) {
    __linear void *cap;
    __linear void *rest;
    unsigned nb;
    unsigned ne;
    unsigned q;
    unsigned n;
    unsigned first;
    unsigned done;
    unsigned lo;
    unsigned hi;
    if(exact_regions == 0) {
        return;
    }
    if(region_live[i] == 0) {
        return;
    }
    if(region_nend[i] == 0) {
        return;
    }
    nb = region_nbase[i];
    ne = region_nend[i];
    n = 0;
    q = nb;
    while(q < ne) {
        q = repr_piece_end(q, ne);
        n += 1;
    }
    if(region_n + n - 1 > CAPSTONE_MAX_REGION_N) {
        REPORT_REGION_OVERFLOW();
        return;
    }
    if(region_cpmp[i] != -1) {
        cap = read_cpmp(region_cpmp[i]);
    } else {
        cap = regions[i];
    }
    if(cap_type(cap) != CAP_TYPE_LINEAR) {
        if(region_cpmp[i] != -1) {
            write_cpmp(region_cpmp[i], cap);
        } else {
            regions[i] = cap;
        }
        return;
    }
    first = 1;
    done = 0;
    while(done == 0) {
        q = repr_piece_end(nb, ne);
        lo = cap_base(cap);
        hi = cap_end(cap);
        if(q < ne) {
            if(lo > nb) {
                q = ne;
            }
            if(q >= hi) {
                q = ne;
            }
            if(q == ne) {
                capstone_report(CAPSTONE_TAG_EXCS, nb);
            }
        }
        if(q < ne) {
            rest = __split(cap, q);
        }
        lo = cap_base(cap);
        hi = cap_end(cap);
        if(lo <= nb) {
            if(q <= hi) {
                __asm__ volatile(".insn r 0x5b, 0x1, 0x1, %0, %1, %2" :: "r"(cap), "r"(nb), "r"(q));
            }
        }
        lo = cap_base(cap);
        hi = cap_end(cap);
        if(lo != nb) {
            capstone_report(CAPSTONE_TAG_EXCT, nb);
            capstone_report(CAPSTONE_TAG_BASE, lo);
            capstone_report(CAPSTONE_TAG_ALEN, hi - lo);
        } else if(hi != q) {
            capstone_report(CAPSTONE_TAG_EXCT, nb);
            capstone_report(CAPSTONE_TAG_BASE, lo);
            capstone_report(CAPSTONE_TAG_ALEN, hi - lo);
        }
        if(first) {
            if(region_cpmp[i] != -1) {
                write_cpmp(region_cpmp[i], cap);
            } else {
                regions[i] = cap;
            }
            region_nbase[i] = nb;
            region_nend[i] = q;
            first = 0;
        } else {
            region_append(cap, nb, q);
        }
        if(q == ne) {
            done = 1;
        } else {
            cap = rest;
            nb = q;
        }
    }
}
#endif

static void *split_out_cap(unsigned base, unsigned len, unsigned linear) {'''


def main():
    c = root + "/sbi_capstone.c"
    s = open(c).read()
    # the forward declaration line we emit must not exist yet; the definition line must exist once
    assert s.count("static void *split_out_cap(unsigned base, unsigned len, unsigned linear) {") == 1
    pairs = [
        ("unsigned region_shared_base[CAPSTONE_MAX_REGION_N];\n", GLOBALS),
        ("static void *split_out_cap(unsigned base, unsigned len, unsigned linear) {", HELPERS),
        # make_hole (FPGA): the hole has no nominal bounds
        ('''    region_live[i] = 0;
    regions[i] = 0;
    region_shared_base[i] = 0;
    capstone_report(CAPSTONE_TAG_HOLE, i);''', '''    region_live[i] = 0;
    regions[i] = 0;
    region_shared_base[i] = 0;
    region_nbase[i] = 0;
    region_nend[i] = 0;
    capstone_report(CAPSTONE_TAG_HOLE, i);'''),
        # coverage: nominal (and decoded) under exact_regions
        ('''        region_base = cap_base(mem_l);
        region_end = cap_end(mem_l);
        /* THE SITE WITH NO RE-SHARE.''', '''        region_base = cap_base(mem_l);
        region_end = cap_end(mem_l);
#ifdef CAPSTONE_TARGET_FPGA
        /* R-55 (regions): with exact_regions, a slot covers the request when its NOMINAL bounds do, and only if its
           decoded bounds do as well (the SPLITs below act on those); head and tail are then judged on the nominal
           bounds (nom_base/nom_end), so a carve at a region's nominal start leaves no fragment. */
        nom_base = region_base;
        nom_end = region_end;
        if(exact_regions != 0) {
            if(region_nend[i] != 0) {
                nom_base = region_nbase[i];
                nom_end = region_nend[i];
            }
        }
#endif
        /* THE SITE WITH NO RE-SHARE.'''),
        ('''        if(base >= region_base && base + len <= region_end)
            break;''', '''#ifdef CAPSTONE_TARGET_FPGA
        if(base >= nom_base && base + len <= nom_end) {
            if(base >= region_base && base + len <= region_end)
                break;
        }
#else
        if(base >= region_base && base + len <= region_end)
            break;
#endif'''),
        ('''    __linear void *mem_l;
    __linear void *mem_r;
    unsigned fill_n;''', '''    __linear void *mem_l;
    __linear void *mem_r;
#ifdef CAPSTONE_TARGET_FPGA
    unsigned nom_base;
    unsigned nom_end;
    unsigned kept_slot;
    unsigned right_slot;
#endif
    unsigned fill_n;'''),
    ]
    pairs += [
        # head/tail decisions on the nominal bounds
        ("""    if(i >= region_n)
        capstone_error(CAPSTONE_NO_REGION);
#endif

    if(base == region_base)
        region = mem_l;""", """    if(i >= region_n)
        capstone_error(CAPSTONE_NO_REGION);
#endif
#ifdef CAPSTONE_TARGET_FPGA
    region_base = nom_base;   /* R-55 (regions): head and tail on the nominal bounds (the decoded ones when not known) */
    region_end = nom_end;
#endif

    if(base == region_base)
        region = mem_l;"""),
        # after the piece's SHRINK: record the slots' nominal bounds, cut the remainders exact
        ("""    __asm__ volatile(".insn r 0x5b, 0x1, 0x1, %0, %1, %2" :: "r"(region), "r"(base), "r"(base + len));
#endif""", """    __asm__ volatile(".insn r 0x5b, 0x1, 0x1, %0, %1, %2" :: "r"(region), "r"(base), "r"(base + len));
    /* R-55 (regions): what slot i, and the slot a middle carve appended, now nominally hold; then cut them into exact
       pieces (region_make_exact does nothing unless exact_regions). An exact fit left slot i a hole. */
    kept_slot = -1;
    right_slot = -1;
    if(base + len == region_end) {
        if(base != region_base) {
            region_nbase[i] = region_base;
            region_nend[i] = base;
            kept_slot = i;
        }
    } else {
        if(base == region_base) {
            region_nbase[i] = base + len;
            region_nend[i] = region_end;
            kept_slot = i;
        } else {
            region_nbase[i] = region_base;
            region_nend[i] = base;
            kept_slot = i;
            region_nbase[region_n - 1] = base + len;
            region_nend[region_n - 1] = region_end;
            right_slot = region_n - 1;
        }
    }
    if(kept_slot != -1) {
        region_make_exact(kept_slot);
    }
    if(right_slot != -1) {
        region_make_exact(right_slot);
    }
#endif"""),
        # the non-linear piece's slot
        ("""        regions[region_n] = region;
        region_live[region_n] = 1;
        region_n += 1;
    }

    return region;
}""", """        regions[region_n] = region;
        region_live[region_n] = 1;
#ifdef CAPSTONE_TARGET_FPGA
        region_nbase[region_n] = base;
        region_nend[region_n] = base + len;
#endif
        region_n += 1;
    }

    return region;
}"""),
        # create_region's slot
        ("""    regions[region_n] = region;
    region_live[region_n] = 1;
    region_n += 1;

    return region_n - 1;""", """    regions[region_n] = region;
    region_live[region_n] = 1;
#ifdef CAPSTONE_TARGET_FPGA
    region_nbase[region_n] = base;   /* R-55 (regions) */
    region_nend[region_n] = base + len;
#endif
    region_n += 1;

    return region_n - 1;"""),
        # share_child_region's head and tail
        ("""    if (head) {
        regions[region_n] = head;
        region_live[region_n] = 1;
        region_n += 1;
    }
    if (tail) {
        regions[region_n] = tail;
        region_live[region_n] = 1;
        region_n += 1;
    }""", """    if (head) {
        regions[region_n] = head;
        region_live[region_n] = 1;
#ifdef CAPSTONE_TARGET_FPGA
        region_nbase[region_n] = pbase;   /* R-55 (regions) */
        region_nend[region_n] = cstart;
#endif
        region_n += 1;
#ifdef CAPSTONE_TARGET_FPGA
        region_make_exact(region_n - 1);
#endif
    }
    if (tail) {
        regions[region_n] = tail;
        region_live[region_n] = 1;
#ifdef CAPSTONE_TARGET_FPGA
        region_nbase[region_n] = cend;
        region_nend[region_n] = pend;
#endif
        region_n += 1;
#ifdef CAPSTONE_TARGET_FPGA
        region_make_exact(region_n - 1);
#endif
    }"""),
        # dpi_share_region: a region arriving from a domain has no known nominal bounds
        ("    region_shared_base[region_n] = shared_base;\n", """    region_shared_base[region_n] = shared_base;
#ifdef CAPSTONE_TARGET_FPGA
    region_nbase[region_n] = 0;   /* R-55 (regions): not known; its decoded bounds stand */
    region_nend[region_n] = 0;
#endif
"""),
        # the managed domain's cache slot: the block it will hold
        ("""            cache_region = region_n;
            region_n += 1;""", """            cache_region = region_n;
#ifdef CAPSTONE_TARGET_FPGA
            region_nbase[cache_region] = base_addr;   /* R-55 (regions): the block this slot will hold */
            region_nend[cache_region] = base_addr + tot_size;
#endif
            region_n += 1;"""),
        # pop_region
        ("        region_live[region_i] = 0;\n", """        region_live[region_i] = 0;
#ifdef CAPSTONE_TARGET_FPGA
        region_nbase[region_i] = 0;
        region_nend[region_i] = 0;
#endif
"""),
    ]
    optional = {"            cache_region = region_n;\n            region_n += 1;"}   # the managed path: absent on the board line
    for old, new in pairs:
        n = s.count(old)
        if n == 0 and old in optional:
            print("absent here (optional):", old.strip().splitlines()[0])
            continue
        assert n == 1, (n, old[:90])
        s = s.replace(old, new)
    open(c, "w").write(s)

    h = root + "/capstone_target.h"
    edit(h, [("""#define CAPSTONE_MAX_DOM_N   32
#define CAPSTONE_MAX_REGION_N   96
#else""", """#define CAPSTONE_MAX_DOM_N   32
/* R-55 (regions, 2026-10-15): 127. Exact remainders under a full-scheme encoder take more slots than the 96 the
 * board ran since 2026-08-18: R-43 a2's twelve invocations and its closing k800 need 111 by the replay
 * (r55-regions/); 0568f93a9's run used 89. 127 is the most capstone-c can carve: dom_init reserves each global with
 * one `addi t1, t1, -size`, a 12-bit immediate, and regions[] is 16 bytes a slot (160 slots failed to assemble).
 * The kernel module's MAX_REGION_N and libcapstone's must match. */
#define CAPSTONE_MAX_REGION_N   127
/* The top of the board's DRAM window: memory@80000000 is 0x3c2d2000 bytes (CMA at 0xac000000 + 256 MiB), and
 * nothing S-mode uses lies above 0xc0000000. cap_env_init ends Linux's region here, so every remainder is <= 1 GiB
 * and its granule <= 1 MiB (R-55 regions). */
#define CAPSTONE_DRAM_TOP   0xc0000000
#else""")])


main()
