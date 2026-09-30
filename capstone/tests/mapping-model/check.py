#!/usr/bin/env python3
"""Contract scenarios, bounded state exploration, and seeded adversarial traces."""

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import sys

from model import (Action, ADDRESS_LIMIT, ARCH_LOGICAL_BASE, ARCH_LOGICAL_LIMIT,
                   ARCH_PAGE, ARCH_PHYSICAL_BITS, ARCH_PHYSICAL_LIMIT,
                   BINDING_GEN_BITS, BINDING_ID_BITS, Cap, HARTS, LOGICAL_BASE,
                   Machine, MUTANTS, PAGES, PHYSICAL_LIMIT, Refused, ResumeSlot,
                   Violation, Word, WORDS, arch_kind, arch_partition_ok,
                   pack_binding, reservation_ok, unpack_binding)


def base(ident):
    return LOGICAL_BASE + ident * 2 * PAGES * WORDS


def create_action(ident, root, pointer, handle, size=PAGES * WORDS, rights="rw"):
    """Fixture recipe; the instruction receives an explicit recipient and range."""
    domain, name = pointer.split(":", 1)
    target = ResumeSlot(int(domain[1:]), name) if domain.startswith("d") else pointer
    recipient = "m:domain%d" % (target.domain if isinstance(target, ResumeSlot) else 0)
    return Action("prepare_create", (ident, root, recipient, target, handle,
                                     base(ident), base(ident) + size, rights))


def offset_action(machine, name, arguments):
    """Test recipes use mapping-relative offsets; execution traces are absolute.

    This is a harness convenience only. Machine instructions never call it and
    must independently reject invalid recipients, geometry and operands.
    """
    if name == "prepare_create":
        return create_action(*arguments)
    args = list(arguments)
    location_index = 1 if name == "issue" else 0
    positions = {"issue": (3,), "prepare_populate": (1,), "begin_unmap": (1,),
                 "narrow": (1, 2), "split": (1,), "cursor": (1,)}.get(name, ())
    cap = machine.wallet.get(args[location_index]) if positions else None
    if cap and cap.binding in machine.mappings:
        lo = machine.mappings[cap.binding].lo
        for position in positions:
            if position < len(args) and args[position] is not None:
                args[position] += lo
    return Action(name, tuple(args))


class Script:
    def __init__(self, mutant=None, barrier_mode="global"):
        self.machine = Machine(mutant, barrier_mode=barrier_mode)
        self.trace = []
        self.refusals = 0

    def step(self, name, *args):
        return self.exact(offset_action(self.machine, name, args))

    def exact(self, action):
        self.trace.append(str(action))
        result = action.apply(self.machine)
        self.machine.check()
        return result

    def refused(self, name, *args):
        self.refused_action(offset_action(self.machine, name, args))

    def refused_action(self, action):
        before = self.machine.key()
        try:
            self.exact(action)
        except Refused:
            assert self.machine.key() == before, "refusal changed state"
            self.refusals += 1
        else:
            raise AssertionError("expected refusal: " + str(self.trace[-1]))

    def pages(self, count=16):
        for domain in HARTS:
            self.machine.bootstrap_domain(domain, "m:domain%d" % domain)
        for page in range(count):
            self.machine.bootstrap(page, "m:p%d" % page, "m:h%d" % page)
        self.machine.check()

    def create(self, ident, page, suffix=None, rights="rw"):
        suffix = str(ident) if suffix is None else suffix
        self.step("prepare_create", ident, "m:p%d" % page,
                  "d%d:p%s" % (ident, suffix), "m:d" + suffix, 4 * WORDS, rights)
        self.step("publish")

    def populate(self, suffix, address, frame, table=None):
        pages = () if table is None else ("m:p%d" % table,)
        self.step("prepare_populate", "m:d" + str(suffix), address,
                  "m:p%d" % frame, pages)
        self.step("publish")

    def barrier(self):
        for hart in HARTS:
            self.step("invalidate", hart)
            for request, access in list(self.machine.accesses.items()):
                if access.hart == hart and access.phase == "checked":
                    self.step("cancel", request)
            self.step("drain", hart)
        self.step("finish")

    def access(self, hart, location, operation="load", address=0, value=7,
               source=None, destination=None):
        request = self.step("issue", hart, location, operation, address, value,
                            source, destination)
        while self.machine.accesses[request].phase in ("root", "leaf", "fill"):
            self.step("walk", request)
        if self.machine.accesses[request].phase == "checked":
            self.step("memory_step", request)
        return self.machine.accesses[request]

    def zero(self, location):
        cap = self.machine.wallet[location]
        method = "scrub_logical" if cap.kind == "logical_uninit" else "scrub"
        for _ in range(cap.hi - cap.cursor):
            self.step(method, location)
        self.step("init", location)


def fixture(mutant=None, barrier_mode="global"):
    s = Script(mutant, barrier_mode)
    s.pages()
    s.create(0, 0)
    s.populate(0, 0, 2, 1)
    s.create(1, 8)
    s.populate(1, 0, 10, 9)
    return s


def check_permissions_and_inputs():
    s = fixture()
    s.refused("prepare_create", 0, "m:p3", "d0:new", "m:new")
    s.refused("prepare_create", 2, "m:p3", "d0:new", "m:new")
    s.refused("prepare_create", 0, "d0:p0", "d0:new", "m:new")
    s.refused("prepare_populate", "m:d0", 0, "m:p3", ())
    s.refused("prepare_populate", "m:d0", 8, "m:p3", ())
    s.refused("prepare_populate", "m:d0", 8, "m:p3", ("m:p3",))
    s.refused("prepare_populate", "m:d0", 8, "d0:p0", ("m:p3",))
    s.refused("prepare_populate", "m:d1", 16, "m:p3", ("m:p4",))
    s.step("delin", "m:p3")
    s.refused("prepare_populate", "m:d0", 8, "m:p3", ("m:p4",))
    s.refused("prepare_populate", "m:d0", 8, "m:p4", ("m:p3",))
    s.refused("physical_read", "m:d0", 0)
    s.refused("begin_unmap", "m:d0", 0, "m:out")
    s.populate(0, 4, 5)
    s.populate(0, 8, 6, 4)
    for address, value in ((0, 11), (4, 22), (8, 33)):
        assert s.access(0, "d0:p0", "store", address, value).phase == "done"
        assert s.access(0, "d0:p0", "load", address).result.value == value
    s.step("narrow", "d0:p0", 4, 8, "r")
    s.refused("issue", 0, "d0:p0", "store", 4)
    s.refused("issue", 0, "d0:p0", "load", 3)
    s.step("cursor", "d0:p0", 8)
    s.refused("issue", 0, "d0:p0")
    assert s.access(0, "d0:p0", "load", 7).phase == "done"
    t = Script()
    t.pages(3)
    t.create(0, 0, rights="r")
    t.refused("prepare_create", 1, "d0:p0", "d1:wrong", "m:wrong")
    t.populate(0, 0, 2, 1)
    assert t.access(0, "d0:p0").phase == "done"
    t.refused("issue", 0, "d0:p0", "store")
    # The unreadable PTE retains the original frame's W for initialization and
    # return scrubbing; the logical capability supplies the R-only ceiling.
    assert t.machine.memory[1][0].rights == "rw"
    t.step("mrev", "d0:p0", "d0:readrev")
    t.step("begin_revoke", "d0:readrev", "d0:readuninit")
    t.barrier()
    t.refused("scrub_logical", "d0:readuninit")
    t.refused("init", "d0:readuninit")
    t.step("begin_detach", "m:d0", "m:t0")
    t.barrier()
    t.step("begin_unmap", "m:t0", 0, "m:returned")
    t.barrier()
    t.zero("m:returned")
    u = Script()
    u.pages(4)
    u.step("narrow", "m:p0", 0, WORDS, "r")
    u.refused("prepare_create", 0, "m:p0", "d0:p0", "m:d0")
    u.create(0, 1)
    u.step("narrow", "m:p2", 0, WORDS, "r")
    u.refused("prepare_populate", "m:d0", 0, "m:p3", ("m:p2",))
    u.refused("prepare_populate", "m:d0", 0, "m:p2", ("m:p3",))
    u.step("mrev", "m:p2", "m:readrev")
    u.step("begin_revoke", "m:readrev", "m:readuninit")
    u.barrier()
    u.refused("scrub", "m:readuninit")
    return s.refusals + t.refusals + u.refusals


def check_reclamation():
    s = fixture()
    s.populate(0, 4, 3)
    s.access(0, "d0:p0", "store", 0, 99)
    s.step("begin_revoke", "m:h2", "m:reclaimed")
    s.barrier()
    assert s.access(0, "d0:p0", "load", 0).phase == "fault"
    assert s.access(0, "d0:p0", "load", 4).phase == "done"
    s.refused("prepare_populate", "m:d0", 0, "m:p4", ())
    s.refused("physical_read", "m:reclaimed")
    s.refused("init", "m:reclaimed")
    s.zero("m:reclaimed")
    for offset in range(WORDS):
        assert s.step("physical_read", "m:reclaimed", offset) == Word()
    s.step("begin_detach", "m:d0", "m:t0")
    s.barrier()
    s.step("move", "m:t0", "stored:token")
    s.step("move", "stored:token", "m:t0")
    s.refused("physical_read", "m:t0")
    s.refused("begin_unmap", "m:t0", 0, "m:again")
    s.step("begin_unmap", "m:t0", 4, "m:unmapped")
    s.barrier()
    s.refused("begin_unmap", "m:t0", 4, "m:twice")
    s.step("destroy", "m:t0")
    s.refused("destroy", "m:t0")
    for page in (0, 1):
        s.step("begin_revoke", "m:h%d" % page, "m:table%d" % page)
        s.barrier()
        s.refused("begin_revoke", "m:table%d" % page, "m:twice")
        s.refused("physical_read", "m:table%d" % page)
        s.zero("m:table%d" % page)
    assert s.access(1, "d1:p1").phase == "done"
    return s.refusals


def check_cut_subtree_and_generation():
    s = fixture()
    s.populate(0, 8, 4, 3)
    old_binding = s.machine.registry[0].binding
    s.step("begin_revoke", "m:h0", "m:root")
    s.barrier()
    s.refused("prepare_create", 0, "m:p5", "d0:new", "m:new")
    assert s.access(0, "d0:p0").phase == "fault"
    assert s.access(1, "d1:p1").phase == "done"
    s.step("begin_detach", "m:d0", "m:t0")
    s.barrier()
    # Present PTEs and two lower tables survive root loss. No traversal needed.
    s.step("destroy", "m:t0")
    s.create(0, 5, "new")
    s.populate("new", 0, 7, 6)
    new_binding = s.machine.registry[0].binding
    assert new_binding != old_binding
    for page in (1, 2, 3, 4):
        s.step("begin_revoke", "m:h%d" % page, "m:old%d" % page)
        s.barrier()
        assert s.machine.registry[0].binding == new_binding
        assert s.access(0, "d0:pnew").phase == "done"
    s.refused("issue", 0, "d0:p0")
    return s.refusals


def check_old_root_and_present_destroy():
    s = fixture()
    s.step("begin_detach", "m:d0", "m:t0")
    s.barrier()
    s.step("destroy", "m:t0")
    s.create(0, 3, "new")
    s.populate("new", 0, 5, 4)
    new_root = s.machine.registry[0].root
    for page in (0, 1, 2):
        s.step("begin_revoke", "m:h%d" % page, "m:old%d" % page)
        s.barrier()
        assert s.machine.registry[0].root == new_root
        assert s.access(0, "d0:pnew").phase == "done"
    return s.refusals


def check_pointer_tree_and_contexts():
    s = fixture()
    s.step("split", "d0:p0", WORDS, "d0:rest")
    s.populate(0, WORDS, 3)
    s.step("mrev", "d0:p0", "d0:object")
    s.step("delin", "d0:p0")
    s.step("move", "d0:p0", "d0:old")
    s.step("move", "d0:p0", "d1:foreign")
    s.access(0, "d0:p0", "store", 0, 18)
    assert s.access(1, "d1:foreign").result.value == 18
    s.step("begin_revoke", "d0:object", "d0:new")
    s.barrier()
    s.refused("issue", 0, "d0:old")
    s.refused("issue", 1, "d1:foreign")
    assert s.access(0, "d0:new").result.value == 18
    s.step("mrev", "d0:new", "d0:second")
    s.step("move", "d0:new", "d1:loan")
    s.step("begin_revoke", "d0:second", "d0:uninit")
    s.barrier()
    assert s.machine.wallet["d0:uninit"].kind == "logical_uninit"
    s.refused("issue", 0, "d0:uninit")
    s.zero("d0:uninit")
    assert s.access(0, "d0:uninit").result == Word()
    assert s.access(0, "d0:rest", "load", WORDS).phase == "done"
    s.step("drop", "d0:uninit")
    s.barrier()
    s.refused("issue", 0, "d0:uninit")
    s.step("begin_detach", "m:d0", "m:t0")
    s.barrier()
    s.refused("issue", 0, "d0:rest", "load", WORDS)
    return s.refusals


def check_capability_transfers():
    s = fixture()
    stored = s.machine.wallet["d1:p1"]
    s.step("move", "d1:p1", "d0:payload")
    assert s.access(0, "d0:p0", "cstore", source="d0:payload").phase == "done"
    assert "d1:p1" not in s.machine.wallet
    s.step("delin", "d0:p0")
    s.step("move", "d0:p0", "d1:foreign")
    requests = [s.step("issue", h, "d%d:%s" % (h, "p0" if h == 0 else "foreign"),
                       "cload", 0, 0, None, "h%d:loaded" % h) for h in HARTS]
    for request in requests:
        while s.machine.accesses[request].phase != "checked":
            s.step("walk", request)
    s.step("memory_step", requests[0])
    s.step("memory_step", requests[1])
    assert s.machine.wallet["h0:loaded"] == stored
    assert s.machine.accesses[requests[1]].phase == "fault"
    # The moved capability still names mapping 1 from context 0.
    assert s.access(0, "h0:loaded").phase == "done"
    assert s.access(0, "d0:p0", "cstore", source="h0:loaded").phase == "done"
    s.step("narrow", "d0:p0", 0, 16, "r")
    assert s.access(0, "d0:p0", "cload", destination="h0:denied").phase == "fault"
    assert isinstance(s.machine.memory[2][0], Cap)
    return s.refusals


def check_preparation_races():
    s = fixture()
    s.step("prepare_populate", "m:d0", 8, "m:p3", ("m:p4",))
    s.step("begin_revoke", "m:h4", "m:reclaimed")
    s.barrier()
    s.refused("publish")
    s.step("abort")
    assert s.machine.memory[0][1] == Word()
    assert s.machine.live(s.machine.wallet["m:p3"].node)
    assert not s.machine.live(s.machine.wallet["m:p4"].node)
    s.step("prepare_populate", "m:d0", 8, "m:p3", ("m:p5",))
    s.step("begin_detach", "m:d0", "m:t0")
    s.barrier()
    s.refused("publish")
    s.step("abort")
    assert s.machine.memory[0][1] == Word()
    t = Script()
    t.pages(1)
    t.step("prepare_create", 0, "m:p0", "d0:p", "m:d")
    t.step("begin_revoke", "m:h0", "m:root")
    t.barrier()
    t.refused("publish")
    t.step("abort")
    assert not t.machine.registry
    return s.refusals + t.refusals


def check_generation_exhaustion_and_tokens():
    s = fixture()
    for ident in (0, 1):
        s.step("begin_detach", "m:d%d" % ident, "m:t%d" % ident)
        s.barrier()
    # A token selects its own mapping. There is no independent target-id operand.
    s.step("begin_unmap", "m:t1", 0, "m:from1")
    s.barrier()
    assert s.machine.wallet["m:from1"].page == 10
    assert isinstance(s.machine.memory[1][0], Cap)
    s.step("destroy", "m:t0")
    for generation, page in ((2, 3), (3, 4)):
        suffix = "g%d" % generation
        s.create(0, page, suffix)
        assert s.machine.registry[0].binding == (0, generation)
        s.step("begin_detach", "m:d" + suffix, "m:t" + suffix)
        s.barrier()
        s.step("destroy", "m:t" + suffix)
    s.refused("prepare_create", 0, "m:p5", "d0:wrapped", "m:wrapped")
    return s.refusals


def check_protected_delivery():
    s = Script()
    s.pages(6)
    valid = create_action(0, "m:p0", "d0:resume", "m:d0")
    ident, root, recipient, slot, handle, lo, hi, rights = valid.arguments
    for destination in ("m:leak", "d0:resume", ResumeSlot(1, "resume")):
        s.refused_action(Action("prepare_create", (ident, root, recipient,
                         destination, handle, lo, hi, rights)))
    s.refused_action(Action("prepare_create", (ident, root, "m:p1", slot,
                                              handle, lo, hi, rights)))
    # Lost recipient authority between preparation and publication cannot
    # redirect the result or leave a partial table conversion behind.
    s.exact(valid)
    s.step("move", "m:domain0", "m:held_domain")
    s.refused("publish")
    s.step("abort")
    s.step("move", "m:held_domain", "m:domain0")
    s.exact(valid)
    s.step("publish")
    s.refused("publish")
    assert s.machine.wallet["d0:resume"].linear
    s.refused("delin", "m:d0")
    # A second id cannot overwrite the occupied protected slot.
    s.refused_action(create_action(1, "m:p1", "d0:resume", "m:d1"))
    assert sum(cap.kind == "logical" for cap in s.machine.wallet.values()) == 1
    return s.refusals


def check_address_geometry():
    s = fixture()
    s.step("move", "d1:p1", "d0:imported")
    own = s.step("address_value", "d0:p0")
    foreign = s.step("address_value", "d0:imported")
    physical = s.step("address_value", "m:p3")
    assert 0 < physical < PHYSICAL_LIMIT < own < foreign
    assert s.access(0, "d0:imported", "store", 0, 23).phase == "done"
    assert s.access(0, "d0:imported").result.value == 23
    assert s.access(0, "d0:p0").result == Word()
    s.step("split", "d0:p0", WORDS, "d0:rest")
    s.step("cursor", "d0:p0", WORDS)  # Legal one-past equality at a split.
    assert s.step("address_value", "d0:p0") == s.step("address_value", "d0:rest")
    s.step("delin", "d0:rest")
    s.step("move", "d0:rest", "d1:alias")
    s.step("narrow", "d1:alias", WORDS, 2 * WORDS, "r")
    assert s.step("address_value", "d0:rest") == s.step("address_value", "d1:alias")

    t = Script()
    t.pages(6)
    t.create(0, 0)
    candidate = create_action(1, "m:p1", "d1:new", "m:dnew")

    def at(lo, hi):
        args = list(candidate.arguments)
        args[5:7] = lo, hi
        return Action("prepare_create", tuple(args))

    for lo, hi in ((0, 16), (PHYSICAL_LIMIT, PHYSICAL_LIMIT + 16),
                   (base(0), base(0) + WORDS), (base(0), base(0) + 16),
                   (ADDRESS_LIMIT - 16, ADDRESS_LIMIT),
                   (ADDRESS_LIMIT, ADDRESS_LIMIT + 16)):
        t.refused_action(at(lo, hi))
    t.step("begin_revoke", "m:h0", "m:oldroot")
    t.barrier()
    t.refused_action(at(base(0), base(0) + 16))
    t.step("begin_detach", "m:d0", "m:t0")
    t.barrier()
    t.refused_action(at(base(0), base(0) + 16))
    t.step("destroy", "m:t0")
    t.exact(at(base(0), base(0) + 16))
    t.step("publish")
    t.populate("new", 0, 3, 2)
    t.refused("issue", 0, "d0:p0")
    assert t.access(1, "d1:new").result == Word()
    return s.refusals + t.refusals


def check_anonymous_initialization():
    s = fixture()
    s.step("physical_store", "m:p3", 0, 4242)
    s.step("delin", "m:p7")
    s.step("physical_store_cap", "m:p3", 1, "m:p7")
    before = tuple(s.machine.memory[3])
    s.refused("prepare_populate", "m:d0", 8, "m:p3", ())
    assert tuple(s.machine.memory[3]) == before
    s.step("prepare_populate", "m:d0", 4, "m:p3", ())
    assert tuple(s.machine.memory[3]) == before
    s.step("abort")
    assert tuple(s.machine.memory[3]) == before
    s.populate(0, 4, 3)
    for address in range(4, 8):
        assert s.access(0, "d0:p0", "load", address).result == Word()
    assert s.access(0, "d0:p0", "cload", 5, destination="d0:injected").phase == "fault"
    return s.refusals


def check_architectural_partition():
    """The decided 64-bit partition, checked with the real constants.

    The walked geometry above is finite; this scenario applies the same range
    rule, kind classification, allocator reservation rule and binding word to
    the architectural numbers of the encoding decision. Pure arithmetic, no
    walk, and no statement about any bounds codec.
    """
    P, L, T, K = ARCH_PHYSICAL_LIMIT, ARCH_LOGICAL_BASE, ARCH_LOGICAL_LIMIT, ARCH_PAGE
    assert P == 1 << 56 and L == 1 << 57 and T == 1 << 63 and K == 4096
    refused = 0
    for lo, hi, ok in ((0, K, False), (P - K, P, False), (P, P + K, False),
                       (L - K, L, False), (L, L + K, True), (L, L + K - 1, False),
                       (L + 1, L + 1 + K, False), (T - K, T, False),
                       (T - 2 * K, T - K, True), ((1 << 64) - K, 1 << 64, False),
                       (L + K, L, False), (L, L + (1 << 40), True)):
        assert arch_partition_ok(lo, hi) is ok, (hex(lo), hex(hi))
        refused += not ok
    # Kind follows the region; the guard octant and the top half are no kind.
    for address, kind in ((0, "physical"), (0x80000000, "physical"),
                          (0xBC3BFFFF, "physical"), (P - 1, "physical"),
                          (P, None), (L - 1, None), (L, "logical"),
                          (T - 1, "logical"), (T, None), ((1 << 64) - 1, None)):
        assert arch_kind(address) == kind, hex(address)
    # A range that passes the rule never straddles the two regions, so a
    # capability's bounds fix its kind and narrowing cannot change it.
    assert arch_kind(L) == arch_kind(L + (1 << 40) - 1) == "logical"
    # The allocator's reservation rule: power-of-two size, base aligned to
    # twice the size. The review's window-crossing example is rejected, as is
    # a second 1 GiB reservation placed directly behind the first.
    assert reservation_ok(L, L + (1 << 30))
    assert reservation_ok(L + (1 << 31), L + (1 << 31) + (1 << 30))
    assert not reservation_ok((1 << 58) - K, (1 << 58) + K)
    assert not reservation_ok(L + (1 << 30), L + (1 << 31))
    assert not reservation_ok(L, L + 3 * K)
    assert reservation_ok(L + 2 * K, L + 3 * K)
    # Every reservation is also a valid partition range, not the converse.
    for lo, hi in ((L, L + (1 << 30)), (L + 2 * K, L + 3 * K)):
        assert arch_partition_ok(lo, hi)
    assert arch_partition_ok((1 << 58) - K, (1 << 58) + K)
    # Binding word: nonzero for every valid (id, gen), round-trips, and refuses
    # generation zero and the exhausted generation.
    assert pack_binding(0, 1) != 0
    top = pack_binding((1 << BINDING_ID_BITS) - 1, (1 << BINDING_GEN_BITS) - 1)
    assert top < (1 << 32) and unpack_binding(top) == ((1 << BINDING_ID_BITS) - 1,
                                                       (1 << BINDING_GEN_BITS) - 1)
    for ident, generation in ((0, 0), (0, 1 << BINDING_GEN_BITS), (1 << BINDING_ID_BITS, 1)):
        try:
            pack_binding(ident, generation)
        except Refused:
            refused += 1
        else:
            raise AssertionError("binding accepted out of range")
    return refused


SCENARIOS = (
    check_permissions_and_inputs, check_reclamation,
    check_cut_subtree_and_generation, check_old_root_and_present_destroy,
    check_pointer_tree_and_contexts, check_capability_transfers,
    check_preparation_races, check_generation_exhaustion_and_tokens,
    check_protected_delivery, check_address_geometry, check_anonymous_initialization,
    check_architectural_partition,
)


def mutation_script(name, mutant):
    """Return the full reproducible trace and its FIRST refusal/violation."""
    s = Script(mutant) if name.startswith("uncleared") or name == "monitor_delivery" \
        else fixture(mutant)
    try:
        if name == "monitor_delivery":
            s.pages(2)
            s.step("prepare_create", 0, "m:p0", "m:leak", "m:d0")
            s.step("publish")
        elif name == "unzeroed_frame":
            s.step("physical_store", "m:p3", 0, 4242)
            s.step("delin", "m:p7")
            s.step("physical_store_cap", "m:p3", 1, "m:p7")
            s.populate(0, 4, 3)
            assert s.access(0, "d0:p0", "load", 4).result == Word()
        elif name.startswith("uncleared"):
            s.pages()
            s.step("delin", "m:p2")
            page = "m:p0" if name == "uncleared_create" else "m:p1"
            s.step("physical_store_cap", page, 1, "m:p2")
            s.create(0, 0)
            if name == "uncleared_populate":
                s.populate(0, 0, 3, 1)
            assert s.machine.memory[int(page[3:])][1] == Word()
        elif name == "plain_detach":
            s.access(0, "d0:p0", "store", 0, 99)
            s.step("delin", "d0:p0")
            s.step("begin_revoke", "m:d0", "m:escaped")
            s.barrier()
        elif name == "walk_only_drain":
            request = s.step("issue", 0, "d0:p0", "store", 0, 99)
            for _ in range(3):
                s.step("walk", request)
            s.step("begin_revoke", "m:h2", "m:reclaimed")
            for hart in HARTS:
                s.step("invalidate", hart)
                s.step("drain", hart)
            s.step("finish")
        elif name == "weak_binding":
            s.access(1, "d1:p1", "store", 0, 99)
        elif name == "duplicate_return":
            s.step("begin_revoke", "m:h0", "m:root")
            s.barrier()
            s.step("begin_detach", "m:d0", "m:t0")
            s.barrier()
            s.step("destroy", "m:t0")
        elif name == "root_release":
            s.step("begin_revoke", "m:h0", "m:root")
            s.barrier()
        elif name == "stale_record":
            s.step("begin_detach", "m:d0", "m:t0")
            s.barrier()
            s.step("destroy", "m:t0")
            s.create(0, 3, "new")
            s.populate("new", 0, 5, 4)
            s.step("begin_revoke", "m:h0", "m:oldroot")
            s.barrier()
            assert s.access(0, "d0:pnew").phase == "done"
    except (Violation, Refused) as error:
        return type(error).__name__, str(error), s.trace
    return "safe", "", s.trace


EXPECTED_MUTATIONS = {
    "plain_detach": "I3:",
    "uncleared_create": "I4:",
    "uncleared_populate": "I4:",
    "walk_only_drain": "I6:",
    "weak_binding": "goal: live pointer redirected",
    "duplicate_return": "I1:",
    "root_release": "I5: registry reservation",
    "stale_record": "I5: registry reservation",
    "monitor_delivery": "I3:",
    "unzeroed_frame": "I4: anonymous frame not zeroed",
}


def mutations():
    results = []
    for name in MUTANTS:
        good, reason, _ = mutation_script(name, None)
        expected_control = {
            "plain_detach": ("Refused", "wrong capability kind"),
            "walk_only_drain": ("Refused", "data access outstanding"),
            "monitor_delivery": ("Refused", "protected resume slot required"),
        }.get(name, ("safe", ""))
        assert (good, reason) == expected_control, (name, "correct model failed", good, reason)
        bad, reason, trace = mutation_script(name, name)
        assert bad == "Violation" and reason.startswith(EXPECTED_MUTATIONS[name]), (
            name, "mutation not detected by intended property", bad, reason)
        results.append({"variant": name, "control": good, "violation": reason,
                        "trace": trace})
    return results


def schedule_actions(machine, trigger):
    actions = []
    if not machine.completed and machine.barrier is None:
        actions.append(trigger)
    for request, access in machine.accesses.items():
        if access.phase in ("root", "leaf", "fill"):
            actions.append(Action("walk", (request,)))
        if access.phase == "checked":
            actions.append(Action("memory_step", (request,)))
        if access.phase not in ("done", "fault", "cancelled"):
            actions.append(Action("cancel", (request,)))
    if machine.barrier:
        for hart in HARTS:
            actions.append(Action("invalidate", (hart,)))
            actions.append(Action("drain", (hart,)))
        actions.append(Action("finish"))
    return actions


def lifecycle_actions(machine):
    actions = []
    if machine.preparation:
        actions += [Action("publish"), Action("abort")]
    if machine.barrier:
        actions += schedule_actions(machine, Action("finish"))
        return actions
    for location, cap in list(machine.wallet.items()):
        if not machine.live(cap.node):
            continue
        if cap.kind == "detach":
            actions.append(Action("begin_detach", (location, "m:token%d" % cap.binding[0])))
            for address in (base(cap.binding[0]), base(cap.binding[0]) + WORDS,
                            base(cap.binding[0]) + 2 * WORDS):
                for frame, table in (("m:p3", ()), ("m:p4", ("m:p5",))):
                    actions.append(Action("prepare_populate", (location, address, frame, table)))
        if cap.kind in ("rev_phys", "rev_logical"):
            output = location.split(":", 1)[0] + ":back:" + location
            actions.append(Action("begin_revoke", (location, output)))
        if cap.kind == "token":
            actions += [Action("destroy", (location,)),
                        Action("begin_unmap", (location, base(cap.binding[0]), "m:unmapped"))]
        if cap.kind == "uninit":
            actions += [Action("scrub", (location,)), Action("init", (location,))]
        if cap.kind == "logical" and location.startswith(("d0:", "d1:")):
            hart = int(location[1])
            actions += [Action("delin", (location,)), Action("drop", (location,)),
                        Action("split", (location, base(cap.binding[0]) + WORDS, location + ":split")),
                        Action("mrev", (location, location + ":rev")),
                        Action("move", (location, "d%d:copy" % (1 - hart)))]
            for address in (base(cap.binding[0]), base(cap.binding[0]) + WORDS,
                            base(cap.binding[0]) + 2 * WORDS):
                for operation in ("load", "store", "amo"):
                    actions.append(Action("issue", (hart, location, operation, address)))
    actions += [create_action(i, "m:p6", "d%d:new" % i, "m:new%d" % i)
                for i in range(2)]
    for request, access in machine.accesses.items():
        if access.phase in ("root", "leaf", "fill"):
            actions.append(Action("walk", (request,)))
        if access.phase == "checked":
            actions.append(Action("memory_step", (request,)))
    return actions


def successors(machine, actions):
    for action in actions:
        next_machine = machine.clone()
        try:
            action.apply(next_machine)
        except Refused:
            assert next_machine.key() == machine.key(), "non-atomic refusal: %s" % action
            continue
        try:
            next_machine.check()
        except Violation as error:
            raise Violation("%s after %s" % (error, action)) from error
        yield action, next_machine


def explore(seed, actions, depth=None, limit=100000, terminal_check=None):
    queue = deque([(seed, 0)])
    seen = {seed.key()}
    edges = 0
    terminals = 0
    max_depth = 0
    frontier = 0
    coverage = Counter()
    while queue:
        state, distance = queue.popleft()
        max_depth = max(max_depth, distance)
        if depth is not None and distance == depth:
            frontier += 1
            continue
        count = 0
        for action, child in successors(state, actions(state)):
            count += 1
            edges += 1
            coverage[action.operation] += 1
            if action.operation == "finish":
                coverage["finish_" + state.barrier.operation] += 1
            if state.barrier and action.operation == "issue":
                coverage["issue_during_barrier"] += 1
                if action.arguments[0] in state.barrier.drained:
                    coverage["issue_after_hart_drain"] += 1
            if state.barrier and action.operation == "memory_step":
                access = state.accesses[action.arguments[0]]
                if not state._blocked(access.cap.binding):
                    coverage["foreign_memory_during_barrier"] += 1
            if action.operation == "finish" and any(
                    a.phase not in ("done", "fault", "cancelled") and
                    not state._blocked(a.cap.binding) for a in state.accesses.values()):
                coverage["finish_with_foreign_pending"] += 1
            key = child.key()
            if key not in seen:
                if len(seen) == limit:
                    raise RuntimeError("state limit reached; exploration INCOMPLETE")
                seen.add(key)
                queue.append((child, distance + 1))
        if not count:
            terminals += 1
            if depth is None:
                assert state.completed and state.barrier is None, "unfinished barrier deadlock"
                assert all(a.phase in ("done", "fault", "cancelled")
                           for a in state.accesses.values()), "unfinished data access"
                if terminal_check:
                    terminal_check(state)
    return {"states": len(seen), "edges": edges, "terminals": terminals,
            "max_depth": max_depth, "depth_frontier": frontier,
            "coverage": dict(sorted(coverage.items()))}


def interleavings(barrier_mode="global"):
    results = []
    for operation in ("load", "store", "amo", "cload", "cstore"):
        for victim in ("frame", "table", "root", "detach"):
            for warm in (False, True):
                s = fixture(barrier_mode=barrier_mode)
                if operation == "cload":
                    s.step("move", "d1:p1", "d0:payload")
                    s.access(0, "d0:p0", "cstore", source="d0:payload")
                s.step("delin", "d0:p0")
                s.step("move", "d0:p0", "d1:foreign")
                if warm:
                    s.access(0, "d0:p0")
                    s.access(1, "d1:foreign")
                for hart, location in ((0, "d0:p0"), (1, "d1:foreign")):
                    source = None
                    if operation == "cstore":
                        # Two distinct linear logical capabilities, one per hart.
                        if hart == 0:
                            s.step("split", "d1:p1", WORDS, "d1:rest")
                            s.step("move", "d1:p1", "d0:payload")
                        source = "d0:payload" if hart == 0 else "d1:rest"
                    s.step("issue", hart, location, operation, 0, 99, source)
                if victim == "detach":
                    trigger = Action("begin_detach", ("m:d0", "m:token"))
                else:
                    page = {"frame": 2, "table": 1, "root": 0}[victim]
                    trigger = Action("begin_revoke", ("m:h%d" % page, "m:reclaimed"))
                result = explore(s.machine, lambda m: schedule_actions(m, trigger))
                assert result["coverage"].get("memory_step", 0) > 0
                assert result["coverage"].get("finish", 0) > 0
                result.update(operation=operation, victim=victim, warm=warm,
                              barrier_mode=barrier_mode, removal_bound=1)
                results.append(result)
    return results


def check_table_record():
    """Refusal/permission controls for the optional protected record."""
    for mode in ("global", "table_record"):
        s = fixture(barrier_mode=mode)
        s.step("begin_revoke", "m:h1", "m:table")
        s.refused("issue", 0, "d0:p0", "store")
        if mode == "global":
            s.refused("issue", 1, "d1:p1", "store")
            s.barrier()
            continue
        assert s.machine.barrier.scope == frozenset(((0, 1),))
        req = s.step("issue", 1, "d1:p1", "store")
        for _ in range(3):
            s.step("walk", req)
        s.step("invalidate", 1)
        assert (1, 1, 1, base(1) // WORDS) in s.machine.tlb
        s.step("drain", 1)  # Foreign checked store does not hold up this return.
        s.step("memory_step", req)
        req = s.step("issue", 1, "d1:p1", "load")
        s.step("invalidate", 0)
        s.step("drain", 0)
        s.step("finish")
        s.step("memory_step", req)
        assert s.machine.accesses[req].result == Word(7, True)
        s.step("begin_revoke", "m:h2", "m:frame")
        assert s.machine.barrier.scope is None  # Unindexed frame => global fallback.
        s.refused("issue", 1, "d1:p1")
        s.barrier()

    s = fixture(barrier_mode="table_record")
    s.step("begin_detach", "m:d0", "m:token")
    s.barrier()
    s.step("destroy", "m:token")
    s.create(0, 6, "new")
    s.populate("new", 0, 4, 5)
    s.step("begin_revoke", "m:h0", "m:oldroot")
    assert s.machine.barrier.scope == frozenset(((0, 1),))
    assert s.machine.registry[0].binding == (0, 2)
    assert s.access(0, "d0:pnew", "store").phase == "done"
    s.barrier()
    assert s.access(0, "d0:pnew").result == Word(7, True)

    # Fault injection is confined to setup: misattribute a root record. A
    # checked victim store must make this scoped return fail at I6.
    s = fixture(barrier_mode="table_record")
    req = s.step("issue", 0, "d0:p0", "store")
    for _ in range(3):
        s.step("walk", req)
    s.machine.table_bindings[s.machine.registry[0].root.node] = (1, 1)
    s.step("begin_revoke", "m:h0", "m:badroot")
    for hart in HARTS:
        s.step("invalidate", hart)
        s.step("drain", hart)
    try:
        s.step("finish")
    except Violation as error:
        assert str(error).startswith("I6:"), str(error)
        return {"wrong_record_control": str(error), "trace": s.trace}
    raise AssertionError("misattributed table record escaped completion oracle")


def record_interference():
    results = []
    for operation in ("load", "store", "amo", "cload", "cstore"):
        for victim in ("table", "root", "detach"):
            s = fixture(barrier_mode="table_record")
            if operation in ("cload", "cstore"):
                s.step("move", "m:p11", "d1:payload")
                if operation == "cload":
                    s.access(1, "d1:p1", "cstore", source="d1:payload")
            s.step("issue", 0, "d0:p0", "store")
            issued = s.machine.next_access
            first = (Action("begin_detach", ("m:d0", "m:token")) if victim == "detach"
                     else Action("begin_revoke", ("m:h1" if victim == "table" else "m:h0",
                                                  "m:first")))
            second = Action("begin_revoke", ("m:h1" if victim == "root" else "m:h0", "m:second"))

            def actions(m):
                result = schedule_actions(m, first)
                if m.barrier is None and len(m.completed) == 1:
                    result.append(second)
                # One late access, scheduled during either barrier or between /
                # after them. Before the first break only the victim is issued.
                if m.next_access == issued and (m.barrier or m.completed):
                    source = "d1:payload" if operation == "cstore" else None
                    result.append(Action("issue", (1, "d1:p1", operation, base(1), 99, source)))
                return result

            def terminal(m):
                assert len(m.completed) == 2, "second removal missing"
                assert m.next_access == issued + 1, "late foreign issue missing"

            result = explore(s.machine, actions, terminal_check=terminal)
            for event in ("issue_during_barrier", "issue_after_hart_drain",
                          "foreign_memory_during_barrier", "finish_with_foreign_pending"):
                assert result["coverage"].get(event, 0), "missing scoped interleaving: " + event
            result.update(operation=operation, first_victim=victim,
                          barrier_mode="table_record", removal_bound=2,
                          late_access_bound=1)
            results.append(result)
    return results


def lifecycle(depth):
    """Reduced alphabets, explicit microsteps, three reachable starting states."""
    results = []
    aggregate = Counter()
    for phase in ("active", "detached", "destroyed"):
        s = fixture()
        s.step("mrev", "d0:p0", "d0:rev")
        if phase != "active":
            s.step("begin_detach", "m:d0", "m:token0")
            s.barrier()
        if phase == "destroyed":
            s.step("destroy", "m:token0")
        initial_completed = len(s.machine.completed)

        def actions(m):
            result = schedule_actions(m, Action("finish"))
            if m.preparation:
                result += [Action("publish"), Action("abort")]
            if m.barrier:
                return result
            # One candidate per removal kind; at most one new barrier per path.
            if len(m.completed) == initial_completed:
                for handle in ("m:h0", "m:h1", "m:h2", "d0:rev"):
                    result.append(Action("begin_revoke", (handle, handle + ":back")))
                result += [Action("begin_detach", ("m:d0", "m:token0")),
                           Action("begin_unmap", ("m:token0", base(0), "m:unmapped"))]
            result += [Action("destroy", ("m:token0",)),
                       create_action(0, "m:p6", "d0:new", "m:new0")]
            # A used leaf and a missing lower table exercise both publications.
            handle = "m:new0" if "m:new0" in m.wallet else "m:d0"
            result += [Action("prepare_populate", (handle, base(0) + WORDS, "m:p3", ())),
                       Action("prepare_populate", (handle, base(0) + 2 * WORDS, "m:p4", ("m:p5",)))]
            if not m.accesses:
                pointer = "d0:new" if "d0:new" in m.wallet else "d0:p0"
                result.append(Action("issue", (0, pointer, "store", base(0))))
            return result

        result = explore(s.machine, actions, depth=depth)
        result.update(initial_state=phase, depth_bound=depth,
                      setup_steps=len(s.trace), additional_barrier_bound=1,
                      additional_access_bound=1)
        aggregate.update(result["coverage"])
        results.append(result)
    required = ("finish_revoke", "finish_detach", "finish_unmap", "memory_step",
                "destroy", "begin_unmap", "publish")
    if depth >= 6:
        assert all(aggregate[k] for k in required), "lifecycle completion coverage missing"
    return {"families": results, "depth_bound": depth,
            "completion_gate": depth >= 6,
            "states": sum(r["states"] for r in results),
            "coverage": dict(sorted(aggregate.items()))}


def randomized(seed_count, steps):
    aggregate = Counter()
    total = 0
    records = []
    revoke_limit = 2
    removal_limit = 3
    minimum_effects = 4
    for seed in range(seed_count):
        rng = random.Random(seed)
        machine = fixture().machine
        trace = []
        coverage = Counter()
        for _ in range(steps):
            groups = {}
            for action in lifecycle_actions(machine):
                op = action.operation
                if op == "issue" and action.arguments[3] != base(
                        machine.wallet[action.arguments[1]].binding[0]):
                    continue
                if op == "begin_revoke" and coverage[op] >= revoke_limit:
                    continue
                if op in ("begin_revoke", "begin_detach", "drop") and sum(
                        coverage[k] for k in ("begin_revoke", "begin_detach", "drop")) >= removal_limit:
                    continue
                cap = machine.wallet.get(action.arguments[0]) if action.arguments else None
                # Keep mapping 1 backed and its pointer live, so each trace must
                # exercise data even after adversarial teardown of mapping 0.
                if cap and op != "issue" and (cap.binding == (1, 1) or
                                              cap.page in (8, 9, 10)):
                    continue
                groups.setdefault(op, []).append(action)
            # Choose opcode groups by weight, then operands uniformly within
            # the group. Many REVOKE operands no longer multiply its weight.
            weights = {"issue": 6, "walk": 8, "memory_step": 8,
                       "publish": 4, "finish": 4, "begin_revoke": 1.5,
                       "cancel": 0.2}
            order = sorted(groups, key=lambda op: rng.random() ** (1 / weights.get(op, 1)),
                           reverse=True)
            actions = []
            for op in order:
                rng.shuffle(groups[op])
                actions.extend(groups[op])
            for action in actions:
                child = machine.clone()
                try:
                    action.apply(child)
                except Refused:
                    assert child.key() == machine.key(), "refusal changed state"
                    continue
                trace.append(str(action))
                try:
                    child.check()
                except Violation as error:
                    raise Violation("seed=%d trace=%s: %s" % (seed, trace, error)) from error
                aggregate[action.operation] += 1
                coverage[action.operation] += 1
                total += 1
                machine = child
                break
            else:
                break
        effects = Counter(effect[3] for effect in machine.effects)
        assert len(machine.effects) >= minimum_effects, (
            "seed=%d completed only %d data effects; minimum=%d" %
            (seed, len(machine.effects), minimum_effects))
        assert effects["load"] and effects["store"], "seed=%d missed load/store effects" % seed
        records.append({"seed": seed, "steps": len(trace), "effects": dict(sorted(effects.items())),
                        "memory_effects": len(machine.effects),
                        "coverage": dict(sorted(coverage.items()))})
    return {"seeds": seed_count, "step_bound": steps, "steps": total,
            "per_seed": records, "minimum_effects_required": minimum_effects,
            "minimum_effects_measured": min(r["memory_effects"] for r in records),
            "revoke_limit": revoke_limit, "removal_limit": removal_limit,
            "protected_control_mapping": [1, 1],
            "coverage": dict(sorted(aggregate.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if not __debug__:
        parser.error("run without -O/PYTHONOPTIMIZE; the harness needs assertions")
    parser.add_argument("--output", type=Path, help="write deterministic JSON result lines")
    parser.add_argument("--depth", type=int, default=6, help="reduced lifecycle BFS depth")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--replay", choices=MUTANTS)
    parser.add_argument("--scenarios-only", action="store_true")
    args = parser.parse_args()
    if args.replay:
        good = mutation_script(args.replay, None)
        bad = mutation_script(args.replay, args.replay)
        print(json.dumps({"control": good, "mutant": bad}, indent=2))
        return 0 if good[0] in ("safe", "Refused") and bad[0] == "Violation" else 1
    if args.depth < 1 or args.seeds < 1 or args.steps < 1:
        parser.error("bounds must be positive")
    result = {"schema": 4, "scope": "Stage-1 abstract model; not QEMU/RTL qualification",
              "contract_base": "00626a232bda",
              "contract_revision": "global ranges, protected delivery, anonymous zeroing, architectural partition",
              "partition": {"physical_bits": ARCH_PHYSICAL_BITS,
                            "logical_base": hex(ARCH_LOGICAL_BASE),
                            "logical_limit": hex(ARCH_LOGICAL_LIMIT),
                            "page": ARCH_PAGE,
                            "binding_bits": [BINDING_ID_BITS, BINDING_GEN_BITS],
                            "registry_entries": 1 << BINDING_ID_BITS},
              "scenarios": [], "mutations": [],
              "python": sys.version.split()[0],
              "source_sha256": {name: hashlib.sha256(
                  Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("model.py", "check.py")}}
    for scenario in SCENARIOS:
        refusals = scenario()
        result["scenarios"].append({"name": scenario.__name__, "refusals": refusals})
        print("PASS", scenario.__name__, flush=True)
    result["mutations"] = mutations()
    print("PASS mutation controls %d/%d" % (len(result["mutations"]), len(MUTANTS)), flush=True)
    result["table_record_controls"] = check_table_record()
    print("PASS table-record scope and fault-injection controls", flush=True)
    if not args.scenarios_only:
        result["interleavings"] = interleavings()
        print("PASS interleaving families %d" % len(result["interleavings"]), flush=True)
        result["record_interleavings"] = interleavings("table_record")
        print("PASS table-record repeated families %d" % len(result["record_interleavings"]), flush=True)
        result["record_interference"] = record_interference()
        print("PASS table-record interference families %d" % len(result["record_interference"]), flush=True)
        result["lifecycle"] = lifecycle(args.depth)
        print("PASS lifecycle states %d" % result["lifecycle"]["states"], flush=True)
        result["random"] = randomized(args.seeds, args.steps)
        print("PASS seeded steps %d" % result["random"]["steps"], flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, Violation, RuntimeError) as failure:
        print("FAIL:", failure, file=sys.stderr)
        raise
