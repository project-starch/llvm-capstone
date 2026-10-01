#!/usr/bin/env python3
"""Bounded transition model for the trusted-Linux M1 boundary; see README.md."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations
import json
from pathlib import Path

ABSENT, RO, RW = 0, 1, 2
ACTIVE, RETIRING, DEAD = 0, 1, 2
ADDRESS, OBJECT_BYTES, MAX_GENERATION = 0x1000, 64, 1
VARIANTS = frozenset({
    "stale_lifetime_selector", "stale_translation_selector", "stale_asid_cache",
    "no_free_drain", "no_unmap_drain", "fault_retry_skips_liveness",
    "pte_write_bypass", "syscall_clamps_request", "clone_shares_backing",
    "clone_revives_dead_node", "reuse_keeps_generation", "mmap_keeps_generation",
    "clone_reuses_context", "reuse_grants_write", "invalidate_keeps_cache",
    "rejects_fresh_generation", "drain_ignores_hart1", "retry_always_denies",
})


def updated(items: tuple, index: int, value: object) -> tuple:
    result = list(items)
    result[index] = value
    return tuple(result)


@dataclass(frozen=True)
class Capability:
    generation: int = 0
    address: int = ADDRESS
    base: int = ADDRESS
    end: int = ADDRESS + OBJECT_BYTES
    tagged: bool = True
    readable: bool = True
    writable: bool = True
    # Ghost provenance. Guards NEVER consult these two fields. They allow the
    # observer to detect stale authority even if an implementation reuses gen.
    namespace: int = 0
    birth: int = 0

    def permits(self, width: int, write: bool) -> bool:
        return (self.tagged and (self.writable if write else self.readable)
                and self.base <= self.address <= self.end
                and 0 <= width <= self.end - self.address)


@dataclass(frozen=True)
class Pending:
    context: int
    frame: int
    epoch: int
    write: bool


@dataclass(frozen=True)
class Fault:
    context: int
    capability: Capability
    write: bool


@dataclass(frozen=True)
class State:
    user: tuple[int, int] = (0, 1)
    lifetime: tuple[int, int] = (0, 1)
    translation: tuple[int, int] = (0, 1)
    asid: tuple[int, int] = (0, 0)  # intentionally reused
    cache: tuple[tuple[int, int] | None, ...] = (None, None)
    # A namespace instance is never recycled. DEAD describes an object, not
    # an unused process context. Only clone may claim a virgin context slot.
    context_used: tuple[bool, bool] = (True, True)
    live: tuple[bool, bool] = (True, True)
    generation: tuple[int, int] = (0, 0)  # -1 means no identity issued yet
    phase: tuple[int, int] = (ACTIVE, ACTIVE)
    unmapping: tuple[bool, bool] = (False, False)
    vma: tuple[bool, bool] = (True, True)
    vm_rights: tuple[int, int] = (RW, RW)
    pte: tuple[int, int] = (RW, RW)
    frame: tuple[int, int] = (0, 1)
    data: tuple[int, int] = (0, 0)
    # Ghost allocation identities and backing incarnations, never guards.
    birth: tuple[int, int] = (0, 0)
    epoch: tuple[int, int] = (0, 0)
    p: tuple[Capability | None, ...] = (Capability(), Capability(namespace=1))
    q: tuple[Capability | None, ...] = (None, None)
    invalidated: tuple[tuple[bool, bool], ...] = ((False, False), (False, False))
    drained: tuple[tuple[bool, bool], ...] = ((False, False), (False, False))
    pending: tuple[Pending | None, ...] = (None, None)
    fault: tuple[Fault | None, ...] = (None, None)
    input_bytes: int = 32


@dataclass(frozen=True)
class Step:
    state: State
    outcome: str
    violation: str | None = None


def refused(state: State) -> Step:
    return Step(state, "refused")


def authority_violation(state: State, context: int, cap: Capability) -> str | None:
    """Observer only: its result reports a bug and never authorizes access."""
    if cap.namespace != context:
        return "foreign_capability"
    if cap.birth != state.birth[context]:
        return "stale_allocation"
    if not state.vma[context] or not state.live[context] or cap.generation != state.generation[context]:
        return "object_authority"
    return None


def switch(state: State, hart: int, context: int, variant: str | None) -> Step:
    if hart not in (0, 1) or context not in (0, 1):
        return refused(state)
    if state.pending[hart] is not None or not state.context_used[context]:
        return refused(state)
    lifetime = state.lifetime if variant == "stale_lifetime_selector" else updated(state.lifetime, hart, context)
    translation = state.translation if variant == "stale_translation_selector" else updated(state.translation, hart, context)
    cache = state.cache if variant == "stale_asid_cache" else updated(state.cache, hart, None)
    return Step(replace(state, user=updated(state.user, hart, context),
                        lifetime=lifetime, translation=translation, cache=cache), "switch")


def issue(state: State, hart: int, cap: Capability, write: bool,
          variant: str | None, *, retry_prechecked: bool = False) -> Step:
    if state.pending[hart] is not None or state.fault[hart] is not None:
        return refused(state)
    context = state.user[hart]
    bypass = retry_prechecked and variant == "fault_retry_skips_liveness"
    if not cap.permits(1, write) or state.phase[context] != ACTIVE and not bypass:
        return Step(state, "cap_fault")
    if variant == "rejects_fresh_generation" and cap.generation > 0:
        return Step(state, "cap_fault")
    lifetime = state.lifetime[hart]
    translation = state.translation[hart]
    cached = state.cache[hart] == (state.asid[context], cap.generation)
    authorized = (bypass or cached or state.vma[lifetime] and state.live[lifetime]
                  and state.generation[lifetime] == cap.generation)
    if not authorized:
        return Step(state, "cap_fault")
    violation = authority_violation(state, context, cap)
    if violation:
        return Step(state, "authorized", violation)
    permission = state.pte[translation]
    if permission == ABSENT:
        return Step(replace(state, fault=updated(state.fault, hart,
                                                 Fault(context, cap, write))), "page_fault")
    if write and permission != RW and variant != "pte_write_bypass":
        return Step(state, "page_permission_fault")
    frame = state.frame[translation]
    next_state = replace(state, pending=updated(state.pending, hart,
                                                Pending(context, frame, state.epoch[frame], write)),
                         cache=updated(state.cache, hart, (state.asid[context], cap.generation)))
    if state.pte[context] == ABSENT or write and state.pte[context] != RW:
        return Step(next_state, "issued", "page_permission")
    if frame != state.frame[context]:
        return Step(next_state, "issued", "foreign_translation")
    return Step(next_state, "issued")


def access(state: State, hart: int, slot: str, write: bool, variant: str | None) -> Step:
    cap = getattr(state, slot)[state.user[hart]]
    return refused(state) if cap is None else issue(state, hart, cap, write, variant)


def complete(state: State, hart: int) -> Step:
    access = state.pending[hart]
    if access is None:
        return refused(state)
    data = (updated(state.data, access.frame, state.data[access.frame] + 1)
            if access.write else state.data)
    next_state = replace(state, pending=updated(state.pending, hart, None), data=data)
    if state.phase[access.context] == DEAD or state.epoch[access.frame] != access.epoch:
        return Step(next_state, "completed", "late_access_after_reuse_permission")
    return Step(next_state, "completed")


def retire(state: State, context: int, *, unmap: bool = False) -> Step:
    if state.phase[context] != ACTIVE or not state.live[context]:
        return refused(state)
    return Step(replace(state, live=updated(state.live, context, False),
                        phase=updated(state.phase, context, RETIRING),
                        unmapping=updated(state.unmapping, context, unmap),
                        pte=updated(state.pte, context, ABSENT) if unmap else state.pte,
                        vma=updated(state.vma, context, False) if unmap else state.vma,
                        invalidated=updated(state.invalidated, context, (False, False)),
                        drained=updated(state.drained, context, (False, False))), "retiring")


def invalidate(state: State, context: int, hart: int, variant: str | None) -> Step:
    if state.phase[context] != RETIRING or state.invalidated[context][hart]:
        return refused(state)
    cache = state.cache if variant == "invalidate_keeps_cache" else updated(state.cache, hart, None)
    return Step(replace(state, cache=cache,
                        invalidated=updated(state.invalidated, context,
                                            updated(state.invalidated[context], hart, True))), "invalidated")


def drain(state: State, context: int, hart: int, variant: str | None) -> Step:
    if state.phase[context] != RETIRING or not state.invalidated[context][hart] or state.drained[context][hart]:
        return refused(state)
    access = state.pending[hart]
    busy = access is not None and access.context == context
    if busy and not (variant == "drain_ignores_hart1" and hart == 1):
        return refused(state)
    next_state = replace(state, drained=updated(state.drained, context,
                                               updated(state.drained[context], hart, True)))
    return Step(next_state, "drained", "drain_ack_with_access_pending" if busy else None)


def finish(state: State, context: int, variant: str | None) -> Step:
    if state.phase[context] != RETIRING or state.invalidated[context] != (True, True):
        return refused(state)
    skip = variant == ("no_unmap_drain" if state.unmapping[context] else "no_free_drain")
    if state.drained[context] != (True, True) and not skip:
        return refused(state)
    next_state = replace(state, phase=updated(state.phase, context, DEAD))
    if any(p is not None and p.context == context for p in state.pending):
        return Step(next_state, "finished", "reuse_authorized_with_access_pending")
    if state.drained[context] != (True, True):
        return Step(next_state, "finished", "finish_without_drain_ack")
    return Step(next_state, "finished")


def allocate_identity(state: State, context: int, variant: str | None, *, mapping: bool) -> State:
    """Shared allocation update; callers enforce mapping/allocator authority."""
    old = state.generation[context]
    keep = variant == ("mmap_keeps_generation" if mapping else "reuse_keeps_generation")
    generation = old if keep and old >= 0 else old + 1
    # Independent observer: a successful allocation always has a new birth,
    # including when the injected implementation mistakenly keeps generation.
    birth = state.birth[context] + 1
    cap = Capability(generation=generation, namespace=context, birth=birth)
    slot = "p" if state.p[context] is None else "q"
    return replace(state, live=updated(state.live, context, True),
                   phase=updated(state.phase, context, ACTIVE),
                   generation=updated(state.generation, context, generation),
                   birth=updated(state.birth, context, birth),
                   epoch=updated(state.epoch, state.frame[context], state.epoch[state.frame[context]] + 1),
                   **{slot: updated(getattr(state, slot), context, cap)})


def reuse(state: State, context: int, variant: str | None) -> Step:
    """malloc(64) inside an existing arena: never create a VMA or change PTEs."""
    if (not state.context_used[context] or not state.vma[context]
            or state.phase[context] != DEAD or state.generation[context] >= MAX_GENERATION):
        return refused(state)
    next_state = allocate_identity(state, context, variant, mapping=False)
    if variant == "reuse_grants_write":
        next_state = replace(next_state, pte=updated(next_state.pte, context, RW))
    if next_state.pte != state.pte or next_state.vm_rights != state.vm_rights:
        return Step(next_state, "allocated", "allocator_changed_page_permissions")
    return Step(next_state, "allocated")


def mmap_one_page(state: State, context: int, variant: str | None, rights: int = RW) -> Step:
    if (not state.context_used[context] or state.vma[context] or state.phase[context] != DEAD
            or state.generation[context] >= MAX_GENERATION or rights not in (RO, RW)):
        return refused(state)
    # The trusted ABI supplies the arena plus the sole modeled object. The
    # identity advances on every mapping, including at a previously used VA.
    next_state = allocate_identity(state, context, variant, mapping=True)
    return Step(replace(next_state, vma=updated(state.vma, context, True),
                        vm_rights=updated(state.vm_rights, context, rights),
                        pte=updated(state.pte, context, ABSENT)), "mapped")


def resolve_fault(state: State, hart: int) -> Step:
    fault = state.fault[hart]
    if fault is None or not state.vma[fault.context] or state.pte[fault.context] != ABSENT:
        return refused(state)
    return Step(replace(state, pte=updated(state.pte, fault.context,
                                          state.vm_rights[fault.context])), "resolved")


def retry(state: State, hart: int, variant: str | None) -> Step:
    fault = state.fault[hart]
    if fault is None or state.user[hart] != fault.context:
        return refused(state)
    cleared = replace(state, fault=updated(state.fault, hart, None))
    if variant == "retry_always_denies":
        return Step(cleared, "retry_denied")
    step = issue(cleared, hart, fault.capability, fault.write, variant, retry_prechecked=True)
    if step.outcome in ("cap_fault", "refused"):
        return Step(cleared, "retry_denied")
    return step


def clone_private(state: State, variant: str | None) -> Step:
    """Fork into a virgin namespace instance; retired contexts are not virgin."""
    if state.context_used[1] and variant != "clone_reuses_context":
        return refused(state)
    if state.phase[1] != DEAD or state.vma[1] or any(state.pending) or state.phase[0] == RETIRING:
        return refused(state)
    live, phase = state.live[0], state.phase[0]
    if variant == "clone_revives_dead_node" and not live:
        live, phase = True, ACTIVE
    frame = state.frame[0] if variant == "clone_shares_backing" else state.frame[1]
    def rebind(cap: Capability | None) -> Capability | None:
        # Privileged clone preserves representation and rebases only ghost
        # provenance into the independently owned child namespace.
        return None if cap is None else replace(cap, namespace=1)
    next_state = replace(state, context_used=updated(state.context_used, 1, True),
                         live=updated(state.live, 1, live), phase=updated(state.phase, 1, phase),
                         generation=updated(state.generation, 1, state.generation[0]),
                         birth=updated(state.birth, 1, state.birth[0]),
                         vma=updated(state.vma, 1, state.vma[0]),
                         vm_rights=updated(state.vm_rights, 1, state.vm_rights[0]),
                         pte=updated(state.pte, 1, state.pte[0]),
                         frame=updated(state.frame, 1, frame),
                         data=updated(state.data, frame, state.data[state.frame[0]]),
                         p=updated(state.p, 1, rebind(state.p[0])),
                         q=updated(state.q, 1, rebind(state.q[0])))
    if state.context_used[1]:
        return Step(next_state, "cloned", "reused_context_instance")
    if frame == state.frame[0]:
        return Step(next_state, "cloned", "private_backing_shared")
    if live and not state.live[0]:
        return Step(next_state, "cloned", "revived_dead_identity")
    return Step(next_state, "cloned")


def read_into_object(state: State, context: int, cap: Capability,
                     requested: int, variant: str | None) -> Step:
    if (not cap.permits(0, True) or requested < 0 or not state.live[context]
            or state.generation[context] != cap.generation):
        return Step(state, "EFAULT")
    fits = cap.permits(requested, True)
    if not fits and variant != "syscall_clamps_request":
        return Step(state, "EFAULT")
    violation = authority_violation(state, context, cap)
    if violation:
        return Step(state, "authorized", violation)
    copied = min(requested, cap.end - cap.address, state.input_bytes)
    next_state = replace(state, input_bytes=state.input_bytes - copied)
    return Step(next_state, f"read:{copied}", None if fits else "syscall_consumed_on_bad_span")


ACTIONS = {
    "warm": lambda s, v: access(s, 0, "p", False, v),
    "switch1": lambda s, v: switch(s, 0, 1, v),
    "access1": lambda s, v: access(s, 0, "p", False, v),
    "issue0": lambda s, v: access(s, 0, "p", True, v),
    "issue1": lambda s, v: access(s, 1, "p", True, v),
    "load_p0": lambda s, v: access(s, 0, "p", False, v),
    "load_p1": lambda s, v: access(s, 1, "p", False, v),
    "load_q0": lambda s, v: access(s, 0, "q", False, v),
    "load_q1": lambda s, v: access(s, 1, "q", False, v),
    "store_q0": lambda s, v: access(s, 0, "q", True, v),
    "store_q1": lambda s, v: access(s, 1, "q", True, v),
    "complete0": lambda s, v: complete(s, 0),
    "complete1": lambda s, v: complete(s, 1),
    "retire0": lambda s, v: retire(s, 0),
    "unmap0": lambda s, v: retire(s, 0, unmap=True),
    "unmap1": lambda s, v: retire(s, 1, unmap=True),
    "invalidate0": lambda s, v: invalidate(s, 0, 0, v),
    "invalidate1": lambda s, v: invalidate(s, 0, 1, v),
    "drain0": lambda s, v: drain(s, 0, 0, v),
    "drain1": lambda s, v: drain(s, 0, 1, v),
    "finish0": lambda s, v: finish(s, 0, v),
    "invalidate_child0": lambda s, v: invalidate(s, 1, 0, v),
    "invalidate_child1": lambda s, v: invalidate(s, 1, 1, v),
    "drain_child0": lambda s, v: drain(s, 1, 0, v),
    "drain_child1": lambda s, v: drain(s, 1, 1, v),
    "finish_child": lambda s, v: finish(s, 1, v),
    "reuse0": lambda s, v: reuse(s, 0, v),
    "resolve0": lambda s, v: resolve_fault(s, 0),
    "mmap0": lambda s, v: mmap_one_page(s, 0, v),
    "retry0": lambda s, v: retry(s, 0, v),
    "bad_read": lambda s, v: read_into_object(s, 0, s.p[0], 4096, v),
    "clone": lambda s, v: clone_private(s, v),
}


@dataclass(frozen=True)
class Family:
    name: str
    initial: State
    schedules: tuple[tuple[str, ...], ...]
    variants: tuple[tuple[str, str], ...] = ()
    # Directed contracts require exact outcomes, including positive accesses.
    # Refusing every access must never satisfy a functionality contract.
    expected: tuple[tuple[str, str], ...] = ()


def empty_object(*, mapped: bool) -> State:
    return State(live=(False, True), phase=(DEAD, ACTIVE),
                 generation=(-1, 0), birth=(-1, 0), epoch=(-1, 0),
                 vma=(mapped, True), pte=(RW if mapped else ABSENT, RW),
                 p=(None, Capability(namespace=1)))


def virgin_child(*, parent_live: bool = True) -> State:
    return State(context_used=(True, False), live=(parent_live, False),
                 phase=(ACTIVE if parent_live else DEAD, DEAD),
                 generation=(0, -1), birth=(0, -1),
                 vma=(True, False), pte=(RW, ABSENT), p=(Capability(), None))


BARRIER = ("invalidate0", "invalidate1", "drain0", "drain1", "finish0")
POST_FREE = ("load_p0", "load_q0", "complete0")
POST_MMAP = ("load_p0", "load_q0", "resolve0", "retry0", "complete0")


def families() -> tuple[Family, ...]:
    ctx = (("warm", "complete0", "switch1", "access1"),)
    events = ("invalidate0", "invalidate1", "drain0", "drain1", "complete0", "finish0")
    free_orders = tuple(("issue0", "retire0", *order, *POST_FREE)
                        for order in permutations((*events, "reuse0")))
    unmap_orders = tuple(("issue0", "unmap0", *order, *POST_MMAP)
                         for order in permutations((*events, "mmap0")))
    shared = State(user=(0, 0), lifetime=(0, 0), translation=(0, 0))
    remote_free = tuple(("issue0", "issue1", "retire0", *order, *POST_FREE, "load_q1", "complete1")
                        for order in permutations((*events, "complete1", "reuse0")))
    remote_unmap = tuple(("issue0", "issue1", "unmap0", *order, *POST_MMAP, "load_q1", "complete1")
                         for order in permutations((*events, "complete1", "mmap0")))
    malloc_trace = (("reuse0", "warm", "complete0", "retire0", *BARRIER, "reuse0", *POST_FREE),)
    remap_trace = (("warm", "complete0", "unmap0", *BARRIER, "mmap0", *POST_MMAP),)
    return (
        Family("lifetime_switch", State(live=(True, False)), ctx,
               (("stale_lifetime_selector", "object_authority"),)),
        Family("translation_switch", State(), ctx,
               (("stale_translation_selector", "foreign_translation"),)),
        Family("asid_reuse", State(live=(True, False)), ctx,
               (("stale_asid_cache", "object_authority"),)),
        Family("free_completion", State(), free_orders,
               (("no_free_drain", "reuse_authorized_with_access_pending"),)),
        Family("unmap_completion", State(), unmap_orders,
               (("no_unmap_drain", "reuse_authorized_with_access_pending"),)),
        Family("two_hart_free", shared, remote_free,
               (("drain_ignores_hart1", "drain_ack_with_access_pending"),)),
        Family("two_hart_unmap", shared, remote_unmap,
               (("drain_ignores_hart1", "drain_ack_with_access_pending"),)),
        Family("unrelated_context", State(),
               (("issue0", "retire0", "issue1", "invalidate0", "invalidate1",
                 "complete1", "drain1", "finish0", "complete0", "drain0", "finish0"),),
               (("no_free_drain", "reuse_authorized_with_access_pending"),)),
        Family("malloc64_same_address", empty_object(mapped=True), malloc_trace,
               (("reuse_keeps_generation", "stale_allocation"),
                ("invalidate_keeps_cache", "stale_allocation"),
                ("rejects_fresh_generation", "functional_contract")),
               (("reuse0", "allocated"), ("load_p0", "cap_fault"),
                ("load_q0", "issued"), ("complete0", "completed"))),
        Family("mmap_same_address", State(), remap_trace,
               (("mmap_keeps_generation", "stale_allocation"),
                ("invalidate_keeps_cache", "stale_allocation")),
               (("mmap0", "mapped"), ("load_p0", "cap_fault"), ("load_q0", "page_fault"),
                ("resolve0", "resolved"), ("retry0", "issued"), ("complete0", "completed"))),
        Family("reuse_preserves_ro", State(pte=(RO, RW), vm_rights=(RO, RW)),
               (("retire0", *BARRIER, "reuse0", "load_p0", "load_q0", "complete0", "store_q0"),),
               (("reuse_grants_write", "allocator_changed_page_permissions"),),
               (("load_p0", "cap_fault"), ("load_q0", "issued"),
                ("complete0", "completed"), ("store_q0", "page_permission_fault"))),
        Family("fault_retry_retired", empty_object(mapped=False),
               (("mmap0", "issue0", "retire0", "resolve0", "retry0"),),
               (("fault_retry_skips_liveness", "object_authority"),),
               (("mmap0", "mapped"), ("issue0", "page_fault"), ("resolve0", "resolved"),
                ("retry0", "retry_denied"))),
        Family("fault_retry_live", empty_object(mapped=False),
               (("mmap0", "issue0", "resolve0", "retry0", "complete0"),),
               (("retry_always_denies", "functional_contract"),),
               (("mmap0", "mapped"), ("issue0", "page_fault"), ("resolve0", "resolved"),
                ("retry0", "issued"), ("complete0", "completed"))),
        Family("page_write", State(pte=(RO, RW), vm_rights=(RO, RW)), (("issue0",),),
               (("pte_write_bypass", "page_permission"),), (("issue0", "page_permission_fault"),)),
        Family("syscall_span", State(), (("bad_read",),),
               (("syscall_clamps_request", "syscall_consumed_on_bad_span"),), (("bad_read", "EFAULT"),)),
        Family("private_clone", virgin_child(), (("clone", "retire0", "issue1", "complete1"),),
               (("clone_shares_backing", "private_backing_shared"),),
               (("clone", "cloned"), ("issue1", "issued"), ("complete1", "completed"))),
        Family("dead_clone", virgin_child(parent_live=False), (("clone", "issue1"),),
               (("clone_revives_dead_node", "revived_dead_identity"),),
               (("clone", "cloned"), ("issue1", "cap_fault"))),
        Family("clone_preserves_stale_and_fresh", virgin_child(),
               (("retire0", *BARRIER, "reuse0", "clone", "load_p1", "store_q1", "complete1",
                 "load_p0", "load_q0", "complete0"),),
               expected=(("clone", "cloned"), ("load_p1", "cap_fault"), ("store_q1", "issued"),
                         ("complete1", "completed"), ("load_p0", "cap_fault"),
                         ("load_q0", "issued"), ("complete0", "completed"))),
        Family("clone_rejects_used_context", State(),
               (("unmap1", "invalidate_child0", "invalidate_child1", "drain_child0",
                 "drain_child1", "finish_child", "clone", "load_p1"),),
               (("clone_reuses_context", "reused_context_instance"),),
               (("clone", "refused"), ("load_p1", "cap_fault"))),
    )


def explore(family: Family, variant: str | None = None, target: str | None = None) -> dict:
    states = {family.initial}
    counts: Counter[str] = Counter()
    by_action: dict[str, Counter[str]] = {}
    violations: dict[str, dict] = {}
    successful_schedules = 0
    expected = dict(family.expected)
    for schedule in family.schedules:
        state = family.initial
        all_executed = True
        trace = []
        for action in schedule:
            before = state
            step = ACTIONS[action](state, variant)
            if step.outcome in ("refused", "cap_fault", "page_permission_fault", "EFAULT") and step.state != before:
                raise AssertionError(f"failed operation changed state: {family.name} {action}")
            counts[step.outcome] += 1
            by_action.setdefault(action, Counter())[step.outcome] += 1
            trace.append({"action": action, "outcome": step.outcome})
            all_executed &= step.outcome != "refused"
            state = step.state
            states.add(state)
            violation = step.violation
            if violation is None and action in expected and step.outcome != expected[action]:
                violation = "functional_contract"
            if violation:
                violations.setdefault(violation, {"trace": trace.copy(), "property": violation})
                all_executed = False
                break
        successful_schedules += all_executed
    return {"schedules": len(family.schedules), "states": len(states),
            "successful_schedules": successful_schedules,
            "outcomes": dict(sorted(counts.items())),
            "actions": {a: dict(sorted(c.items())) for a, c in sorted(by_action.items())},
            "counterexample": violations.get(target) or (next(iter(violations.values())) if violations else None),
            "violations": sorted(violations)}


def directed_replay(family: Family) -> State:
    state = family.initial
    for action in family.schedules[0]:
        step = ACTIONS[action](state, None)
        if step.violation:
            raise AssertionError(step.violation)
        state = step.state
    return state


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> dict:
    results = {}
    all_families = families()
    used = set()
    for family in all_families:
        correct = explore(family)
        if correct["counterexample"]:
            raise AssertionError(f"correct model failed: {family.name}: {correct['counterexample']}")
        controls = {}
        for variant, prop in family.variants:
            used.add(variant)
            injected = explore(family, variant, prop)
            witness = injected["counterexample"]
            if witness is None or witness["property"] != prop:
                raise AssertionError(f"variant not detected: {family.name}/{variant}: {witness}")
            controls[variant] = injected
        results[family.name] = {"correct": correct, "variants": controls}
    if used != VARIANTS:
        raise AssertionError(f"variant coverage mismatch: {used ^ VARIANTS}")
    for name in ("free_completion", "unmap_completion", "two_hart_free", "two_hart_unmap"):
        result = results[name]["correct"]
        if result["successful_schedules"] == 0:
            raise AssertionError(f"no complete lifecycle: {name}")
        for action, outcome in (("load_p0", "cap_fault"), ("retry0" if "unmap" in name else "load_q0", "issued"),
                                ("complete0", "completed"), ("finish0", "finished")):
            if result["actions"].get(action, {}).get(outcome, 0) == 0:
                raise AssertionError(f"vacuous lifecycle: {name}/{action}/{outcome}")
        if name.startswith("two_hart"):
            for action, outcome in (("issue1", "issued"), ("drain1", "refused"),
                                    ("complete1", "completed"), ("load_q1", "issued")):
                if result["actions"].get(action, {}).get(outcome, 0) == 0:
                    raise AssertionError(f"vacuous remote-hart check: {name}/{action}")
    malloc_family = next(f for f in all_families if f.name == "malloc64_same_address")
    final = directed_replay(malloc_family)
    p, q = final.p[0], final.q[0]
    require(p is not None and q is not None, "malloc did not publish both pointers")
    require(p.address == q.address == ADDRESS and p.end - p.base == q.end - q.base == 64,
            "malloc(64) did not reuse the exact address and size")
    require(p.generation == 0 and q.generation == 1 and p.birth != q.birth,
            "malloc reused an allocation identity")
    initial = State()
    require(issue(initial, 0, replace(initial.p[0], tagged=False), False, None) == Step(initial, "cap_fault"),
            "untagged pointer accepted")
    require(read_into_object(initial, 0, initial.p[0], 4096, None) == Step(initial, "EFAULT"),
            "bad read consumed input")
    require(read_into_object(initial, 0, initial.p[0], 16, None).state.input_bytes == 16,
            "valid read did not consume input")
    warm = complete(access(initial, 0, "p", False, None).state, 0).state
    retiring = retire(warm, 0).state
    require(access(retiring, 1, "p", True, None).outcome == "issued", "unrelated context blocked")
    require(access(retiring, 0, "p", False, None).outcome == "cap_fault",
            "cached authorization issued after retirement break")
    mapped_ro = mmap_one_page(empty_object(mapped=False), 0, None, RO).state
    fault_ro = access(mapped_ro, 0, "p", False, None)
    require(fault_ro.outcome == "page_fault", "read-only mapping skipped demand fault")
    resolved_ro = resolve_fault(fault_ro.state, 0).state
    require(resolved_ro.pte[0] == RO, "fault resolution upgraded VMA rights")
    retried_ro = retry(resolved_ro, 0, None)
    require(retried_ro.outcome == "issued" and not retried_ro.violation,
            "read-only mapping denied valid load retry")
    completed_ro = complete(retried_ro.state, 0).state
    require(access(completed_ro, 0, "p", True, None) == Step(completed_ro, "page_permission_fault"),
            "read-only demand mapping permitted a store")
    cloned = clone_private(virgin_child(), None).state
    child_write = complete(access(cloned, 1, "p", True, None).state, 1).state
    require(child_write.data == (0, 1), "private clone shared writes")
    inherited = directed_replay(next(f for f in all_families if f.name == "clone_preserves_stale_and_fresh"))
    require(inherited.data == (0, 1), "cloned replacement shared backing")
    # Exhaustion fails without wrapping identities or modifying state.
    for unmap in (False, True):
        exhausted = retire(final, 0, unmap=unmap).state
        for action in BARRIER:
            step = ACTIONS[action](exhausted, None)
            require(step.outcome != "refused" and not step.violation, "exhaustion teardown failed")
            exhausted = step.state
        result = mmap_one_page(exhausted, 0, None) if unmap else reuse(exhausted, 0, None)
        require(result == refused(exhausted), "generation exhaustion wrapped or modified state")
    return {"schema": 2, "model_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bounds": {"contexts": 2, "harts": 2, "objects_per_context": 1,
                       "pages_per_context": 1, "generations": 2, "object_bytes": OBJECT_BYTES,
                       "pending_accesses_per_hart": 1},
            "families": results,
            "malloc64_contract": {"address_p": p.address, "address_q": q.address,
                                  "generation_p": p.generation, "generation_q": q.generation,
                                  "dereference_p": "cap_fault", "dereference_q": "issued_then_completed"},
            "checks": {"untagged_pointer": "PASS", "bad_read_no_input_consumption": "PASS",
                       "valid_read_consumes_input": "PASS", "unrelated_issue_during_retirement": "PASS",
                       "cached_issue_after_retire_rejected": "PASS",
                       "read_only_fault_resolution": "PASS", "private_clone_writes_independent": "PASS",
                       "generation_exhaustion_no_wrap": "PASS"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, help="Write deterministic JSON result")
    args = parser.parse_args()
    result = run()
    for name, family in result["families"].items():
        print(f"{name}: safe {family['correct']['schedules']} schedules / "
              f"{family['correct']['states']} states; {len(family['variants'])} controls detected")
    if args.record:
        args.record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
