#!/usr/bin/env python3
"""Finite transition model for the trusted-Linux M1 execution boundary.

The state is deliberately small: two address spaces, two harts, one object
and one virtual page per space, two generations and one access per hart. See
README.md for the abstraction boundary and the commands used to run it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from itertools import permutations
import json
from pathlib import Path
import hashlib


ABSENT, RO, RW = 0, 1, 2
ACTIVE, RETIRING, DEAD = 0, 1, 2
VARIANTS = (
    "stale_lifetime_selector",
    "stale_translation_selector",
    "stale_asid_cache",
    "no_free_drain",
    "no_unmap_drain",
    "fault_retry_skips_liveness",
    "pte_write_bypass",
    "syscall_clamps_request",
    "clone_shares_backing",
    "clone_revives_dead_node",
)


def updated(items: tuple, index: int, value: object) -> tuple:
    result = list(items)
    result[index] = value
    return tuple(result)


@dataclass(frozen=True)
class Pending:
    context: int
    frame: int
    epoch: int
    write: bool


@dataclass(frozen=True)
class Fault:
    context: int
    generation: int
    write: bool


@dataclass(frozen=True)
class State:
    # user, lifetime and translation selectors are separate so a missing
    # context-switch update is observable.
    user: tuple[int, int] = (0, 1)
    lifetime: tuple[int, int] = (0, 1)
    translation: tuple[int, int] = (0, 1)
    asid: tuple[int, int] = (0, 0)  # intentionally reused
    cache: tuple[tuple[int, int] | None, ...] = (None, None)
    live: tuple[bool, bool] = (True, True)
    generation: tuple[int, int] = (0, 0)
    phase: tuple[int, int] = (ACTIVE, ACTIVE)
    vma: tuple[bool, bool] = (True, True)
    pte: tuple[int, int] = (RW, RW)
    frame: tuple[int, int] = (0, 1)
    epoch: tuple[int, int] = (0, 0)
    data: tuple[int, int] = (0, 0)
    invalidated: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
    drained: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
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


def switch(state: State, hart: int, context: int, variant: str | None) -> Step:
    if state.pending[hart] is not None or hart not in (0, 1) or context not in (0, 1):
        return refused(state)
    lifetime = state.lifetime if variant == "stale_lifetime_selector" else updated(state.lifetime, hart, context)
    translation = state.translation if variant == "stale_translation_selector" else updated(state.translation, hart, context)
    cache = state.cache if variant == "stale_asid_cache" else updated(state.cache, hart, None)
    return Step(replace(state, user=updated(state.user, hart, context),
                        lifetime=lifetime, translation=translation, cache=cache), "switch")


def issue(state: State, hart: int, generation: int, write: bool,
          variant: str | None, *, retry_prechecked: bool = False,
          tagged: bool = True) -> Step:
    if state.pending[hart] is not None or state.fault[hart] is not None:
        return refused(state)
    context = state.user[hart]
    if state.phase[context] != ACTIVE and not (
        retry_prechecked and variant == "fault_retry_skips_liveness"
    ):
        return refused(state)
    lifetime = state.lifetime[hart]
    translation = state.translation[hart]
    # A cache keyed only by a reusable ASID is safe here only because switch
    # flushes it. Generation and tag checks remain mandatory.
    cached = state.cache[hart] == (state.asid[context], generation)
    authorized = tagged and (
        retry_prechecked and variant == "fault_retry_skips_liveness"
        or cached
        or state.vma[lifetime] and state.live[lifetime] and state.generation[lifetime] == generation
    )
    if not authorized:
        return refused(state)
    permission = state.pte[translation]
    if permission == ABSENT:
        return Step(replace(state, fault=updated(state.fault, hart,
                                                 Fault(context, generation, write))), "page_fault")
    if write and permission != RW and variant != "pte_write_bypass":
        return refused(state)
    frame = state.frame[translation]
    pending = Pending(context, frame, state.epoch[frame], write)
    next_state = replace(state, pending=updated(state.pending, hart, pending),
                         cache=updated(state.cache, hart, (state.asid[context], generation)))
    if not state.vma[context] or not state.live[context] or state.generation[context] != generation:
        return Step(next_state, "issued", "object_authority")
    if state.pte[context] == ABSENT or write and state.pte[context] != RW:
        return Step(next_state, "issued", "page_permission")
    if frame != state.frame[context]:
        return Step(next_state, "issued", "foreign_translation")
    return Step(next_state, "issued")


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
    # Break new authority first. Invalidation and drain are independent
    # per-hart transitions; a cached approval cannot issue during retirement.
    # An already issued access still retains its physical destination.
    pte = updated(state.pte, context, ABSENT) if unmap else state.pte
    vma = updated(state.vma, context, False) if unmap else state.vma
    return Step(replace(state, live=updated(state.live, context, False),
                        phase=updated(state.phase, context, RETIRING),
                        pte=pte, vma=vma,
                        invalidated=updated(state.invalidated, context, (False, False)),
                        drained=updated(state.drained, context, (False, False))), "retiring")


def invalidate(state: State, context: int, hart: int) -> Step:
    if state.phase[context] != RETIRING or state.invalidated[context][hart]:
        return refused(state)
    return Step(replace(state, cache=updated(state.cache, hart, None),
                        invalidated=updated(state.invalidated, context,
                                            updated(state.invalidated[context], hart, True))), "invalidated")


def drain(state: State, context: int, hart: int) -> Step:
    if state.phase[context] != RETIRING or not state.invalidated[context][hart] or state.drained[context][hart]:
        return refused(state)
    access = state.pending[hart]
    if access is not None and access.context == context:
        return refused(state)
    return Step(replace(state, drained=updated(state.drained, context,
                                              updated(state.drained[context], hart, True))), "drained")


def finish(state: State, context: int, variant: str | None, *, unmap: bool = False) -> Step:
    if state.phase[context] != RETIRING:
        return refused(state)
    if state.invalidated[context] != (True, True):
        return refused(state)
    in_flight = any(p is not None and p.context == context for p in state.pending)
    skip = variant == ("no_unmap_drain" if unmap else "no_free_drain")
    if state.drained[context] != (True, True) and not skip:
        return refused(state)
    next_state = replace(state, phase=updated(state.phase, context, DEAD))
    if in_flight:
        return Step(next_state, "finished", "reuse_authorized_with_access_pending")
    if state.drained[context] != (True, True):
        return Step(next_state, "finished", "finish_without_drain_ack")
    return Step(next_state, "finished")


def reuse(state: State, context: int) -> Step:
    if state.phase[context] != DEAD or state.generation[context] >= 1:
        return refused(state)
    frame = state.frame[context]
    return Step(replace(state, phase=updated(state.phase, context, ACTIVE),
                        live=updated(state.live, context, True),
                        vma=updated(state.vma, context, True),
                        generation=updated(state.generation, context, state.generation[context] + 1),
                        epoch=updated(state.epoch, frame, state.epoch[frame] + 1),
                        pte=updated(state.pte, context, RW)), "reused")


def resolve_fault(state: State, hart: int) -> Step:
    fault = state.fault[hart]
    if fault is None or not state.vma[fault.context] or state.pte[fault.context] != ABSENT:
        return refused(state)
    # Linux resolves the VMA fault independently of the inner object's life.
    return Step(replace(state, pte=updated(state.pte, fault.context, RW)), "resolved")


def mmap_one_page(state: State, context: int) -> Step:
    if state.vma[context] or state.phase[context] != DEAD or state.generation[context] != 0:
        return refused(state)
    # The trusted ABI returns tagged arena authority; the sole modeled object
    # is created inside that arena. Backing remains absent until a page fault.
    return Step(replace(state, vma=updated(state.vma, context, True),
                        live=updated(state.live, context, True),
                        phase=updated(state.phase, context, ACTIVE)), "mapped")


def clone_private(state: State, variant: str | None) -> Step:
    """Privileged fork copy into unused context 1; node bits remain equal."""
    if state.phase[1] != DEAD or state.vma[1] or state.pending[0] or state.pending[1]:
        return refused(state)
    live = state.live[0]
    phase = state.phase[0]
    if variant == "clone_revives_dead_node" and not live:
        live, phase = True, ACTIVE
    frame = state.frame[0] if variant == "clone_shares_backing" else state.frame[1]
    data = state.data if frame == state.frame[0] else updated(state.data, frame, state.data[state.frame[0]])
    next_state = replace(state, live=updated(state.live, 1, live),
                         phase=updated(state.phase, 1, phase),
                         generation=updated(state.generation, 1, state.generation[0]),
                         vma=updated(state.vma, 1, state.vma[0]),
                         pte=updated(state.pte, 1, state.pte[0]),
                         frame=updated(state.frame, 1, frame), data=data)
    if frame == state.frame[0]:
        return Step(next_state, "cloned", "private_backing_shared")
    if live and not state.live[0]:
        return Step(next_state, "cloned", "revived_dead_identity")
    return Step(next_state, "cloned")


def retry(state: State, hart: int, variant: str | None) -> Step:
    fault = state.fault[hart]
    if fault is None or state.user[hart] != fault.context:
        return refused(state)
    cleared = replace(state, fault=updated(state.fault, hart, None))
    step = issue(cleared, hart, fault.generation, fault.write, variant,
                 retry_prechecked=True)
    # Consuming the saved fault is a real architectural transition, even if
    # the retried instruction is denied because the object died meanwhile.
    if step.outcome == "refused":
        return Step(cleared, "retry_denied")
    return step


def read_into_object(state: State, context: int, generation: int,
                     requested: int, bound: int, variant: str | None) -> Step:
    if requested < 0 or not state.live[context] or state.generation[context] != generation:
        return refused(state)
    if requested > bound and variant != "syscall_clamps_request":
        return Step(state, "EFAULT")
    copied = min(requested, bound, state.input_bytes)
    next_state = replace(state, input_bytes=state.input_bytes - copied)
    if requested > bound:
        return Step(next_state, f"read:{copied}", "syscall_consumed_on_bad_span")
    return Step(next_state, f"read:{copied}")


def apply(state: State, action: str, variant: str | None) -> Step:
    if action == "warm":
        return issue(state, 0, 0, False, variant)
    if action == "switch1":
        return switch(state, 0, 1, variant)
    if action == "access1":
        return issue(state, 0, 0, False, variant)
    if action == "issue0":
        return issue(state, 0, 0, True, variant)
    if action == "issue1":
        return issue(state, 1, 0, True, variant)
    if action == "complete0":
        return complete(state, 0)
    if action == "complete1":
        return complete(state, 1)
    if action == "retire0":
        return retire(state, 0)
    if action == "unmap0":
        return retire(state, 0, unmap=True)
    if action == "finish0":
        return finish(state, 0, variant)
    if action == "finish_unmap0":
        return finish(state, 0, variant, unmap=True)
    if action == "reuse0":
        return reuse(state, 0)
    if action == "invalidate0":
        return invalidate(state, 0, 0)
    if action == "invalidate1":
        return invalidate(state, 0, 1)
    if action == "drain0":
        return drain(state, 0, 0)
    if action == "drain1":
        return drain(state, 0, 1)
    if action == "resolve0":
        return resolve_fault(state, 0)
    if action == "mmap0":
        return mmap_one_page(state, 0)
    if action == "retry0":
        return retry(state, 0, variant)
    if action == "bad_read":
        return read_into_object(state, 0, 0, 4096, 16, variant)
    if action == "clone":
        return clone_private(state, variant)
    raise ValueError(action)


@dataclass(frozen=True)
class Family:
    name: str
    initial: State
    schedules: tuple[tuple[str, ...], ...]
    variant: str
    expected_violation: str


def families() -> tuple[Family, ...]:
    context = State(live=(True, False))
    both_live = State()
    # A successful c0 access warms the cache; completion permits a switch.
    context_schedule = (("warm", "complete0", "switch1", "access1"),)
    barrier_events = ("invalidate0", "invalidate1", "drain0", "drain1", "complete0")
    free_schedules = tuple(("issue0", "retire0", *order) for order in
                           permutations((*barrier_events, "finish0", "reuse0")))
    unmap_schedules = tuple(("issue0", "unmap0", *order) for order in
                            permutations((*barrier_events, "finish_unmap0", "reuse0")))
    foreign_schedule = (("issue0", "retire0", "issue1", "invalidate0", "invalidate1",
                         "complete1", "drain1", "finish0", "complete0", "drain0", "finish0"),)
    return (
        Family("lifetime_switch", context, context_schedule,
               "stale_lifetime_selector", "object_authority"),
        Family("translation_switch", both_live, context_schedule,
               "stale_translation_selector", "foreign_translation"),
        Family("asid_reuse", context, context_schedule,
               "stale_asid_cache", "object_authority"),
        Family("free_completion", State(), free_schedules,
               "no_free_drain", "reuse_authorized_with_access_pending"),
        Family("unmap_completion", State(), unmap_schedules,
               "no_unmap_drain", "reuse_authorized_with_access_pending"),
        Family("unrelated_context", State(), foreign_schedule,
               "no_free_drain", "reuse_authorized_with_access_pending"),
        Family("fault_retry", State(live=(False, True), phase=(DEAD, ACTIVE),
                                     vma=(False, True), pte=(ABSENT, RW)),
               (("mmap0", "issue0", "retire0", "resolve0", "retry0"),),
               "fault_retry_skips_liveness", "object_authority"),
        Family("page_write", State(pte=(RO, RW)), (("issue0",),),
               "pte_write_bypass", "page_permission"),
        Family("syscall_span", State(), (("bad_read",),),
               "syscall_clamps_request", "syscall_consumed_on_bad_span"),
        Family("private_clone", State(live=(True, False), phase=(ACTIVE, DEAD),
                                      vma=(True, False), pte=(RW, ABSENT)),
               (("clone", "retire0", "issue1", "complete1"),),
               "clone_shares_backing", "private_backing_shared"),
        Family("dead_clone", State(live=(False, False), phase=(DEAD, DEAD),
                                   vma=(True, False), pte=(RW, ABSENT)),
               (("clone", "issue1"),),
               "clone_revives_dead_node", "revived_dead_identity"),
    )


def explore(family: Family, variant: str | None) -> dict:
    states: set[State] = {family.initial}
    counts: Counter[str] = Counter()
    violations: dict[str, dict] = {}
    success_schedules = 0
    for schedule in family.schedules:
        state = family.initial
        successful = set()
        for action in schedule:
            before = state
            step = apply(state, action, variant)
            if step.outcome == "refused" and step.state != before:
                raise AssertionError(f"refusal changed state: {family.name} {action}")
            counts[step.outcome] += 1
            if step.outcome != "refused":
                successful.add(action)
            state = step.state
            states.add(state)
            if step.violation:
                violations.setdefault(step.violation, {"schedule": schedule,
                                                        "action": action,
                                                        "property": step.violation})
                break
        if len(successful) == len(schedule):
            success_schedules += 1
    return {"schedules": len(family.schedules), "states": len(states),
            "successful_schedules": success_schedules,
            "outcomes": dict(sorted(counts.items())),
            "counterexample": violations.get(family.expected_violation) or
            (next(iter(violations.values())) if violations else None),
            "violations": sorted(violations)}


def run() -> dict:
    results = {}
    used_variants = set()
    for family in families():
        used_variants.add(family.variant)
        correct = explore(family, None)
        injected = explore(family, family.variant)
        if correct["counterexample"] is not None:
            raise AssertionError(f"correct model failed: {family.name}: {correct['counterexample']}")
        witness = injected["counterexample"]
        if witness is None or witness["property"] != family.expected_violation:
            raise AssertionError(f"variant not detected: {family.name}: {witness}")
        results[family.name] = {"correct": correct, "variant": family.variant,
                                "injected": injected}
    if used_variants != set(VARIANTS):
        raise AssertionError(f"variant coverage mismatch: {used_variants ^ set(VARIANTS)}")
    for name in ("free_completion", "unmap_completion"):
        result = results[name]["correct"]
        if result["successful_schedules"] == 0 or not {
            "issued", "retiring", "invalidated", "drained", "completed",
            "finished", "reused", "refused"
        } <= result["outcomes"].keys():
            raise AssertionError(f"vacuous lifecycle search: {name}")
    fault_outcomes = results["fault_retry"]["correct"]["outcomes"]
    if not {"mapped", "page_fault", "retiring", "resolved", "retry_denied"} <= fault_outcomes.keys():
        raise AssertionError("fault retry did not test retirement")
    if results["private_clone"]["correct"]["successful_schedules"] != 1:
        raise AssertionError("private clone did not reach child access")
    if results["dead_clone"]["correct"]["outcomes"].get("refused") != 1:
        raise AssertionError("inherited dead pointer was not rejected")
    # A forged tag and a requested span wider than the object must be refused
    # without consuming input or issuing an access.
    initial = State()
    if issue(initial, 0, 0, False, None, tagged=False) != refused(initial):
        raise AssertionError("untagged pointer accepted")
    if read_into_object(initial, 0, 0, 4096, 16, None) != Step(initial, "EFAULT"):
        raise AssertionError("bad read consumed input")
    # An unrelated address space must be able to issue during c0 retirement.
    state = retire(initial, 0).state
    if issue(state, 1, 0, True, None).outcome != "issued":
        raise AssertionError("unrelated context blocked")
    warm = complete(issue(initial, 0, 0, False, None).state, 0).state
    retired = retire(warm, 0).state
    if issue(retired, 0, 0, False, None) != refused(retired):
        raise AssertionError("cached authorization issued after retirement break")
    cloned = clone_private(State(live=(True, False), phase=(ACTIVE, DEAD),
                                 vma=(True, False), pte=(RW, ABSENT)), None).state
    child_write = complete(issue(cloned, 1, 0, True, None).state, 1).state
    if child_write.data != (0, 1):
        raise AssertionError("private clone shared writes")
    source = Path(__file__)
    return {"schema": 1, "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bounds": {"contexts": 2, "harts": 2, "objects_per_context": 1,
                       "pages_per_context": 1, "generations": 2,
                       "pending_accesses_per_hart": 1},
            "families": results,
            "checks": {"untagged_pointer": "PASS", "bad_read_no_input_consumption": "PASS",
                       "unrelated_issue_during_retirement": "PASS",
                       "cached_issue_after_retire_rejected": "PASS",
                       "private_clone_writes_independent": "PASS"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, help="Write deterministic JSON result")
    args = parser.parse_args()
    result = run()
    for name, family in result["families"].items():
        print(f"{name}: safe {family['correct']['schedules']} schedules / "
              f"{family['correct']['states']} states; "
              f"{family['variant']} -> {family['injected']['counterexample']['property']}")
    if args.record:
        args.record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
