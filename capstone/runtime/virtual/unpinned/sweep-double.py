#!/usr/bin/env python3
"""Executable contract double for a stopped namespace's physical sweep.

This models ownership and lifetime ordering, not Linux page-table mechanics.
Run without arguments for positive and deliberately broken controls. A selected
--mutant must exit nonzero: the same assertions apply to every implementation.
"""
import argparse
from dataclasses import dataclass


@dataclass
class Page:
    tag: bool = False
    present: bool = True
    private: bool = True
    refs: int = 0
    generation: int = 0


class Sweep:
    def __init__(self, mutant=None):
        self.mutant = mutant

    def collect(self, pages, interpose=lambda: None):
        held = []
        complete = False
        try:
            for page in pages:
                if not page.present or (page.tag and not page.private):
                    if self.mutant != 'partial-success':
                        return False
                    break
                if not page.tag or self.mutant == 'omit-tagged':
                    continue
                if self.mutant != 'omit-reference':
                    page.refs += 1
                held.append((page, page.generation))
            interpose()
            for page, generation in held:
                assert page.generation == generation, 'physical page reused before sweep'
                page.tag = False
            complete = True
            return complete
        finally:
            for page, _ in held:
                if self.mutant != 'omit-reference':
                    page.refs -= 1


def check(mutant=None):
    sweep = Sweep(mutant)
    # Unregistered/retired mappings remain part of the mm until Linux unmaps them.
    old = Page(tag=True)
    assert sweep.collect([Page(), old]) and not old.tag, 'old namespace tag survived'
    assert old.refs == 0, 'successful sweep leaked a reference'
    for unsupported in (Page(present=False), Page(tag=True, private=False)):
        first = Page(tag=True)
        assert not sweep.collect([first, unsupported]), 'incomplete sweep recycled identities'
        assert first.tag and first.refs == 0, 'failed sweep changed authority or leaked a reference'
    resident = Page(tag=True)

    def migrate():
        if not resident.refs:
            resident.generation += 1

    assert sweep.collect([resident], migrate)
    assert resident.refs == 0
    # A different process retains its own lifetime state throughout collection.
    foreign = Page(tag=True)
    assert sweep.collect([Page(tag=True)]) and foreign.tag


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mutants = ('partial-success', 'omit-tagged', 'omit-reference')
    p.add_argument('--mutant', choices=mutants)
    a = p.parse_args()
    if a.mutant:
        check(a.mutant)
        return 0
    check()
    print('PASS complete sweep, rollback, references, namespace separation')
    for mutant in mutants:
        try:
            check(mutant)
        except AssertionError as error:
            print(f'PASS control {mutant}: {error}')
        else:
            raise AssertionError(f'control {mutant} escaped')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
