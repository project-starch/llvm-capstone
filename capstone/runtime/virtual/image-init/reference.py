#!/usr/bin/env python3
"""Draft image-initialization interface double; no executable loader is installed."""
from dataclasses import dataclass
from contextlib import nullcontext
from threading import RLock
from types import MappingProxyType

VERSION = 1
PAGE = 4096
CAP_BYTES = 16
U64 = 1 << 64
USER_END = 1 << 38


class Refused(ValueError):
    pass


@dataclass(frozen=True)
class Grant:
    namespace: int
    identity: int
    generation: int
    base: int
    size: int
    rights: str


@dataclass(frozen=True)
class Symbol:
    grant: Grant | None
    offset: int = 0
    size: int = 0
    weak: bool = False


@dataclass(frozen=True)
class Relocation:
    destination: int
    symbol: int
    addend: int = 0


@dataclass(frozen=True)
class Capability:
    grant: Grant
    base: int
    size: int
    cursor: int


def require(ok, reason):
    if not ok:
        raise Refused(reason)


def span(base, size):
    return (type(base) is int and type(size) is int and
            0 <= base < U64 and 0 < size <= U64 - base)


def user_span(base, size):
    return span(base, size) and base + size <= USER_END


def representable(base, size):
    granule = 1 if size < PAGE else 1 << (size.bit_length() - 10)
    return base % granule == 0 and size % granule == 0


def in_window(base, size, cursor):
    # Exact ranges only. The exponent and modular window are R-60's rule.
    exponent = max(0, size.bit_length() - 13)
    width = 1 << (exponent + 11)
    low = ((base & -width) - width) % U64
    return (cursor - low) % U64 < 1 << (exponent + 14)


class Image:
    """Protected plan plus per-page publication state.

    `admit(grant, address, size, access)` is a protected grant callback, never
    a task-supplied boolean. `guard(namespace)` holds lifetime exclusion across
    validation and publication, including every potentially sleeping admission.
    The caller supplies event classification from its context/fault backend.
    """
    def __init__(self, version, destination, symbols, records, *, metadata_writable,
                 admit, guard, mutant=None):
        self._mutant = mutant
        require(mutant in (None, 'writable-table', 'weak-rebase', 'cursor-window',
                           'destination-span', 'unlocked-publication',
                           'address-policy'), 'unsupported mutant')
        require(type(version) is int and version == VERSION, 'unsupported version')
        require(not metadata_writable or mutant == 'writable-table', 'writable metadata')
        require(span(destination.base, destination.size) and
                destination.base % PAGE == 0 and destination.size % PAGE == 0 and
                'w' in destination.rights, 'destination grant')
        require(mutant == 'address-policy' or
                user_span(destination.base, destination.size), 'unsupported destination address policy')
        self._destination = destination
        self._symbols = tuple(symbols)
        self._records = tuple(records)
        self._admit = admit
        self._guard = guard
        self._lock = RLock()
        self._pages = {}
        self._memory = {}
        previous = -1
        for r in self._records:
            require(type(r) is Relocation and type(r.destination) is int and
                    type(r.symbol) is int and type(r.addend) is int, 'record type')
            require(r.destination > previous and r.destination % CAP_BYTES == 0,
                    'unsorted, duplicate or unaligned destination')
            require(mutant == 'destination-span' or
                    (r.destination >= 0 and r.destination + CAP_BYTES <= destination.size),
                    'destination outside grant')
            require(0 <= r.symbol < len(self._symbols), 'symbol index')
            require(-(1 << 63) <= r.addend < (1 << 63), 'addend width')
            self._resolve(r)
            self._pages.setdefault(r.destination // PAGE, []).append(r)
            previous = r.destination
        self._done = set()

    def _resolve(self, record):
        s = self._symbols[record.symbol]
        require(type(s) is Symbol, 'symbol type')
        if s.grant is None:
            require(s.weak, 'undefined strong symbol')
            # An undefined weak symbol is NULL, including a nonzero addend.
            return 1 if self._mutant == 'weak-rebase' else None
        g = s.grant
        require(g.namespace == self._destination.namespace and span(g.base, g.size),
                'source namespace or span')
        require(self._mutant == 'address-policy' or user_span(g.base, g.size),
                'unsupported source address policy')
        require(type(s.offset) is int and type(s.size) is int and
                s.offset >= 0 and s.size > 0 and s.offset + s.size <= g.size,
                'symbol outside grant or unsupported unsized symbol')
        base = g.base + s.offset
        require(representable(base, s.size), 'inexact bounds')
        cursor = base + record.addend
        require(0 <= cursor < U64, 'cursor overflow')
        require(self._mutant == 'cursor-window' or in_window(base, s.size, cursor),
                'cursor outside representable window')
        # This first draft supports only low Sv39 userspace addresses.
        require(cursor < USER_END, 'unsupported address policy')
        return Capability(g, base, s.size, cursor)

    def touch(self, namespace, page, *, event, cause, classify):
        """Publish once, after all source/destination checks; retry is idempotent."""
        guard = nullcontext() if self._mutant == 'unlocked-publication' else self._guard(namespace)
        with self._lock, guard:
            require(namespace == self._destination.namespace, 'foreign namespace')
            require(classify(event, cause) in ('read', 'write'), 'not a data page fault')
            require(type(page) is int and 0 <= page < self._destination.size // PAGE,
                    'page outside image')
            address = self._destination.base + page * PAGE
            require(self._admit(self._destination, address, PAGE, 'w'), 'destination retired')
            if page in self._done:
                return 0
            pending = {}
            for r in self._pages.get(page, ()):
                value = self._resolve(r)
                if isinstance(value, Capability):
                    require(self._admit(value.grant, value.base, value.size, 'r'), 'source retired')
                pending[r.destination] = value
            # No fallible operation follows publication in this model.
            self._memory.update(pending)
            self._done.add(page)
            return len(pending)

    @property
    def memory(self):
        return MappingProxyType(self._memory)
