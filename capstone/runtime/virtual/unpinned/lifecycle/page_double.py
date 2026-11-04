#!/usr/bin/env python3
"""Executable physical-page lifecycle v1 double; no architectural authority."""
import json
import sys
from dataclasses import dataclass, field

PAGE = 4096
GRANULE = 16


@dataclass
class Page:
    epoch: int = 0
    owned: bool = False
    dma: bool = False
    data: bytearray = field(default_factory=lambda: bytearray(PAGE))
    tags: dict = field(default_factory=dict)
    sticky: bool = False


class Refused(Exception):
    pass


class Lifecycle:
    """Single-hart, exclusive allocator ownership; handles are (frame, epoch).

    Tag values are opaque identity/exact-bounds records, including stale ones.
    This layer transports them and does not decide whether a capability is live.
    """
    def __init__(self, *, skip_clear=False, scalar_copy=False):
        self.pages = {}
        self.skip_clear = skip_clear
        self.scalar_copy = scalar_copy

    def page(self, handle):
        if not isinstance(handle, (tuple, list)) or len(handle) != 2:
            raise Refused('invalid handle')
        p = self.pages.get(handle[0])
        if p is None or not p.owned or p.epoch != handle[1]:
            raise Refused('expired handle')
        return p

    @staticmethod
    def exclusive(p):
        if p.dma:
            raise Refused('writable DMA mapping')

    def allocate(self, frame):
        if type(frame) is not int or frame < 0 or frame % PAGE:
            raise Refused('invalid physical frame')
        p = self.pages.setdefault(frame, Page())
        if p.owned:
            raise Refused('frame already owned')
        p.owned = True
        return [frame, p.epoch]

    def query(self, handle, ack=False):
        p = self.page(handle)
        if ack:
            self.exclusive(p)
            if p.tags:
                return 1
            p.sticky = False
        return int(p.sticky)

    def clear(self, handle):
        p = self.page(handle)
        self.exclusive(p)
        p.data[:] = bytes(PAGE)
        if not self.skip_clear:
            p.tags.clear()
        # Zeroing is a writer, not an implicit summary acknowledgement.

    def release(self, handle):
        p = self.page(handle)
        self.exclusive(p)
        if self.query(handle):
            self.clear(handle)
            if self.query(handle, ack=True):
                raise Refused('tag survived clear')
        p.owned = False
        p.epoch += 1

    def copy(self, to, source):
        dst, src = self.page(to), self.page(source)
        self.exclusive(dst)
        self.exclusive(src)
        if to == source:
            raise Refused('overlapping pages')
        dst.data[:] = src.data
        dst.tags = {} if self.scalar_copy else dict(src.tags)
        if dst.tags:
            dst.sticky = True

    def write(self, handle, offset, payload, tag=None):
        p = self.page(handle)
        if type(offset) is not int or offset < 0 or offset + len(payload) > PAGE:
            raise Refused('out of bounds')
        if tag is not None:
            self.exclusive(p)
            if offset % GRANULE or len(payload) != GRANULE:
                raise Refused('unaligned capability')
        if not payload:
            return
        first, last = offset // GRANULE, (offset + len(payload) - 1) // GRANULE
        for i in range(first, last + 1):
            p.tags.pop(i, None)
        p.data[offset:offset + len(payload)] = payload
        if tag is not None:
            # JSON round trip takes a snapshot independent of the caller.
            p.tags[first] = json.loads(json.dumps(tag))
            p.sticky = True

    def dma(self, handle, begin):
        p = self.page(handle)
        if p.dma == begin:
            raise Refused('unbalanced DMA ownership')
        p.tags.clear()
        p.dma = begin

    def dispatch(self, request):
        if request.get('version') != 1:
            raise Refused('unsupported version')
        op = request.get('op')
        if op == 'allocate':
            return self.allocate(request['frame'])
        if op == 'query':
            return self.query(request['page'], request.get('ack', False))
        if op == 'clear':
            return self.clear(request['page'])
        if op == 'release':
            return self.release(request['page'])
        if op == 'copy':
            return self.copy(request['to'], request['from'])
        if op == 'write':
            return self.write(request['page'], request['offset'],
                              bytes.fromhex(request['hex']), request.get('tag'))
        if op == 'dma':
            return self.dma(request['page'], request['begin'])
        raise Refused('unsupported operation')


def main():
    model = Lifecycle()
    for line in sys.stdin:
        try:
            result = model.dispatch(json.loads(line))
            reply = {'version': 1, 'ok': True, 'result': result}
        except (Refused, ValueError, KeyError, TypeError) as e:
            reply = {'version': 1, 'ok': False, 'cause': str(e)}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    main()
