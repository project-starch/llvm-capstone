"""Finite, executable Stage-1 contract; no emulator or hardware implementation.

Addresses count words, pages contain four words, and tables have two levels
with two entries per level. Authority moves between explicit locations. Node
identities are never reused. See README.md for abstractions and coverage limits.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace


WORDS = 4
FANOUT = 2
PAGES = FANOUT * FANOUT
# A finite stand-in for the architectural physical/logical address partition.
PHYSICAL_BASE = WORDS
PHYSICAL_LIMIT = PHYSICAL_BASE + 16 * WORDS
LOGICAL_BASE = 128
ADDRESS_LIMIT = 256

# The architectural partition decided in
# docs/design/caplified-mapping-encoding-decision.md. The finite constants above
# stand in for it inside the walked geometry; partition_ok() is the one range
# rule, applied with either set of constants.
ARCH_PHYSICAL_BITS = 56            # CVA6 PLEN and QEMU TARGET_PHYS_ADDR_SPACE_BITS
ARCH_PHYSICAL_LIMIT = 1 << ARCH_PHYSICAL_BITS
ARCH_LOGICAL_BASE = 1 << 57        # [2^56, 2^57) is a guard octant, never valid
ARCH_LOGICAL_LIMIT = 1 << 63       # one-past cursors stay below 2^63, no wrap
ARCH_PAGE = 4096
BINDING_ID_BITS = 12               # registry entries system-wide
BINDING_GEN_BITS = 20              # generations per id before the id retires


def partition_ok(lo, hi, base=LOGICAL_BASE, limit=ADDRESS_LIMIT, page=WORDS,
                 window=None):
    """CREATE's range rule, independent of table geometry.

    [lo, hi) must lie inside [base, limit) with hi strictly below limit, so the
    one-past cursor hi is representable without wrap; both ends page-aligned;
    with a window, the range fits one aligned window (finite model only).
    """
    if not (base <= lo < hi < limit):
        return False
    if lo % page or hi % page:
        return False
    return window is None or (lo % window == 0 and hi - lo <= window)


def arch_partition_ok(lo, hi):
    return partition_ok(lo, hi, ARCH_LOGICAL_BASE, ARCH_LOGICAL_LIMIT, ARCH_PAGE)


def arch_kind(address):
    """Kind by region: the bounds of a capability decide, not an encoding bit."""
    if 0 <= address < ARCH_PHYSICAL_LIMIT:
        return "physical"
    if ARCH_LOGICAL_BASE <= address < ARCH_LOGICAL_LIMIT:
        return "logical"
    return None


def exactly_representable(lo, hi):
    """CHERI-128 grain rule of cap_compress.c: below 4 KiB exact, above it
    base and top must be multiples of 2^(E+3) with E = highest bit of the
    length minus 12. Spec-derived; the current QEMU keeps fat bounds exact."""
    length = hi - lo
    if length < 4096:
        return True
    grain = 1 << (length.bit_length() - 1 - 12 + 3)
    return lo % grain == 0 and hi % grain == 0


def pack_binding(ident, generation):
    """The 32-bit binding word stored in a revocation node; zero means none."""
    require(0 <= ident < (1 << BINDING_ID_BITS), "id out of range")
    require(1 <= generation < (1 << BINDING_GEN_BITS), "generation exhausted")
    return ident | (generation << BINDING_ID_BITS)


def unpack_binding(word):
    require(word != 0, "no binding")
    return word & ((1 << BINDING_ID_BITS) - 1), word >> BINDING_ID_BITS
HARTS = (0, 1)
LOCKED = "locked"
MUTANTS = (
    "plain_detach", "uncleared_create", "uncleared_populate",
    "walk_only_drain", "weak_binding", "duplicate_return",
    "root_release", "stale_record", "monitor_delivery", "unzeroed_frame",
)


class Refused(Exception):
    """An instruction precondition failed, or a microstep must wait."""


class Violation(Exception):
    """An independently checked safety property failed."""


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def invariant(condition, reason):
    if not condition:
        raise Violation(reason)


@dataclass(frozen=True)
class Word:
    value: int = 0
    private: bool = False


@dataclass(frozen=True)
class Cap:
    node: int
    kind: str
    binding: tuple | None = None
    page: int | None = None
    lo: int = 0
    hi: int = WORDS
    rights: str = "rw"
    linear: bool = True
    cursor: int = 0
    domain: int | None = None


@dataclass(frozen=True)
class ResumeSlot:
    """Protected destination, not an arbitrary software capability address."""
    domain: int
    name: str

    @property
    def location(self):
        return "d%d:%s" % (self.domain, self.name)


@dataclass
class Node:
    parent: int | None
    binding: tuple | None
    linear: bool = True
    valid: bool = True


@dataclass
class Mapping:
    binding: tuple
    root: Cap
    pointer_root: int
    lo: int
    hi: int
    rights: str
    state: str = "ACTIVE"


@dataclass(frozen=True)
class Translation:
    binding: tuple
    frame: Cap
    dependencies: tuple
    issued: int


@dataclass
class Access:
    hart: int
    cap: Cap
    address: int
    operation: str
    value: Word
    destination: str
    phase: str = "root"
    path: list = field(default_factory=list)
    table: Cap | None = None
    translation: Translation | None = None
    source: str | None = None
    carried: Cap | None = None
    result: Word | Cap | None = None
    reason: str = ""


@dataclass
class Preparation:
    operation: str
    arguments: tuple
    held: dict


@dataclass
class Barrier:
    operation: str
    output: str
    held: Cap
    binding: tuple | None
    affected: set
    return_kind: str
    scope: frozenset | None = None
    invalidated: set = field(default_factory=set)
    drained: set = field(default_factory=set)


def freeze(value):
    """Canonical state key, independent of dictionary insertion order."""
    if is_dataclass(value):
        return (type(value).__name__, tuple(freeze(getattr(value, f.name))
                                           for f in fields(value)))
    if isinstance(value, dict):
        return tuple(sorted(((freeze(k), freeze(v)) for k, v in value.items()),
                            key=repr))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze(v) for v in value), key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    return value


class Machine:
    def __init__(self, mutant=None, ids=2, generations=3, barrier_mode="global"):
        require(mutant is None or mutant in MUTANTS, "unknown mutant")
        require(barrier_mode in ("global", "table_record"), "unknown barrier mode")
        self.mutant = mutant
        self.barrier_mode = barrier_mode
        self.ids = ids
        self.generations = generations
        self.nodes = {}
        self.next_node = 0
        self.wallet = {}
        self.memory = {}
        self.registry = {}
        self.last_generation = {i: 0 for i in range(ids)}
        self.preparation = None
        self.barrier = None
        self.tlb = {}
        self.accesses = {}
        self.next_access = 0
        # Protected implementation state for the optional scoped experiment.
        # Unlike table_owners below, instructions may consult this record.
        self.table_bindings = {}
        # Ghost state below is an oracle, not authority available to instructions.
        self.mappings = {}
        self.reservations = {}
        self.roots = {}
        self.table_owners = {}
        self.admitted = set()
        self.slot_history = {}
        self.completed = []
        self.observations = []
        self.effects = []
        self.admission_images = []

    def key(self):
        # Exact textual equality, not a lossy hash. Keeping nested Python
        # objects as visited keys dominated the breadth-first search's memory.
        return repr(freeze(self.__dict__))

    def clone(self):
        return deepcopy(self)

    def node(self, parent=None, binding=None, linear=True):
        result = self.next_node
        self.next_node += 1
        self.nodes[result] = Node(parent, binding, linear)
        return result

    def live(self, node):
        while node is not None:
            n = self.nodes[node]
            if not n.valid:
                return False
            node = n.parent
        return True

    def descendants(self, ancestor):
        result = set()
        for node in self.nodes:
            parent = self.nodes[node].parent
            while parent is not None:
                if parent == ancestor:
                    result.add(node)
                    break
                parent = self.nodes[parent].parent
        return result

    def bootstrap(self, page, location, handle):
        """Fixture-only initial authority, before adversarial execution starts."""
        require(page not in self.memory, "already bootstrapped")
        require(0 <= page < (PHYSICAL_LIMIT - PHYSICAL_BASE) // WORDS,
                "physical address limit")
        require(location not in self.wallet and handle not in self.wallet,
                "occupied bootstrap output")
        senior = self.node()
        junior = self.node(parent=senior)
        self.wallet[handle] = Cap(senior, "rev_phys", page=page)
        self.wallet[location] = Cap(junior, "physical", page=page)
        self.memory[page] = [Word() for _ in range(WORDS)]

    def bootstrap_domain(self, domain, handle):
        """Fixture-only recipient authority; domain creation is outside scope."""
        require(domain in HARTS and handle not in self.wallet, "invalid domain setup")
        self.wallet[handle] = Cap(self.node(), "domain", domain=domain)

    def get(self, location, kinds=None):
        require(location in self.wallet, "empty capability location")
        cap = self.wallet[location]
        require(self.live(cap.node), "revoked capability")
        require(kinds is None or cap.kind in kinds, "wrong capability kind")
        return cap

    def vacant(self, location):
        require(location not in self.wallet, "occupied output")
        if self.preparation:
            require(location not in self.preparation.held, "reserved operand")
        if self.barrier:
            require(location != self.barrier.output, "reserved return")
        for access in self.accesses.values():
            if access.phase not in ("done", "fault", "cancelled"):
                require(location not in (access.source, access.destination),
                        "reserved transfer location")

    def mapping(self, binding, for_access=False):
        require(binding is not None, "no mapping binding")
        ident, generation = binding
        if self.mutant == "weak_binding" and for_access:
            # Deliberately wrong: a context/global entry substitutes for the id.
            ident = min(self.registry, default=ident)
        require(ident in self.registry, "missing mapping")
        mapping = self.registry[ident]
        require(mapping.binding[1] == generation, "wrong generation")
        return mapping

    def table(self, cap):
        require(cap.kind == "table" and self.live(cap.node), "dead/wrong table")
        return self.memory[cap.page]

    def indices(self, address):
        require(LOGICAL_BASE <= address < ADDRESS_LIMIT, "unsupported geometry")
        # Each modeled range fits one aligned four-page window. The binding
        # selects the root; its low address bits select slots within that root.
        page = (address // WORDS) % PAGES
        return page // FANOUT, page % FANOUT

    def leaf(self, mapping, address):
        top, low = self.indices(address)
        require(mapping.lo <= address < mapping.hi, "outside mapping")
        entry = self.table(mapping.root)[top]
        require(isinstance(entry, Cap), "missing/locked table entry")
        return entry, low, self.table(entry)[low]

    def physical_page(self, cap, uninit=False):
        kinds = ("physical", "uninit") if uninit else ("physical",)
        require(cap.kind in kinds and cap.linear and self.live(cap.node),
                "exclusive physical page required")
        require(cap.page in self.memory and cap.lo == 0 and cap.hi == WORDS,
                "whole physical page required")
        if uninit:
            # A table will receive writes after initialization too. Its page
            # authority must retain W even when the input is currently UNINIT.
            require("w" in cap.rights, "table page must retain write authority")

    def convert(self, cap, binding, operation):
        # Publication calls this only after all authority/geometry checks pass.
        if self.mutant != "uncleared_" + operation:
            self.memory[cap.page] = [Word() for _ in range(WORDS)]
        self.table_owners[cap.node] = (binding, cap.page)
        if self.barrier_mode == "table_record":
            self.table_bindings[cap.node] = binding
        return replace(cap, kind="table", binding=binding, cursor=0)

    def create_range(self, lo, hi):
        require(partition_ok(lo, hi, window=PAGES * WORDS), "invalid mapping geometry")
        require(all(hi <= m.lo or m.hi <= lo for m in self.registry.values()),
                "logical range reserved")

    def delivery(self, recipient, pointer):
        cap = self.get(recipient, ("domain",))
        if self.mutant == "monitor_delivery" and isinstance(pointer, str):
            return pointer
        require(isinstance(pointer, ResumeSlot), "protected resume slot required")
        require(pointer.domain == cap.domain and bool(pointer.name),
                "resume slot belongs to another domain")
        return pointer.location

    def prepare_create(self, ident, root, recipient, pointer, handle, lo, hi,
                       rights="rw"):
        require(self.preparation is None, "preparation busy")
        require(self.barrier is None, "revocation busy")
        require(ident in self.last_generation and ident not in self.registry,
                "mapping id unavailable")
        require(self.last_generation[ident] < self.generations,
                "generation exhausted; id retired")
        self.create_range(lo, hi)
        destination = self.delivery(recipient, pointer)
        require(rights in ("r", "rw"), "unsupported protection")
        cap = self.get(root)
        self.physical_page(cap, uninit=True)
        require(len({root, recipient, destination, handle}) == 4, "aliased operands")
        self.vacant(destination)
        self.vacant(handle)
        self.preparation = Preparation("create", (ident, root, recipient, pointer,
                                                   handle, lo, hi, rights), {root: cap})
        del self.wallet[root]

    def populate_plan(self, handle, address, frame, pages):
        require(handle.kind == "detach" and self.live(handle.node),
                "detach handle required")
        mapping = self.mapping(handle.binding)
        require(mapping.state == "ACTIVE", "mapping not active")
        require(address % WORDS == 0 and mapping.lo <= address < mapping.hi,
                "page address outside mapping")
        self.physical_page(frame)
        require("w" in frame.rights, "anonymous frame must retain write authority")
        require(set(mapping.rights) <= set(frame.rights), "insufficient frame rights")
        top, low = self.indices(address)
        entry = self.table(mapping.root)[top]
        if entry == Word():
            require(len(pages) == 1, "one missing table page required")
            self.physical_page(pages[0], uninit=True)
            require(pages[0].page != frame.page, "overlapping operands")
        else:
            require(isinstance(entry, Cap), "locked table slot")
            require(self.table(entry)[low] == Word(), "PTE not none")
            require(not pages, "unexpected table pages")
        return mapping, top, low, entry

    def prepare_populate(self, handle, address, frame, pages=()):
        require(self.preparation is None, "preparation busy")
        require(self.barrier is None, "revocation busy")
        locations = (frame,) + tuple(pages)
        require(len(set(locations + (handle,))) == len(locations) + 1,
                "aliased input operands")
        caps = {loc: self.get(loc) for loc in locations}
        self.populate_plan(self.get(handle), address, caps[frame],
                           [caps[p] for p in pages])
        self.preparation = Preparation("populate", (handle, address, frame,
                                                     tuple(pages)), caps)
        for loc in locations:
            del self.wallet[loc]

    def abort(self):
        require(self.preparation is not None, "no preparation")
        # A concurrent senior revoke may have killed held operands. Restoring
        # their tagged but dead representations does not resurrect authority.
        self.wallet.update(self.preparation.held)
        self.preparation = None

    def publish(self):
        require(self.preparation is not None, "no preparation")
        require(self.barrier is None, "revocation busy")
        p = self.preparation
        if p.operation == "create":
            ident, root, recipient, pointer, handle, lo, hi, rights = p.arguments
            require(ident not in self.registry, "mapping id unavailable")
            self.create_range(lo, hi)
            destination = self.delivery(recipient, pointer)
            self.physical_page(p.held[root], uninit=True)
            require(destination not in self.wallet and handle not in self.wallet,
                    "occupied publication output")
            generation = self.last_generation[ident] + 1
            require(generation <= self.generations, "generation exhausted")
            binding = (ident, generation)
            senior = self.node(binding=binding)
            junior = self.node(parent=senior, binding=binding)
            root_cap = self.convert(p.held[root], binding, "create")
            mapping = Mapping(binding, root_cap, senior, lo, hi, rights)
            self.registry[ident] = mapping
            self.last_generation[ident] = generation
            self.wallet[destination] = Cap(junior, "logical", binding=binding,
                                           lo=lo, hi=hi, rights=rights, cursor=lo)
            self.wallet[handle] = Cap(senior, "detach", binding=binding,
                                      lo=lo, hi=hi, rights=rights, cursor=lo)
            self.mappings[binding] = mapping
            self.reservations[ident] = binding
            self.roots[binding] = root_cap
        else:
            handle, address, frame, pages = p.arguments
            mapping, top, low, entry = self.populate_plan(
                self.get(handle), address, p.held[frame], [p.held[x] for x in pages])
            if self.mutant != "unzeroed_frame":
                self.memory[p.held[frame].page] = [Word() for _ in range(WORDS)]
            # Independent admission evidence, never consulted by an instruction.
            self.admission_images.append(tuple(self.memory[p.held[frame].page]))
            if pages:
                entry = self.convert(p.held[pages[0]], mapping.binding, "populate")
                self.memory[mapping.root.page][top] = entry
                self.admitted.add((mapping.root.node, top, entry.node))
                self.slot_history[(mapping.root.node, top)] = entry.node
            self.memory[entry.page][low] = p.held[frame]
            self.admitted.add((entry.node, low, p.held[frame].node))
            self.slot_history[(entry.node, low)] = p.held[frame].node
        self.preparation = None

    def move(self, source, destination):
        cap = self.get(source)
        require(cap.kind != "table", "table is not software-loadable")
        self.vacant(destination)
        self.wallet[destination] = cap
        if cap.linear:
            del self.wallet[source]

    def delin(self, location):
        cap = self.get(location, ("physical", "logical"))
        require(cap.linear, "already non-linear")
        self.wallet[location] = replace(cap, linear=False)
        self.nodes[cap.node].linear = False

    def split(self, location, middle, other):
        cap = self.get(location, ("logical",))
        require(cap.linear and cap.lo < middle < cap.hi, "invalid split")
        self.vacant(other)
        parent = self.nodes[cap.node].parent
        self.nodes[cap.node].valid = False
        left = self.node(parent, cap.binding)
        right = self.node(parent, cap.binding)
        self.wallet[location] = replace(cap, node=left, hi=middle)
        self.wallet[other] = replace(cap, node=right, lo=middle, cursor=middle)

    def mrev(self, location, handle):
        cap = self.get(location, ("logical", "physical"))
        require(cap.linear, "MREV requires linear authority")
        self.vacant(handle)
        parent = self.nodes[cap.node].parent
        senior = self.node(parent, cap.binding)
        self.nodes[cap.node].parent = senior
        kind = "rev_logical" if cap.kind == "logical" else "rev_phys"
        self.wallet[handle] = replace(cap, node=senior, kind=kind)

    def drop(self, location):
        require(self.barrier is None, "barrier busy")
        cap = self.get(location, ("logical",))
        if cap.linear:
            self.nodes[cap.node].valid = False
            self.barrier = Barrier("drop", "", cap, cap.binding, {cap.node}, "discard")
            self.barrier.scope = self._scope(cap.binding, {cap.node})
        del self.wallet[location]

    def narrow(self, location, lo, hi, rights):
        cap = self.get(location, ("logical", "physical"))
        require(cap.lo <= lo < hi <= cap.hi, "widened bounds")
        require(set(rights) <= set(cap.rights), "widened rights")
        self.wallet[location] = replace(cap, lo=lo, hi=hi, rights=rights)

    def cursor(self, location, address):
        cap = self.get(location, ("logical",))
        # Forming a one-past pointer is legal; its use is checked at issue.
        require(cap.lo <= address <= cap.hi, "unrepresentable cursor")
        self.wallet[location] = replace(cap, cursor=address)

    def _scope(self, binding, affected):
        if self.barrier_mode == "global":
            return None
        if binding is not None:
            return frozenset((binding,))
        bindings = frozenset(self.table_bindings[n] for n in affected
                             if n in self.table_bindings)
        # No frame-to-mapping index: unclassified physical REVOKE remains global.
        return bindings or None

    def _blocked(self, binding):
        return self.barrier is not None and (
            self.barrier.scope is None or binding in self.barrier.scope)

    def begin_revoke(self, handle, output):
        require(self.barrier is None, "barrier busy")
        kinds = ("rev_phys", "rev_logical")
        if self.mutant == "plain_detach":
            kinds += ("detach",)
        cap = self.get(handle, kinds)
        require(output.split(":", 1)[0] == handle.split(":", 1)[0],
                "revocation result belongs to the issuing context")
        self.vacant(output)
        affected = {n for n in self.descendants(cap.node) if self.live(n)}
        uninit = any(self.nodes[n].linear for n in affected)
        physical = cap.kind == "rev_phys"
        kind = ("uninit" if uninit else "physical") if physical else (
            "logical_uninit" if uninit else "logical")
        self.barrier = Barrier("revoke", output, cap, cap.binding, affected, kind)
        self.barrier.scope = self._scope(cap.binding, affected)
        del self.wallet[handle]
        for node in affected:
            self.nodes[node].valid = False
        if self.mutant in ("root_release", "stale_record"):
            for binding, root in self.roots.items():
                if root.node in affected:
                    if self.mutant == "root_release" or (
                            binding[0] in self.registry and
                            self.registry[binding[0]].binding != binding):
                        self.registry.pop(binding[0], None)

    def begin_detach(self, handle, output):
        require(self.barrier is None, "barrier busy")
        cap = self.get(handle, ("detach",))
        mapping = self.mapping(cap.binding)
        require(mapping.state == "ACTIVE", "mapping not active")
        self.vacant(output)
        affected = {n for n in self.descendants(cap.node) if self.live(n)}
        self.barrier = Barrier("detach", output, cap, cap.binding, affected, "token")
        self.barrier.scope = self._scope(cap.binding, affected)
        del self.wallet[handle]
        mapping.state = "DETACHING"
        for node in affected:
            self.nodes[node].valid = False

    def begin_unmap(self, token, address, output):
        require(self.barrier is None, "barrier busy")
        cap = self.get(token, ("token",))
        mapping = self.mapping(cap.binding)
        require(mapping.state == "DETACHED", "mapping not detached")
        require(address % WORDS == 0, "unaligned UNMAP")
        table, index, frame = self.leaf(mapping, address)
        require(isinstance(frame, Cap) and frame.kind == "physical" and
                self.live(frame.node), "PTE not present")
        self.vacant(output)
        self.memory[table.page][index] = LOCKED
        self.barrier = Barrier("unmap", output, frame, cap.binding,
                               {frame.node}, "uninit")
        self.barrier.scope = self._scope(cap.binding, {frame.node})

    def invalidate(self, hart):
        b = self.barrier
        require(b is not None and hart in HARTS, "no barrier/hart")
        require(hart not in b.invalidated, "already invalidated")
        self.tlb = {k: v for k, v in self.tlb.items()
                    if k[0] != hart or not self._blocked(v.binding)}
        for request, access in list(self.accesses.items()):
            if access.hart == hart and self._blocked(access.cap.binding) and \
                    access.phase in ("root", "leaf", "fill"):
                self.cancel(request)
        b.invalidated.add(hart)

    def drain(self, hart):
        b = self.barrier
        require(b is not None and hart in b.invalidated, "invalidate first")
        require(hart not in b.drained, "already drained")
        if self.mutant != "walk_only_drain":
            require(not any(a.hart == hart and a.phase == "checked" and
                            self._blocked(a.cap.binding)
                            for a in self.accesses.values()), "data access outstanding")
        b.drained.add(hart)

    def finish(self):
        b = self.barrier
        require(b is not None and b.drained == set(HARTS), "drain incomplete")
        if b.return_kind != "discard":
            self.wallet[b.output] = replace(b.held, kind=b.return_kind,
                                            linear=True, cursor=b.held.lo)
        if b.operation == "detach":
            self.mapping(b.binding).state = "DETACHED"
        self.completed.append((b.operation, b.binding, frozenset(b.affected),
                               self.next_access))
        self.barrier = None

    def destroy(self, token):
        require(self.barrier is None, "barrier busy")
        cap = self.get(token, ("token",))
        mapping = self.mapping(cap.binding)
        require(mapping.state == "DETACHED" and cap.lo == mapping.lo and
                cap.hi == mapping.hi, "whole-mapping token required")
        mapping.state = "DESTROYED"
        del self.registry[cap.binding[0]]
        del self.reservations[cap.binding[0]]
        del self.wallet[token]
        if self.mutant == "duplicate_return":
            # Wrong recovery from a saved root address, ignoring its dead node.
            root = self.roots[cap.binding]
            self.wallet["m:duplicate"] = Cap(self.node(), "uninit", page=root.page)

    def scrub(self, location):
        cap = self.get(location, ("uninit",))
        require("w" in cap.rights, "scrub requires retained write authority")
        require(cap.cursor < cap.hi, "scrub complete")
        self.memory[cap.page][cap.cursor] = Word()
        self.wallet[location] = replace(cap, cursor=cap.cursor + 1)

    def init(self, location):
        cap = self.get(location, ("uninit", "logical_uninit"))
        require(cap.cursor == cap.hi, "write the whole page first")
        kind = "physical" if cap.kind == "uninit" else "logical"
        self.wallet[location] = replace(cap, kind=kind, cursor=cap.lo)

    def scrub_logical(self, location):
        require(self.barrier is None, "barrier busy")
        cap = self.get(location, ("logical_uninit",))
        require("w" in cap.rights, "scrub requires retained write authority")
        require(cap.cursor < cap.hi, "scrub complete")
        mapping = self.mapping(cap.binding)
        require(mapping.state == "ACTIVE", "mapping not active")
        _, _, frame = self.leaf(mapping, cap.cursor)
        require(isinstance(frame, Cap) and frame.kind == "physical" and
                self.live(frame.node) and "w" in frame.rights, "invalid backing")
        self.memory[frame.page][cap.cursor % WORDS] = Word()
        self.wallet[location] = replace(cap, cursor=cap.cursor + 1)

    def physical_read(self, location, offset=0):
        cap = self.get(location, ("physical",))
        require("r" in cap.rights and cap.lo <= offset < cap.hi, "read denied")
        word = self.memory[cap.page][offset]
        self.observations.append((location, word))
        return word

    def physical_store_cap(self, page_location, offset, source):
        """Ordinary tagged store before a page is converted to a table."""
        page = self.get(page_location, ("physical",))
        cap = self.get(source)
        require("w" in page.rights and page.lo <= offset < page.hi, "store denied")
        require(cap.kind != "table", "table cannot be stored by software")
        self.memory[page.page][offset] = cap
        if cap.linear:
            del self.wallet[source]

    def physical_store(self, location, offset, value):
        cap = self.get(location, ("physical",))
        require("w" in cap.rights and cap.lo <= offset < cap.hi, "store denied")
        self.memory[cap.page][offset] = Word(value)

    def address_value(self, location):
        """Numeric cursor projection used by the existing comparison lowering."""
        cap = self.get(location, ("logical", "physical"))
        if cap.kind == "physical":
            return PHYSICAL_BASE + cap.page * WORDS + cap.cursor
        return cap.cursor

    def issue(self, hart, location, operation="load", address=None, value=7,
              source=None, destination=None):
        require(hart in HARTS and operation in ("load", "store", "amo", "cload", "cstore"),
                "unsupported access")
        require(location.startswith(("d%d:" % hart, "h%d:" % hart, "m:")),
                "capability is not in this context")
        require(not any(a.hart == hart and a.phase not in ("done", "fault", "cancelled")
                        for a in self.accesses.values()), "hart access busy")
        cap = self.get(location, ("logical",))
        require(not self._blocked(cap.binding), "new accesses blocked during barrier")
        address = cap.cursor if address is None else address
        require(cap.lo <= address < cap.hi, "logical bounds")
        needed = "r" if operation in ("load", "cload") else "w"
        require(needed in cap.rights and (operation != "amo" or "r" in cap.rights),
                "logical permissions")
        mapping = self.mapping(cap.binding, for_access=True)
        require(mapping.state == "ACTIVE", "mapping not active")
        carried = None
        destination = destination or "h%d:result" % hart
        if operation == "cload":
            self.vacant(destination)
        if operation == "cstore":
            require(source is not None, "missing transfer source")
            require(source.startswith(("d%d:" % hart, "h%d:" % hart, "m:")),
                    "transfer source is not in this context")
            carried = self.get(source)
            require(carried.kind != "table", "table cannot be transferred")
        access = Access(hart, cap, address, operation, Word(value, True), destination,
                        source=source, carried=carried)
        cached = self.tlb.get((hart, *mapping.binding, address // WORDS))
        if cached:
            self.check_rights(access, cached.frame)
            access.translation = cached
            access.path = list(cached.dependencies)
            access.phase = "checked"
        if carried and carried.linear:
            del self.wallet[source]
        request = self.next_access
        self.next_access += 1
        self.accesses[request] = access
        return request

    def check_rights(self, access, frame):
        needed = "r" if access.operation in ("load", "cload") else "w"
        require(needed in frame.rights and
                (access.operation != "amo" or "r" in frame.rights), "backing rights")

    def walk(self, request):
        access = self.accesses[request]
        require(access.phase in ("root", "leaf", "fill"), "not walking")
        try:
            if access.phase == "root":
                mapping = self.mapping(access.cap.binding, for_access=True)
                top, _ = self.indices(access.address)
                entry = self.table(mapping.root)[top]
                require(isinstance(entry, Cap), "missing/locked path")
                access.path = [mapping.root.node]
                access.table = entry
                access.phase = "leaf"
            elif access.phase == "leaf":
                _, low = self.indices(access.address)
                frame = self.table(access.table)[low]
                require(isinstance(frame, Cap) and frame.kind == "physical" and
                        self.live(frame.node), "absent/locked/nonphysical PTE")
                self.check_rights(access, frame)
                access.path += [access.table.node, frame.node]
                # The binding belongs to the table actually selected, not the
                # requesting capability; the independent checker compares them.
                binding = self.mapping(access.cap.binding, for_access=True).binding
                access.translation = Translation(binding, frame, tuple(access.path), request)
                access.phase = "fill"
            else:
                t = access.translation
                self.tlb[(access.hart, *t.binding, access.address // WORDS)] = t
                access.phase = "checked"
        except Refused as fault:
            access.reason = str(fault)
            self.cancel(request, fault=True)

    def cancel(self, request, fault=False):
        access = self.accesses[request]
        require(access.phase not in ("done", "fault", "cancelled"), "already completed")
        if access.carried and access.carried.linear:
            self.wallet[access.source] = access.carried
        access.carried = None
        access.phase = "fault" if fault else "cancelled"

    def memory_step(self, request):
        access = self.accesses[request]
        require(access.phase == "checked", "access not checked")
        frame = access.translation.frame
        offset = access.address % WORDS
        old = self.memory[frame.page][offset]
        try:
            if access.operation == "load":
                # A scalar read does not transfer a capability tag.
                access.result = old if isinstance(old, Word) else Word()
            elif access.operation == "store":
                self.memory[frame.page][offset] = access.value
            elif access.operation == "amo":
                require(isinstance(old, Word), "AMO on tagged word unsupported")
                access.result = old
                self.memory[frame.page][offset] = Word(old.value + 1, True)
            elif access.operation == "cload":
                require(isinstance(old, Cap), "not a tagged capability")
                if old.linear:
                    require("w" in frame.rights and "w" in access.cap.rights,
                            "linear load requires write permission")
                    self.memory[frame.page][offset] = Word()
                self.wallet[access.destination] = old
                access.result = old
            else:
                self.memory[frame.page][offset] = access.carried
                access.carried = None
            access.phase = "done"
            self.effects.append((request, access.cap.binding, access.translation,
                                 access.operation))
        except Refused as fault:
            access.reason = str(fault)
            self.cancel(request, fault=True)

    def locations(self):
        for name, cap in self.wallet.items():
            yield name, cap
        for page, words in self.memory.items():
            for index, word in enumerate(words):
                if isinstance(word, Cap):
                    yield "page:%d:%d" % (page, index), word
        for ident, mapping in self.registry.items():
            yield "registry:%d" % ident, mapping.root
        if self.preparation:
            for name, cap in self.preparation.held.items():
                yield "preparing:" + name, cap
        if self.barrier:
            yield "barrier", self.barrier.held
        for request, access in self.accesses.items():
            if access.carried:
                yield "transfer:%d" % request, access.carried

    def check(self):
        """Ghost assertions never repair state or gate an instruction."""
        owned = {}
        physical = {}
        for location, cap in self.locations():
            if cap.kind in ("logical", "logical_uninit", "rev_logical", "detach", "token"):
                invariant(cap.binding is not None and cap.binding == self.nodes[cap.node].binding,
                          "I7: derivation/storage changed binding")
            else:
                invariant(self.nodes[cap.node].binding is None,
                          "I7: logical authority reinterpreted as physical")
            if not self.live(cap.node):
                continue
            if cap.kind == "table":
                invariant(location.startswith(("registry:", "page:")),
                          "I2: table capability escaped protected storage")
            if cap.linear:
                invariant(cap.node not in owned, "I1: duplicated linear authority")
                owned[cap.node] = location
            if cap.kind in ("physical", "uninit", "table"):
                previous = physical.setdefault(cap.page, [])
                invariant(not previous or (not cap.linear and
                                           all(not c.linear for c in previous)),
                          "I1: overlapping physical authority")
                previous.append(cap)
            if location.startswith("m:") and cap.kind == "logical":
                invariant(False, "I3: monitor obtained logical data authority")
        for node, (_, page) in self.table_owners.items():
            if not self.live(node):
                continue
            for index, entry in enumerate(self.memory[page]):
                if isinstance(entry, Cap):
                    invariant(entry.linear and entry.kind in ("physical", "table"),
                              "I4: invalid PRIVATE table entry")
                    invariant((node, index, entry.node) in self.admitted,
                              "I4: entry bypassed POPULATE")
                else:
                    invariant(entry in (Word(), LOCKED), "I4: uninitialized table slot")
                key = (node, index)
                if key in self.slot_history:
                    invariant(entry == LOCKED or (isinstance(entry, Cap) and
                              entry.node == self.slot_history[key]),
                              "I5: previously used slot replaced or cleared")
        invariant({i: m.binding for i, m in self.registry.items()} == self.reservations,
                  "I5: registry reservation released without DESTROY")
        for image in self.admission_images:
            invariant(all(word == Word() for word in image),
                      "I4: anonymous frame not zeroed before publication")
        ranges = sorted((m.lo, m.hi) for m in self.registry.values())
        invariant(all(PHYSICAL_LIMIT < lo < hi < ADDRESS_LIMIT for lo, hi in ranges),
                  "goal: logical address overlaps physical space or wraps")
        invariant(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])),
                  "goal: distinct mappings have overlapping addresses")
        for binding, mapping in self.mappings.items():
            invariant(mapping.root == self.roots[binding], "I5: root changed")
            if mapping.state in ("DETACHED", "DESTROYED"):
                invariant(not any(self.live(n) for n in self.descendants(mapping.pointer_root)),
                          "I6: logical authority survived DETACH")
        for access in self.accesses.values():
            if access.phase in ("checked", "done"):
                invariant(access.translation.binding == access.cap.binding,
                          "goal: live pointer redirected to another mapping")
        for operation, binding, affected, cutoff in self.completed:
            for request, access in self.accesses.items():
                if request >= cutoff or access.phase in ("done", "fault", "cancelled"):
                    continue
                relevant = (operation == "detach" and access.cap.binding == binding) or (
                    access.cap.node in affected or bool(affected.intersection(access.path)))
                invariant(not relevant, "I6: outstanding access after return")
            for translation in self.tlb.values():
                if translation.issued >= cutoff:
                    continue
                relevant = (operation == "detach" and translation.binding == binding) or (
                    bool(affected.intersection(translation.dependencies)))
                invariant(not relevant, "I6: stale translation after return")
        for location, word in self.observations:
            invariant(not (location.startswith("m:") and isinstance(word, Word) and word.private),
                      "goal: monitor read domain bytes")


@dataclass(frozen=True)
class Action:
    operation: str
    arguments: tuple = ()

    def apply(self, machine):
        require(not self.operation.startswith("_") and self.operation not in
                ("node", "bootstrap", "bootstrap_domain", "convert"),
                "not an execution action")
        return getattr(machine, self.operation)(*self.arguments)

    def __str__(self):
        return "%s%r" % (self.operation, self.arguments)
