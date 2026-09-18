/* Twenty CPython pymalloc defects, as domain programs.
 *
 * Every consumer-side temporal defect live in the pinned 3.13.7 whose freed
 * memory is pymalloc's. The inventory that produced the list, and what was
 * excluded and why, is docs/ref/cpython-pymalloc-defects.md.
 *
 * One program, one defect per boot, selected through the first event's id --
 * the port's own security tests are built this way, and for the same reason: a
 * capability fault ends the domain, so a case that provokes one cannot also
 * report the results beside it.
 *
 * Each case runs twice against the SAME binary:
 *
 *   spatial  mode 0. pym_issue and pym_release keep the block's alias, so a
 *            freed block stays addressable and the stale access SUCCEEDS.
 *            Expected: completed.
 *   sublet   mode 1. Each issue and release does sublet_give then sublet_take,
 *            so the stale pointer is a revoked alias. Expected: a capability
 *            fault at the declared instruction.
 *
 * That pairing is the claim. Note what the spatial arm is and is not: it is
 * this port's unprotected baseline, not a CHERI model. CHERI would narrow
 * bounds per allocation and would still miss all twenty, because a reused block
 * stays tagged and in bounds -- but that argument is made in prose, not by this
 * arm.
 *
 * WHAT IS REAL AND WHAT IS REDUCED
 *
 * The allocator is real: obmalloc.c from the pinned CPython 3.13.7, compiled
 * unmodified but for the capability-ABI and Sublet patches the port applies.
 * The consumers are reduced to the allocator calls the upstream defect makes,
 * in the same order, because reaching them in place needs a running
 * interpreter -- which the port explicitly does not put in a domain. Each case
 * names its upstream fix; the per-case PROVENANCE.md says line by line what was
 * reduced.
 *
 * TWENTY REPORTS, NINE SHAPES, AND THAT RATIO IS THE FINDING
 *
 * Eight of the twenty -- cases 0, 1, 3, 8, 12, 16, 17, 19 -- reduce to the same
 * sequence: free a small object, allocate the same size again, read through the
 * pointer that was kept. They are eight separately reported defects across
 * seven modules, fixed one at a time over more than a year. They are kept apart
 * rather than merged because the sameness is the point: one revocation
 * mechanism covers a class upstream has to keep rediscovering.
 *
 * The other shapes, each held by one or two cases: an interior pointer into a
 * freed block (2, 13); a stale entry reached through a live array (4); a
 * pointer LOADED out of a freed block and followed (5); a payload buffer (6, 9);
 * a cursor surviving in a struct field across two API calls (7); a block ended
 * by a REALLOC that moved it (10); a free and a use on adjacent lines with no
 * callback in between (11); a dangling pointer parked in a surviving object or
 * a global, with no bound on when it is next read (14, 18); and a bare
 * PyMem_Malloc block cached by a third party (15).
 *
 * SIZE. Every block here is well under pymalloc's 512-byte threshold, so all of
 * it is pool memory that never reaches malloc. Three cases have upstream
 * defects that can exceed it -- 6 (a join buffer), 10 (bytearray storage) and
 * 19 (a buffered-input snapshot). For those the same defect becomes an ordinary
 * malloc use-after-free that ASan does see, which is why the size is pinned
 * here and stated in each of their PROVENANCE.md files.
 */
#include "port.h"
#include <string.h>

#define CHECK(x, n)                                                            \
  do {                                                                         \
    if (!(x))                                                                  \
      pym_fail(n);                                                             \
  } while (0)

/* Small enough to be a pymalloc block on any of the port's size classes, and
 * the same for every case so that a freed block is reused by the next
 * allocation rather than landing in a different class. */
#define OBJ 48

static volatile unsigned char *held;

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere in the program. */
__attribute__((noinline)) static unsigned
read_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl pyc_defect_read\npyc_defect_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
}

__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long value = 93;
  __asm__ volatile(
      ".globl pyc_defect_write\npyc_defect_write:\nsb %0, 0(%1)\n" ::"r"(value),
      "r"(p)
      : "memory");
}

/* Publish the case and both probe addresses, so the host knows which
 * instruction a fault is allowed to be at. */
static void mark(unsigned which) {
  extern void pyc_defect_read(void), pyc_defect_write(void);
  unsigned long code = 0xcf19000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(pyc_defect_read), "r"(pyc_defect_write)
                   : "memory");
}

/* An odict node: a link the loop follows, and a payload. Sized so the whole
 * node is one pymalloc block. */
struct node {
  struct node *next;
  unsigned char payload[16];
};

static void defect(unsigned which) {
  if (which == 0) {
    /* gh-143543 -- itertools.groupby. groupby compares gbo->tgtkey with
     * gbo->currkey, both borrowed. A user-defined __eq__ re-enters the
     * iterator and advances it, dropping the last reference to the key being
     * compared; the comparison then continues through the stale pointer. */
    unsigned char *key = pym_malloc(OBJ);
    CHECK(key, 701);
    key[0] = 17;
    held = key;                /* gbo->currkey, borrowed by the comparison */
    pym_free(key);             /* the re-entrant __eq__ advanced the iterator */
    unsigned char *successor = pym_malloc(OBJ); /* the next group's key */
    CHECK(successor == key, 702); /* the block comes straight back */
    successor[0] = 19;
    mark(0);
    (void)read_probe(held);
  } else if (which == 1) {
    /* gh-146613 -- itertools._grouper, the child iterator. Two keys are live
     * at once: igo->tgtkey in the grouper and gbo->currkey in the parent. The
     * comparison frees the parent's while the grouper is still holding its
     * borrowed alias. */
    unsigned char *tgtkey = pym_malloc(OBJ);
    unsigned char *currkey = pym_malloc(OBJ);
    CHECK(tgtkey && currkey && tgtkey != currkey, 703);
    tgtkey[0] = 23;
    currkey[0] = 29;
    held = currkey;
    pym_free(currkey);                          /* the parent advanced */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == currkey && tgtkey[0] == 23, 704);
    fresh[0] = 31;
    mark(1);
    (void)read_probe(held);
  } else if (which == 2) {
    /* gh-142829 -- Context.__eq__ through _PyHamt_Eq. The comparison walks the
     * map with an iterator whose state points INSIDE the node it is visiting;
     * a re-entrant ContextVar.set drops the last reference to that node, and
     * the walk resumes from the interior pointer. */
    unsigned char *hamt_node = pym_malloc(OBJ);
    CHECK(hamt_node, 705);
    memset(hamt_node, 37, OBJ);
    held = hamt_node + 16;     /* iter.i_nodes[level], mid-node */
    pym_free(hamt_node);       /* the re-entrant set dropped the last ref */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == hamt_node, 706);
    memset(fresh, 41, OBJ);
    mark(2);
    (void)read_probe(held);
  } else if (which == 3) {
    /* gh-142831 -- the JSON encoder over a dict's items list. The item is
     * borrowed from a list that stays live; user code invoked from the encoder
     * mutates the list, dropping the item. The stale pointer is therefore
     * reached through a LIVE array, not through a local. */
    void **ob_item = pym_malloc(OBJ); /* the items list's storage */
    unsigned char *item = pym_malloc(OBJ);
    CHECK(ob_item && item, 707);
    item[0] = 43;
    ob_item[0] = item;         /* PyList_GET_ITEM(items, i), borrowed */
    pym_free(item);            /* the default callback mutated the list */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == item, 708);
    fresh[0] = 47;
    mark(3);
    (void)read_probe(ob_item[0]);
  } else if (which == 4) {
    /* gh-145244 -- the JSON encoder's borrowed dict key, on the ERROR path.
     * key comes from PyDict_Next; the default callback clears the dict, which
     * frees every entry at once, and the error path then formats the key with
     * _PyErr_FormatNote("%R", key). The stale access happens while unwinding,
     * which is where a fault is least expected.
     *
     * Verified live at the pin by reading the source, not by the apply test:
     * Modules/_json.c:1621 of v3.13.7 passes key and value to
     * encoder_encode_key_value with no Py_INCREF at all. */
    unsigned char *entries[4];
    for (unsigned i = 0; i < 4; ++i) {
      entries[i] = pym_malloc(OBJ);
      CHECK(entries[i], 709);
      entries[i][0] = (unsigned char)(50 + i);
    }
    held = entries[2];         /* the key the error path will format */
    for (unsigned i = 0; i < 4; ++i)
      pym_free(entries[i]);    /* PyDict_Clear, in the callback */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh, 710);
    fresh[0] = 59;
    mark(4);
    (void)read_probe(held);
  } else if (which == 5) {
    /* gh-148660 -- OrderedDict.copy() under re-entrant mutation. The copy loop
     * is _odict_FOREACH, which advances by reading node->next OUT OF the node
     * it just processed. Re-entrant mutation frees that node, so the loop
     * reads its link field from freed memory and then follows it.
     *
     * This is the one case whose stale access is a POINTER load rather than a
     * byte read: the block is reused by another node, so in the spatial arm the
     * link reads back as a valid pointer and the walk continues -- silently
     * visiting the wrong node. */
    struct node *first = pym_malloc(sizeof *first);
    struct node *second = pym_malloc(sizeof *second);
    CHECK(first && second && first != second, 711);
    second->next = NULL;
    second->payload[0] = 61;
    first->next = second;
    first->payload[0] = 67;
    held = (volatile unsigned char *)first; /* the loop's cursor */
    pym_free(first);                        /* the re-entrant mutation */
    struct node *fresh = pym_malloc(sizeof *fresh);
    CHECK(fresh == first, 712);
    fresh->next = second;      /* the block is now a different node */
    fresh->payload[0] = 71;
    mark(5);
    (void)read_probe(held);    /* _odict_FOREACH reads node->next */
    struct node *next = ((struct node *volatile *)held)[0];
    CHECK(next == second && next->payload[0] == 61, 713);
  } else if (which == 6) {
    /* gh-151295 -- bytes.join()/bytearray.join() through a re-entrant
     * __buffer__. buffers[i].buf points INTO the item's payload; the item's
     * __buffer__ runs Python that drops the sequence's last reference to it.
     * The join then copies from the released buffer.
     *
     * SIZE MATTERS HERE AND NOWHERE ELSE IN THIS CORPUS. The payload is pinned
     * below pymalloc's 512-byte threshold, which is what makes the defect
     * invisible to a malloc-level tool. On a large input the same defect is an
     * ordinary malloc use-after-free and ASan reports it. */
    unsigned char *payload = pym_malloc(OBJ);
    CHECK(payload, 714);
    memset(payload, 73, OBJ);
    held = payload + 8;        /* buffers[i].buf, an interior pointer */
    pym_free(payload);         /* __buffer__ dropped the sequence's last ref */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == payload, 715);
    memset(fresh, 79, OBJ);
    mark(6);
    (void)read_probe(held);    /* the join's memcpy source */
  } else if (which == 7) {
    /* gh-148395 -- {LZMA,BZ2,_Zlib}Decompressor. On the error path the stream
     * struct keeps next_in pointing into the caller's input buffer, which the
     * caller then releases. The stale pointer therefore survives BETWEEN two
     * API calls, in a long-lived struct field, rather than inside one
     * operation -- and the next decompress() resumes from it. */
    struct stream {
      unsigned char *next_in;
      unsigned long avail_in;
    };
    struct stream *d = pym_malloc(sizeof *d);
    unsigned char *input = pym_malloc(OBJ);
    CHECK(d && input, 716);
    memset(input, 83, OBJ);
    d->next_in = input;        /* set by the failed decompress() */
    d->avail_in = OBJ;
    held = d->next_in;
    pym_free(input);           /* the caller released the input buffer */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == input, 717);
    memset(fresh, 89, OBJ);
    mark(7);
    (void)read_probe(held);    /* the NEXT decompress() resumes from next_in */
  } else if (which == 8) {
    /* gh-112127 -- atexit.unregister(). The loop compares the caller's func
     * against each registered callback, borrowing the tuple out of the live
     * callbacks list. PyObject_RichCompareBool runs a user __eq__, which can
     * call atexit.unregister again and mutate the list, dropping the tuple the
     * comparison is standing on -- and the loop then carries on to the next
     * index through the same list. */
    void **callbacks = pym_malloc(OBJ);   /* the list's ob_item */
    unsigned char *tuple = pym_malloc(OBJ);
    unsigned char *other = pym_malloc(OBJ);
    CHECK(callbacks && tuple && other, 719);
    tuple[0] = 97;
    other[0] = 101;
    callbacks[0] = tuple;
    callbacks[1] = other;
    pym_free(tuple);           /* the re-entrant unregister removed it */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == tuple && other[0] == 101, 720);
    fresh[0] = 103;
    mark(8);
    (void)read_probe(callbacks[0]);
  } else if (which == 9) {
    /* gh-139210 -- xml.etree.ElementTree.iterparse(). event_name is a C string
     * pointing INTO an item of events_seq. The pre-fix order drops the
     * sequence first and formats the message second, so PyErr_Format reads the
     * string out of memory the sequence took with it.
     *
     * The distinguishing feature is that nothing here is a PyObject* the
     * checker could have followed -- it is a char* into a payload, consumed by
     * a formatter on the error path. */
    unsigned char *item = pym_malloc(OBJ);   /* the event name string object */
    CHECK(item, 721);
    memset(item, 107, OBJ);
    held = item + 24;          /* event_name, into the string's payload */
    pym_free(item);            /* Py_DECREF(events_seq), before the format */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == item, 722);
    memset(fresh, 109, OBJ);
    mark(9);
    (void)read_probe(held);    /* PyErr_Format("unknown event '%s'", ...) */
  } else if (which == 10) {
    /* gh-142560 -- bytearray's search-like methods. They cache
     * PyByteArray_AS_STRING(self) and then call into code that can run user
     * Python, which may resize the bytearray. The resize REALLOCATES the
     * storage, and when it moves, the cached base pointer is left addressing
     * the old block.
     *
     * This is the only case in the corpus where the block is ended by a
     * REALLOC rather than a free, so the assertion that it really moved is
     * part of the case: a realloc that returned the same address would leave
     * the driver testing nothing at all. */
    unsigned char *storage = pym_malloc(OBJ);
    CHECK(storage, 723);
    memset(storage, 113, OBJ);
    held = storage;            /* the cached PyByteArray_AS_STRING(self) */
    unsigned char *grown = pym_realloc(storage, 300); /* user code resized it */
    CHECK(grown, 724);
    CHECK(grown != storage, 725); /* it MUST have moved, or this tests nothing */
    grown[0] = 127;
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == storage, 726); /* the old block came back */
    memset(fresh, 131, OBJ);
    mark(10);
    (void)read_probe(held);    /* _Py_bytes_find over the old base */
  } else if (which == 11) {
    /* gh-142783 -- the zoneinfo weak cache. get_weak_cache asked for the
     * attribute, immediately Py_XDECREF'd it, and returned what the comment
     * called "a borrowed reference" on the assumption that the type held one.
     * When it does not, the object is gone before the caller's first use.
     *
     * No re-entrancy and no user callback: the free and the use are adjacent
     * lines. Every other case here needs something to run in between. */
    unsigned char *cache = pym_malloc(OBJ);
    CHECK(cache, 727);
    cache[0] = 137;
    held = cache;
    pym_free(cache);           /* Py_XDECREF, one line after the lookup */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == cache, 728);
    fresh[0] = 139;
    mark(11);
    (void)read_probe(held);    /* PyObject_CallMethod(weak_cache, "get", ...) */
  } else if (which == 12) {
    /* gh-143004 -- collections.Counter.update via _count_elements. oldval is
     * borrowed from the mapping; PyNumber_Add runs a user __add__ that can
     * mutate or clear the dict, freeing the value while the sum is being
     * computed. The mapping itself survives, which is what separates this from
     * case 4: there the container was emptied and abandoned, here it is
     * emptied and kept. */
    void **values = pym_malloc(OBJ);
    unsigned char *oldval = pym_malloc(OBJ);
    CHECK(values && oldval, 729);
    oldval[0] = 149;
    values[0] = oldval;        /* the dict's slot, borrowed as oldval */
    pym_free(oldval);          /* the user __add__ cleared the dict */
    values[0] = NULL;          /* ... and the container is still live */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == oldval, 730);
    fresh[0] = 151;
    held = oldval;             /* PyNumber_Add is still holding it */
    mark(12);
    (void)read_probe(held);
  } else if (which == 13) {
    /* gh-144833 -- the SSL module when SSL_new() fails. The error path did
     * Py_DECREF(self) and then get_state_ctx(self), reading a field out of the
     * object it had just released.
     *
     * The stale access is to the freed object ITSELF, not to anything it
     * pointed at, and there is no second party involved at all. */
    unsigned char *self = pym_malloc(OBJ);
    CHECK(self, 731);
    memset(self, 157, OBJ);
    held = self + 8;           /* the ctx field inside self */
    pym_free(self);            /* Py_DECREF(self), first on the error path */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == self, 732);
    memset(fresh, 163, OBJ);
    mark(13);
    (void)read_probe(held);    /* get_state_ctx(self), second */
  } else if (which == 14) {
    /* gh-146011 -- _decimal's signal dict. traps->flags is a borrowed pointer
     * INTO the context object's own storage. context_clear released the
     * context while the signal dict, which can outlive it, kept the interior
     * pointer -- and signaldict_repr reads it whenever it is next called.
     *
     * The gap is unbounded here. Every other case's stale access happens
     * within the operation that created it, or at worst on the next API call;
     * this one waits for an unrelated repr() that may never come. */
    unsigned char *context = pym_malloc(OBJ);
    unsigned char *signaldict = pym_malloc(OBJ);
    CHECK(context && signaldict, 733);
    memset(context, 167, OBJ);
    signaldict[0] = 173;
    held = context + 16;       /* traps->flags, into the context's storage */
    pym_free(context);         /* context_clear, without clearing traps->flags */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == context && signaldict[0] == 173, 734);
    memset(fresh, 179, OBJ);
    mark(14);
    (void)read_probe(held);    /* signaldict_repr, arbitrarily later */
  } else if (which == 15) {
    /* gh-149449 -- unicodedata's capsule. _PyUnicode_Name_CAPI was a raw
     * PyMem_Malloc block owned by a capsule; other code cached the pointer,
     * and when unicodedata left sys.modules the capsule's destructor freed it
     * under them. Upstream's fix was to make the struct static.
     *
     * The freed thing is not a PyObject at all -- it is a bare allocation, and
     * it is pymalloc's because obmalloc sets PYMEM_DOMAIN_MEM to
     * PYMALLOC_ALLOC, so PyMem_Malloc reaches the same pools that
     * PyObject_Malloc does. That is the fact this case exists to exercise. */
    unsigned char *capi = pym_malloc(32); /* the _PyUnicode_Name_CAPI block */
    CHECK(capi, 735);
    memset(capi, 191, 32);
    held = capi;
    pym_free(capi);            /* the capsule's destructor, at module teardown */
    unsigned char *fresh = pym_malloc(32);
    CHECK(fresh == capi, 736);
    fresh[0] = 193;
    mark(15);
    (void)read_probe(held);    /* the cached capi->getname, called later */
  } else if (which == 16) {
    /* gh-151403 -- subprocess fork_exec. borrowed_arg comes from fast_args and
     * is handed to PyUnicode_FSConverter, whose __fspath__ can mutate args and
     * drop the sequence's last reference to it. */
    void **fast_args = pym_malloc(OBJ);
    unsigned char *arg = pym_malloc(OBJ);
    CHECK(fast_args && arg, 737);
    arg[0] = 197;
    fast_args[0] = arg;        /* PySequence_Fast_GET_ITEM, borrowed */
    pym_free(arg);             /* __fspath__ mutated args */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == arg, 738);
    fresh[0] = 199;
    mark(16);
    (void)read_probe(fast_args[0]);
  } else if (which == 17) {
    /* gh-151416 -- os.spawnv/spawnve. The same __fspath__ trigger as case 16,
     * one module over, reached through a getitem function pointer rather than
     * a fast-sequence macro. Kept separate because it was reported and fixed
     * separately, months apart, which is the corpus's point about how narrowly
     * each of these gets patched. */
    void **argv = pym_malloc(OBJ);
    unsigned char *item = pym_malloc(OBJ);
    CHECK(argv && item, 739);
    item[0] = 211;
    argv[0] = item;            /* (*getitem)(argv, i), borrowed */
    pym_free(item);            /* __fspath__ mutated the list */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == item, 740);
    fresh[0] = 223;
    mark(17);
    (void)read_probe(argv[0]);
  } else if (which == 18) {
    /* gh-151695 -- the curses screen encoding. A MODULE-LEVEL static pointed
     * into the encoding string owned by the window object initscr() returned.
     * The window is an ordinary object and can be deallocated while
     * module-level functions -- unctrl(), ungetch() -- keep reading through
     * the static.
     *
     * The dangling pointer outlives every frame here. Case 14's lived in
     * another object; this one lives in a global, so nothing in the program's
     * structure bounds when it is next used. */
    unsigned char *window = pym_malloc(OBJ);
    CHECK(window, 741);
    memset(window, 227, OBJ);
    held = window + 32;        /* curses_screen_encoding, into ->encoding */
    pym_free(window);          /* the window object was deallocated */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == window, 742);
    memset(fresh, 229, OBJ);
    mark(18);
    (void)read_probe(held);    /* unctrl(), through the module-level static */
  } else if (which == 19) {
    /* gh-153539 -- TextIOWrapper.tell() with a re-entrant decoder. next_input
     * is the snapshot bytes object, borrowed; the decoder's getstate can run
     * Python that seeks the file and replaces the snapshot, dropping the last
     * reference while tell() is still measuring against it.
     *
     * SIZE. The snapshot is a bytes object holding buffered input, so a large
     * buffer puts it above pymalloc's 512-byte threshold and back within a
     * malloc-level tool's reach -- the same caveat as case 6 and case 10. The
     * driver pins the small size. */
    unsigned char *next_input = pym_malloc(OBJ);
    CHECK(next_input, 743);
    memset(next_input, 233, OBJ);
    held = next_input;         /* the borrowed snapshot */
    pym_free(next_input);      /* the re-entrant decoder seeked */
    unsigned char *fresh = pym_malloc(OBJ);
    CHECK(fresh == next_input, 744);
    memset(fresh, 239, OBJ);
    mark(19);
    (void)read_probe(held);    /* cookie.start_pos -= PyBytes_GET_SIZE(...) */
  } else {
    pym_fail(718);
  }
  (void)write_probe; /* the label must exist even where no case writes */
}

void pym_replay(const struct pym_header *input, struct pym_header *out,
                void *scratch) {
  (void)scratch;
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->mode = mode;
  out->magic = PYM_MAGIC;
  out->count = 1;
  const struct pym_event *e = (const void *)(input + 1);
  unsigned which = e->id;
  CHECK(input->magic == PYM_MAGIC && input->count == 1 && which < 20, 700);
  defect(which);
  /* Only the spatial arm is expected to arrive here. */
  out->completed = 1;
  pym_backing_stats(out);
}
