/* Eight CPython pymalloc defects, as domain programs.
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
 * bounds per allocation and would still miss all eight, because a reused block
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
 * FOUR OF THE EIGHT SHARE ONE ALLOCATOR SHAPE, AND THAT IS A FINDING
 *
 * Cases 0, 1, 3 and 4 come from four separately reported defects in three
 * modules, and all four reduce to: free a small object, allocate the same size
 * again, read through the pointer that was kept. They are kept apart rather
 * than merged because they are independent upstream reports, and because the
 * sameness is the point -- one revocation mechanism covers a defect class that
 * upstream has had to fix one module at a time.
 *
 * SIZE. Every block here is well under pymalloc's 512-byte threshold, so all of
 * it is pool memory that never reaches malloc. Case 6 is the one whose upstream
 * defect can exceed that: it carries a payload buffer, and on a large input the
 * same defect becomes an ordinary malloc use-after-free that ASan does see.
 * That is stated in its PROVENANCE.md and is why the size is pinned here.
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
  CHECK(input->magic == PYM_MAGIC && input->count == 1 && which < 8, 700);
  defect(which);
  /* Only the spatial arm is expected to arrive here. */
  out->completed = 1;
  pym_backing_stats(out);
}
