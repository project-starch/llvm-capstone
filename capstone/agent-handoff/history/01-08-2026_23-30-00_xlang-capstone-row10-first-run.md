# xlang Capstone column — row 10 runs; revocation catches it, but as a QEMU abort

**2026-08-01.** First end-to-end run of the xlang Capstone column
(`plans/capstone-column-xlang.md`). Row 10 = CVE-2022-1106, the corpus's
template row: a VM-register-stack UAF where a cached interior pointer is written
through after the stack was reallocated and moved.

## The A/B

Identical program, identical allocator, identical free path. The *only*
difference is whether `rof_free` fires the revoke (`XLANG_NO_REVOKE` sets
`rof_no_revoke`, ALLOCATOR-CONTRACT §4).

| build | serial output |
|---|---|
| control (revoke suppressed) | `XLANG mruby_range_uaf use-after-free-survived` + `xlang-host: call retval = 0x8001a5ec` |
| revoke enabled | `helper_cscincoffsetimm: Assertion 'rs1_v->tag' failed` |

The control completing is what makes the fault interpretable: the stale write
lands and the domain returns normally when the revoke is suppressed, so the
failure in the other build is attributable to the revoke and nothing else.

## What this does and does not establish

**Establishes:** the revoke cleared the tag on every alias of the freed region,
including the one the shim cached before the callback. The stale write never
executed. By the corpus's taxonomy the defect is blocked, and blocked
*synchronously* — at the first use after the free, not by a later sweep.

**Does NOT establish: that the fault is DELIVERED.** The manifestation is a QEMU
assertion abort, not a monitor fault line with a cause code. The sqlite rows
classify on `Cap mem access on revoked capability` (cause 25); this row produces
no such line, so it cannot be classified the same way. Whether real hardware
traps cleanly, or wedges as in R-5, is untested.

**RESOLVED the same session — fault delivery IS verified.** See "The delivery
probe" below. The remaining gap is narrower than it first looked: what real
hardware does on the *arithmetic*, not whether the mechanism works.

## The delivery probe (added after the above)

The row dies on `cscincoffsetimm`, where QEMU has a bare `assert(rs1_v->tag)`
and **no exception path at all** — `op_helper.c` has 13 such asserts against 46
real `riscv_raise_exception` calls. So the arithmetic route cannot answer "is a
fault delivered"; it only shows the emulator has no model for that case.

`DELIVERY=1` builds a probe that allocates, revokes, and then **dereferences at
offset 0** — no arithmetic — using our allocator. Its own A/B:

| build | result |
|---|---|
| revoke enabled | `[CAPSTONE] Cap mem access requires capability: rs1 = x10, imm = 0`<br>`[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x1015a2720, badaddr = 0x10141f000` |
| revoke suppressed | `XLANG-DELIVERY about-to-deref` → `XLANG-DELIVERY NOTRAP deref returned` → domain returns |

**A properly delivered monitor fault, cause 24, attributable to the revoke.**
Not an assert, not an abort. So our allocator's revoke reaches the modelled
fault path, and the rows' `cincoffset` aborts are the emulator gap rather than a
dead mechanism.

### What that leaves genuinely open

Only *which* fault a row produces on real hardware. Either the arithmetic on an
untagged capability traps — the row is caught — or it yields an untagged result
and the subsequent store faults cause 24 exactly like this probe — the row is
caught. The only way a row escapes is if the arithmetic *restored* the tag,
which is not a thing capability hardware does. So the catch is robust across
both plausible semantics; only its manifestation is unsettled.

**A trap in the first run of the control was caught and fixed:** the delivery
probe block sat *before* the `XLANG_NO_REVOKE` setter and returned early, so the
"control" never actually suppressed the revoke and faulted identically. That
would have read as "the fault is not attributable to the revoke" — the exact
opposite of the truth. The setter now precedes every early-returning probe.

## Why it aborts on the arithmetic rather than the access

The shim does:

```c
mrb_value *regs = mrb->c->stack;     /* cached */
mrb_funcall_cb(mrb, deep_callback);  /* reallocs+moves; old region revoked */
unsigned char *stale = (unsigned char *)regs + ACCESS_OFF;   /* <-- aborts HERE */
*(volatile uint64_t *)stale = 0xdeadbeef;                    /* never reached */
```

The offset arithmetic is a `cscincoffsetimm` on a now-untagged capability, and
the emulator asserts on it. The sqlite rows dereference a revoked pointer
directly and get cause 25; this row does arithmetic *first*.

**This is an architectural difference between the two columns worth reporting,
not an artifact to tune away.** On CHERI, arithmetic on an untagged capability is
legal and the fault arrives on the dereference — which is exactly why the same
shim yields `SIGPROT` there. On Capstone the arithmetic itself is illegal. The
corpus surfaced this by accident and it is a real distinction between the
mechanisms.

**Do not "fix" the shim by hoisting the arithmetic above the callback.** It is
shared with the CHERI column and gated by `check_shim_fidelity.py`; changing it
to obtain a tidier Capstone result would invalidate the CHERI verdicts and would
be tuning the instrument to the answer.

## Relation to R-5

R-5 (`ISSUES.md`) is "illegal/meaningless capability ops wedge rather than trap",
evidenced by M-mode spinning on silicon. This is the QEMU-side sibling: the
emulator *asserts and exits* rather than modelling a trap. Same underlying gap —
an operation on a tagless capability has no defined fault behaviour — different
manifestation. Worth folding into R-5 rather than opening a new issue.

## Three bugs found and fixed getting here

Each would have produced a silently wrong result rather than an error:

1. **`revoke_on_free_alloc.h` is entirely `static`** — written for one
   translation unit (sqlite includes it in a single domain file). Including it
   in both the mock and the domain gave each TU its own `rof_arena`: the
   domain's `rof_init` filled the domain's copy while the mock's `rof_malloc`
   read its own still-null one, aborting in `helper_cslcc`. The allocator now
   has exactly one owner (`mock_mruby_capstone.c`), reached through
   `xlang_arena_init` / `xlang_set_no_revoke`.
2. **`struct sqlite_hostcall_v0` is `{phase, opcode, offset, length}`** — the
   length cursor is at offset 24. A locally re-declared one-field version wrote
   the cursor into `phase`, so the host would have read length 0, lost all
   output, and scored a returning row as a fault.
3. **`ROF_MAX_SLOTS` defaults to 16384** = 512 KiB of `.bss` for a domain making
   five allocations. Now 64.

Plus two toolchain constraints, worked around rather than fought: `<stdio.h>`
cannot be included freestanding at 16-byte pointers (glibc sizes
`struct _IO_FILE` padding as `12*sizeof(int) - 5*sizeof(void*)`, which goes
negative), and a ternary selecting between two string capabilities crashes the
backend with `Cannot select: i128 = CapstoneISD::SELECT_CC` (C-2 family) —
branch instead of selecting.

## Next

- Decide how to classify a row whose evidence is an abort rather than a fault
  line. This is a methodology question for the column, not a code question, and
  it affects all 14 rows.
- The other 13 rows are `ROW=<name> ./build-xlang-capstone.sh`; expect the same
  manifestation for the five other `vm_stack_uaf.h` rows (4, 5, 8, 13, 15),
  since they share the template and therefore the arithmetic-after-free shape.
- Row 2 (stack-use-after-return) involves no allocator and should MISS. Predict
  it in writing before running.
