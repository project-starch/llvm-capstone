# Why the comparison uses a distilled shim, not the real library

Each Lua↔C cross-domain-pointer bug in this corpus is measured through a small
**shim** — a distilled C program that reproduces only the memory-lifecycle events
the bug depends on (`alloc → free → stale dereference`), not the real library.
This is a deliberate methodological choice, for three reasons — none of them
convenience.

## 1. Capstone cannot run the real thing

The real reproduction needs real Lua + the real native library (OpenSSL, SDL2,
libdbus, libpq, …) + an operating system with a sanitizer. A Capstone **domain**
is *freestanding*: no OS, no syscalls, no libc++. Those libraries exist precisely
*to talk to the OS* — random numbers, files, sockets — so with the OS removed they
do not run at all.

The shim keeps only the three memory events that decide the verdict, and those
**do** compile and run on `capstone64`. (SQLite is the one real engine that runs
on both sides, because it is self-contained C with an in-memory, OS-free mode.)

## 2. Fairness — the main reason

The **byte-identical** shim source runs on **both** platforms
(`shims/<case>.c`, compiled once for Capstone and once purecap for CHERI). That
is what makes the comparison fair: the workload is held constant and only the
capability/revocation mechanism varies.

The real library would run **only on CHERI** (which has a full OS, CheriBSD), so
using it would compare unlike things. The shim is the **common denominator** both
platforms can execute — and that is exactly why the measured result
(async 0/13, eager & Capstone 13/13) is a fair comparison rather than an artifact
of which platform can run more software.

## 3. Isolation of the measurement

A shim contains **only** the cross-domain-pointer event. So a capability fault is
unambiguously attributable to the mechanism under test — and the no-revoke
**control**, which completes (MISS), proves it. In the real library a crash could
have a dozen unrelated causes. This is why the numbers are clean:
**13/13 caught, with the control missing on every row.**

## Concrete example — lua-openssl #141

The real bug: a Lua userdata wraps an OpenSSL `EVP_CIPHER_CTX`; `close()` frees the
C context, and the userdata's `__gc` reads and frees it a second time. The shim is
~15 lines:

```c
ud.ctx = malloc(168);          // = EVP_CIPHER_CTX_new  (the 168-byte C object)
/* openssl_cipher_ctx_free, as cipher.c:551–552 */
sink = *(uint64_t*)ud.ctx;     // cipher.c:551 — the stale READ (the CDP event)
free(ud.ctx);                  // cipher.c:552 — free → on Capstone: REVOKE
/* … called once for close(), once for __gc … */
```

It reproduces the real `heap-use-after-free` natively under ASan (`READ of size 8`
at offset 0, matching upstream) — so the distillation is verified, not a strawman —
and it needs neither OpenSSL, nor a Lua interpreter, nor an OS.

## What the shim deliberately drops (and the limitation)

It drops everything the bug does **not** depend on: the encryption itself, the AES
key, the real allocator internals, the language runtime. Those never affect
"does revocation catch the stale pointer?", so leaving them out changes no answer.

The limitation this creates is honest and belongs in the write-up: because the
real libraries cannot run on Capstone, we compare distilled shims, not full
applications. That gap between the two platforms *is* the separate **compatibility**
axis — the shim measures **security** fairly; it does not claim Capstone can run
the real software.

---

## Project position on the three reasons (recorded on integration)

The three reasons above are the collaborator's. On integration into the main branch the
project lead accepted **reason 1 only**, and explicitly did **not** accept reasons 2 and 3.
Recording that here so the distinction survives into the write-up:

- **Reason 1 (Capstone cannot run the real thing) — ACCEPTED.** A Capstone domain is
  freestanding, and the real reproductions need an OS, syscalls and third-party native
  libraries. This is a genuine, checkable constraint and it is the *actual* reason the corpus
  is measured through shims.
- **Reasons 2 (fairness) and 3 (isolation) — NOT accepted as justifications.** They are real
  *properties* of the shim method, but presenting them as reasons to prefer a shim inverts
  cause and effect: the shim was forced by reason 1, and 2/3 are consequences we then benefit
  from. Framing a limitation as a design virtue is the kind of thing a reviewer discounts, and
  it weakens an otherwise strong result.
- **The preferred direction is full Lua on BOTH platforms.** That is what would make the
  comparison unarguable. It is blocked on dependency/porting work, not on principle, and it
  stays an open item rather than something the shim closes.

**Consequence for the write-up.** State the shim as a *limitation with a mitigation* — "the
real libraries cannot run on Capstone, so we measure distilled shims; the same source runs on
both sides, so the comparison remains controlled" — not as a methodological preference. The
paper-facing doc `capstone/agent-handoff/ref/xlang-security-measurements-for-paper.md` already
phrases it this way (both columns are upper bounds; the symmetry makes the *comparison* fair
"even though neither absolute is realistic"), and that is the phrasing to keep.

This changes no measurement. The numbers, the fidelity gate and the controls are unaffected.
