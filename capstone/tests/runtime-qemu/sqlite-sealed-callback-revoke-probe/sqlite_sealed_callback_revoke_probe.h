#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_SEALED_CALLBACK_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_SEALED_CALLBACK_REVOKE_PROBE_H

/* Stage-2 "after" for the SEALED-CALLBACK shape (cve-repros rows 1/2/6/16:
 * cpython progress-handler UAF, rusqlite hook-closure UAF, php UDF UAF, datasette
 * authorizer-context UAF). A host registers a callback whose context pointer
 * (SQLite's `pApp`) is later freed/replaced while the engine still holds the
 * registration; a subsequent engine invocation of the callback dereferences the
 * stale context -- a use-after-free.
 *
 * Mapping onto the Capstone monitor-mediated model (the #70 borrow/revoke path,
 * validated by sqlite-borrow-revoke-probe), using the fact that a domain is itself
 * a SEALED capability (constructed with __seal in the monitor's create_dom, entered
 * via __domcallsaves): so `call_dom` == invoking a sealed callback entry.
 *
 *   host binding (lender / .user)  == owns the callback context (pApp), registers
 *                                     it, drives invocations, and unregisters.
 *   engine       (callee  / .smode) == on each sealed invocation runs the callback
 *                                     body, which stashed pApp at registration and
 *                                     reads it (the E->H context borrow).
 *   register callback              == shared_region_annotated(PERM_IN, REV_BORROWED)
 *   unregister / replace / close   == revoke_region(): set_authorizer(db,NULL,NULL),
 *                                     sqlite3_create_function replacement, handler
 *                                     removal, or connection close frees pApp.
 *
 * Flow: the host lends the callback context as a REV_BORROWED region ("register");
 * round 1 the engine invokes the sealed callback, which caches pApp and reads it
 * (real context value); the host "unregisters" (revoke); round 2 the engine invokes
 * the callback again and re-reads the cached pApp = the use-after-free. With
 * revocation enforced, the cached capability reloads untagged and the read faults;
 * the monitor cleanly terminates the domain and the host observes the fault
 * sentinel instead of a stale context value. Safe-fail: the callback UAF becomes a
 * deterministic trap.
 *
 * FEASIBILITY intent: this probe answers whether the SEALED-CALLBACK shape composes
 * from EXISTING ops (borrow/revoke on the context + the already-sealed domain entry)
 * or needs a new sealed-callback monitor op. TRAP in round 2 == it composes.
 */

#define SQLITE_SEALED_CB_REGION_SIZE 4096UL

/* The callback context (pApp) the host registers. Unmistakable 64-bit magic,
 * distinct from the borrow-revoke column value, the round-1 ack, and the fault
 * sentinel. */
#define SQLITE_SEALED_CB_CONTEXT_VALUE 0xCA11BAC0CA11BAC0UL

/* Round-1 handshake return (kept distinct; unused as a data value). */
#define SQLITE_SEALED_CB_RET_ROUND1 0x101UL

/* The monitor's clean-fault sentinel: call_dom() returns this when the domain
 * faulted on the use-after-revoke access and was terminated via
 * fault_return_from_domain() (#70). Observing it in round 2 == the callback UAF
 * trapped. */
#define SQLITE_SEALED_CB_FAULT_SENTINEL 0x0FA017EDUL

#endif
