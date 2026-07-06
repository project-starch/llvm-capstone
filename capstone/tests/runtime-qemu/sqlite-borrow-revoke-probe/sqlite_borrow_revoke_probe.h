#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_BORROW_REVOKE_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_SQLITE_BORROW_REVOKE_PROBE_H

/* Stage-2 "after" for cve-repros/row3_diesel_colname_cached (diesel
 * RUSTSEC-2021-0037): a SQLite column pointer cached across sqlite3_step,
 * dereferenced after the row it belonged to has advanced -- a use-after-free.
 *
 * Mapped onto the Capstone monitor-mediated borrow/revoke model (the working
 * #70 path validated by run-revoke-matrix-probe):
 *
 *   engine  (lender / .user)   == the SQLite engine that owns the row buffer and
 *                                 hands out sqlite3_column_text() pointers.
 *   host    (borrower / .smode) == the language binding that reads the column and
 *                                 (buggily) caches the borrowed pointer.
 *   step                        == revoke_region(): sqlite3_step advances the row,
 *                                 ending the borrow / freeing the old row buffer.
 *
 * Flow: the engine lends the current row buffer as a REV_BORROWED region; round 1
 * the host reads the column while the borrow is live and caches the pointer;
 * the engine "steps" (revokes); round 2 the host re-reads its CACHED pointer =
 * the diesel use-after-free. With revocation enforced, the cached capability
 * reloads untagged and the read faults; the monitor cleanly terminates the domain
 * and the engine observes the fault sentinel instead of a stale column value.
 * Safe-fail: the use-after-free becomes a deterministic trap.
 */

#define SQLITE_BORROW_REGION_SIZE 4096UL

/* The "column text" the engine writes into the current row buffer. Chosen to be
 * an unmistakable 64-bit magic (not the round-1 ack, not the fault sentinel). */
#define SQLITE_BORROW_COLUMN_VALUE 0xC01A0DEDC01A0DEDUL

/* Round-1 handshake return (kept distinct; unused as a data value). */
#define SQLITE_BORROW_RET_ROUND1 0x101UL

/* The monitor's clean-fault sentinel: call_dom() returns this when the domain
 * faulted on the use-after-revoke access and was terminated via
 * fault_return_from_domain() (#70). Observing it in round 2 == the UAF trapped. */
#define SQLITE_BORROW_FAULT_SENTINEL 0x0FA017EDUL

#endif
