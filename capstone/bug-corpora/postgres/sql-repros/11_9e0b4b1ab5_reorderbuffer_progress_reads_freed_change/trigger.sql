-- SERVER-SETTING: wal_level=logical
-- EXPECT-ERRORS: 0
-- Upstream fix 9e0b4b1ab5: ReorderBufferProcessTXN updates the transaction's progress from
-- change->lsn after every CHANGES_THRESHOLD (100) changes, and an INSERT ON CONFLICT's speculative
-- insert is freed while its confirmation is processed. Upstream forced it by recompiling the threshold
-- to 1; here the transaction is sized instead: 98 rows, then the speculative insert (change 99) and
-- its confirmation (change 100), so the progress update reads the LSN of the change just freed.
CREATE TABLE t(id int primary key, v int);
SELECT * FROM pg_create_logical_replication_slot('s', 'test_decoding');
BEGIN;
INSERT INTO t SELECT g, 0 FROM generate_series(1, 98) g;
INSERT INTO t VALUES (100000, 0) ON CONFLICT (id) DO UPDATE SET v = 1;
COMMIT;
SELECT count(*) FROM pg_logical_slot_get_changes('s', NULL, NULL);
SELECT 1 AS survived;
