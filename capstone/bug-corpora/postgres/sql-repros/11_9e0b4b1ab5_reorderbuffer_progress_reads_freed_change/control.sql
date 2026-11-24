-- SERVER-SETTING: wal_level=logical
-- NEGATIVE CONTROL for trigger.sql: the same statements with 99 rows instead of 98. The hundredth
-- change is then the speculative insert itself, still held for its confirmation, so the progress
-- update reads a live change. A fault here does not depend on the defect.
CREATE TABLE t(id int primary key, v int);
SELECT * FROM pg_create_logical_replication_slot('s', 'test_decoding');
BEGIN;
INSERT INTO t SELECT g, 0 FROM generate_series(1, 99) g;
INSERT INTO t VALUES (100000, 0) ON CONFLICT (id) DO UPDATE SET v = 1;
COMMIT;
SELECT count(*) FROM pg_logical_slot_get_changes('s', NULL, NULL);
SELECT 1 AS survived;
