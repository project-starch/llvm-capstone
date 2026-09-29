-- One single-user session's work: DDL, 2000 rows, an index, a scan, a join,
-- an update, a delete, a vacuum and a count. The native survey greps the
-- final count = "1500" to know the session ran to its end.
CREATE TABLE t (id int PRIMARY KEY, name text, v float8);
INSERT INTO t SELECT i, 'name-' || i, i * 1.5 FROM generate_series(1, 2000) i;
CREATE INDEX ON t (name);
SELECT count(*), sum(v) FROM t WHERE name LIKE 'name-1%';
SELECT a.id, b.name FROM t a JOIN t b ON a.id = b.id + 1 WHERE a.id < 5 ORDER BY a.id;
UPDATE t SET v = v * 2 WHERE id % 7 = 0;
DELETE FROM t WHERE id > 1500;
VACUUM t;
SELECT count(*) FROM t;
