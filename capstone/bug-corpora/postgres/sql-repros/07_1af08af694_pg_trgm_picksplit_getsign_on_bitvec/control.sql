-- ONE STATEMENT PER LINE: `postgres --single` takes a line at a time.
-- Control for case 07: the same index build, with the defective branch never
-- entered.
--
-- The defect at trgm_gist.c:900 and :913 sits inside
--
--     if (ISALLTRUE(datum_l) || cache[j].allistrue)
--
-- and reads GETSIGN(cache[j].sign), a pointer that is already a BITVECP, so
-- TRGMHDRSIZE is added to it a second time. Reaching it needs an all-true
-- signature in the split. This control supplies only sparse rows, so no
-- signature saturates, the branch is never taken, and the correct path at
-- :904 and :917 -- hemdistsign(cache[j].sign, ...), with no GETSIGN wrapping --
-- is what runs instead.
--
-- Everything else is the trigger's: the same extension, the same index build,
-- the same number of rows, so gtrgm_picksplit runs the same number of times
-- over the same cache_sign chunk. A fault here would mean the arm faults on
-- building a GiST trigram index at all, and the case's verdict would be void.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE trgm_split_control (t text);

-- Sparse rows only: no signature goes all-true, so cache[j].allistrue is never
-- true and the defective expression is never evaluated.
INSERT INTO trgm_split_control SELECT 'ab' || i FROM generate_series(1, 6000) i;

CREATE INDEX trgm_split_control_idx ON trgm_split_control USING gist (t gist_trgm_ops);

DROP TABLE trgm_split_control;
