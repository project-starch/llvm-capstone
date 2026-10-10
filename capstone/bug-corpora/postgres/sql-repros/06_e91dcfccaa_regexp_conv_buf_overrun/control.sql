-- NEGATIVE CONTROL for trigger.sql.
--
-- The defect needs bytes that are INVALID in the database encoding:
-- pg_mb2wchar_with_len turns each into one pg_wchar and pg_wchar2mb_with_len
-- spends two bytes putting it back, so n bytes in become 2n out against a
-- buffer of n + 1. A subject of the same length made of VALID characters
-- re-encodes to exactly what came in, so nothing overruns and the same call
-- must COMPLETE.
--
-- 0x41 is 'A'. Same length, same allocation, same regexp, same function --
-- only the validity of the bytes changes, which is the whole condition.
CREATE EXTENSION pgcorpus_reach;
SELECT corpus_regexp_invalid_subject(64, 65);
