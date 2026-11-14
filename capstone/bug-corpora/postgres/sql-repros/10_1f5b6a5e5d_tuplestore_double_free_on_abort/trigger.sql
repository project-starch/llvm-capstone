-- EXPECT-ERRORS: 1
-- Upstream bug #19438's reproducer (fix 1f5b6a5e5d), one change: the report discards the forward
-- FETCH ALL's million rows with psql's \o /dev/null, which a single-user backend does not have, so
-- the cursor is moved to its end with MOVE ALL instead. The FETCH BACKWARD ALL's result spills to a
-- temporary file, temp_file_limit stops the spill part way (the one expected error), and the
-- tuplestore is left holding pointers to tuples the spill already pfree'd. The ROLLBACK drops the
-- portal, tuplestore_end pfrees them again.
begin;
declare cur1 cursor for select c, c c2 from generate_series(0, 1000000) x(c) order by c;
move all in cur1;
set temp_file_limit TO '1MB';
fetch backward all from cur1;
rollback;
select 1 as survived;
