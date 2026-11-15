-- NEGATIVE CONTROL for trigger.sql: the same statements, with FETCH BACKWARD 1000 in place of
-- FETCH BACKWARD ALL. The thousand rows stay in memory under the same 1MB temp_file_limit, so
-- nothing spills, nothing is freed by the spill, and the ROLLBACK's tuplestore_end frees each tuple
-- once. A fault here does not depend on the defect.
begin;
declare cur1 cursor for select c, c c2 from generate_series(0, 1000000) x(c) order by c;
move all in cur1;
set temp_file_limit TO '1MB';
fetch backward 1000 from cur1;
rollback;
select 1 as survived;
