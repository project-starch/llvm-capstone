-- NEGATIVE CONTROL for trigger.sql.
--
-- The defect is the EMPTY array: casting array[] to oidvector reads off the
-- end of a zero-element allocation. A non-empty array takes the same cast
-- through the same code with something to read, so these two statements must
-- COMPLETE.
--
-- Note the trigger's own oracle is a differential, not a fault, on two of the
-- three arms -- but sublet does report a fault, and that fault needs the same
-- qualification as any other.
select array[1,2]::oidvector;
select array[1,2]::int2vector;
