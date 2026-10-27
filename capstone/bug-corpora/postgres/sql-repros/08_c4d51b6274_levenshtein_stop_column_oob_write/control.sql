-- Control for case 08: the same write instruction, at a valid index.
--
-- levenshtein.c:240 `prev[stop_column] = max_d + 1` is reached here too -- the
-- costs are small enough that nothing overflows, so stop_column comes out 6
-- and the write is in bounds. The string lengths are the trigger's second
-- rung's, 20 and 100, so the allocation traffic is identical and the only
-- difference is the costs.
--
-- A fault here would mean the arm faults on this pattern rather than on the
-- defect, and the case's verdict would be void.
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- stop_column = 6, in bounds: reaches the write, does not cross the chunk.
SELECT levenshtein_less_equal(repeat('a',20), repeat('x',100), 1, 1, 1, 90);
