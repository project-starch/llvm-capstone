-- levenshtein_less_equal out-of-bounds write (upstream c4d51b6274), live at
-- the 17.5 pin.
--
-- THE PARAMETERS CHANGED ON 2026-10-18, because the previous ones never
-- reached the write. They were ins_c=715827883 with max_d=2000000000 and
-- 2147483647, and they do overflow -- they return -715827814, a negative
-- Levenshtein distance, which is the "nonsensical results" half of upstream's
-- advisory. But levenshtein.c:159 then takes
--
--     if (max_d >= max_theo_d) max_d = -1;
--
-- because max_theo_d came out negative too, and that branch leaves
-- stop_column at its default m + 1 (:138). The write at :240 is guarded by
-- `if (stop_column < m)` with m already incremented, so m + 1 < m is false and
-- the write never executes. The three MISSED verdicts recorded for this case
-- before today were therefore not measurements of any mechanism.
--
-- The arithmetic, all in int32 (PostgreSQL builds -fwrapv):
--
--   :152  min_theo_d  = net_inserts * ins_c           overflows negative
--   :159  if (max_d >= max_theo_d) max_d = -1         MUST NOT be taken, so
--                                                     sub_c * Min(m,n) has to
--                                                     overflow positive too
--   :175  slack_d     = max_d - min_theo_d            overflows negative
--   :178  stop_column = best_column + slack_d / (ins_c + del_c) + 1
--   :179  if (stop_column > m) ...                    a ONE-SIDED clamp
--   :240  prev[stop_column] = max_d + 1               the write
--
-- prev is palloc(2 * (m+1) * sizeof(int)), so the write lands stop_column * 4
-- bytes from the chunk. Two rungs, because the distance is caller-controlled
-- and one point does not show that (PROVENANCE.md asks for more than one cost
-- triple):
--
--   stop_column = -1    -4 bytes    upstream's own case, inside the chunk header
--   stop_column = -16  -64 bytes    past the header, out of the chunk
--
-- Solved from the source above, not from any arm's output.
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- stop_column = -1: prev[-1], four bytes in front of the chunk.
SELECT levenshtein_less_equal(repeat('a',3), repeat('x',200), 100000000, 715827882, 1073741823, 536870912);

-- stop_column = -16: prev[-16], sixty-four bytes in front of the chunk.
SELECT levenshtein_less_equal(repeat('a',20), repeat('x',100), 100000000, 20000000, 200000000, 1610612736);
