-- levenshtein_less_equal out-of-bounds write (upstream c4d51b6274), live at
-- the 17.5 pin.
--
-- levenshtein.c:152 computes  min_theo_d = net_inserts * ins_c  as int * int.
-- net_inserts is (length(target) - length(source)); with ins_c near INT_MAX the
-- product overflows, slack_d (:175) inherits it, and stop_column (:178) comes
-- out of range. The clamp at :179-180 only catches stop_column > m, so a value
-- below range survives to :240, which writes prev[stop_column].
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- signature: levenshtein_less_equal(source, target, ins_c, del_c, sub_c, max_d)
-- A length difference of a few characters multiplied by a near-INT_MAX
-- insertion cost is enough to overflow the 32-bit product.
SELECT levenshtein_less_equal('abc', 'abcdefgh', 2000000000, 1, 1, 100);
SELECT levenshtein_less_equal('abcdefgh', 'abc', 1, 2000000000, 1, 100);
SELECT levenshtein_less_equal('a', 'abcdefghij', 1073741824, 1073741824, 1, 50);
