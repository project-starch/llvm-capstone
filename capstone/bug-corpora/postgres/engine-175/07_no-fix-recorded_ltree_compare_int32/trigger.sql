-- Reachability probe for an arm with no sanitizer.
-- l.v is 'a.a.....a' with 15000 levels and s.v is the single level 'a'. ltree
-- ordering compares level by level and then by length, so a path that begins
-- with the whole of the shorter one and continues is strictly GREATER: the only
-- correct answer is gt_ok=t, lt_ok=f. 17.5 returns lt_ok=t because the length
-- difference overflows int32 in ltree_compare. lt_ok="t" is therefore a value a
-- correct build cannot print, and seeing it proves the overflow was executed.
-- EXPECT-ABSENT: lt_ok = "t"
CREATE EXTENSION ltree;
WITH s AS (SELECT 'a'::ltree AS v), l AS (SELECT (repeat('a.', 14999) || 'a')::ltree AS v) SELECT (l.v > s.v) AS gt_ok, (l.v < s.v) AS lt_ok, (l.v = s.v) AS eq_ok FROM s, l;
