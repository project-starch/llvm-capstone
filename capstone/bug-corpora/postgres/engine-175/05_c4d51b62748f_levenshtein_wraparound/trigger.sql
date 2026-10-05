-- An edit distance is a count and cannot be negative; 17.5 returns
-- -1474836480 here. No sanitizer fires because the only observable of this
-- overflow is the answer itself.
-- EXPECT-ABSENT: levenshtein = "-
CREATE EXTENSION fuzzystrmatch;
SELECT levenshtein('GUMBO', 'GAMBOL', 2000000000, 2000000000, 2000000000);
