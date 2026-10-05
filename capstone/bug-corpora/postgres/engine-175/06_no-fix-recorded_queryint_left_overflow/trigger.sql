-- Upstream added the overflow check this cast is missing, so the cast raises
-- there. The column name states the assertion: 17.5 answers t, meaning no
-- error was raised.
-- EXPECT-ERRORS: 1
CREATE EXTENSION intarray;
SELECT (SELECT '0 | ' || string_agg(i::text, ' & ') FROM generate_series(1, 17000) AS i)::query_int IS NOT NULL AS no_error_raised;
