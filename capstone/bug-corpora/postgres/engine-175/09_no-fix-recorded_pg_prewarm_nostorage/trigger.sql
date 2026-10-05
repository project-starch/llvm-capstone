CREATE EXTENSION pg_prewarm;
CREATE TABLE pw_t (c1 int) PARTITION BY RANGE (c1);
SELECT pg_prewarm('pw_t', 'buffer');
