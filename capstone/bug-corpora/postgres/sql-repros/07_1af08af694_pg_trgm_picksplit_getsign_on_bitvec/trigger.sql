-- pg_trgm picksplit reads past a signature (upstream 1af08af694, CVE-2026-14678),
-- live at the 17.5 pin.
--
-- The defective branch at trgm_gist.c:898-901 needs a page split in which one
-- SEED datum is all-true while the entry being compared is not, so the index
-- has to be large enough to split many times and must hold both kinds of row.
--
-- Row length matters and was measured on a native 17.5 cluster: at 120
-- md5-chunks per row the index row exceeds the 8191-byte limit and CREATE
-- INDEX fails before picksplit ever runs. 60 chunks fits.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE trgm_split (t text);

-- Saturating rows: enough distinct trigrams that the signature goes all-true.
INSERT INTO trgm_split
SELECT string_agg(md5(random()::text), ' ')
FROM generate_series(1, 60) g, generate_series(1, 3000) s
GROUP BY s;

-- Sparse rows: these keep a non-all-true signature, which is the cache[j] side
-- the defective branch reads.
INSERT INTO trgm_split
SELECT 'ab' || i FROM generate_series(1, 3000) i;

-- 6000 rows builds an index of about 3700 pages, so gtrgm_picksplit runs many
-- times over a mixture of both signature kinds.
CREATE INDEX trgm_split_idx ON trgm_split USING gist (t gist_trgm_ops);

DROP TABLE trgm_split;
