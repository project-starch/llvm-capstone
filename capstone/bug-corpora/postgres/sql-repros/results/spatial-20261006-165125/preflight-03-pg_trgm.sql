CREATE EXTENSION pg_trgm;
SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension WHERE extname = 'pg_trgm';
