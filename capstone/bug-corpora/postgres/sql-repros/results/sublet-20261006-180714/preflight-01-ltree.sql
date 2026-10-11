CREATE EXTENSION ltree;
SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension WHERE extname = 'ltree';
