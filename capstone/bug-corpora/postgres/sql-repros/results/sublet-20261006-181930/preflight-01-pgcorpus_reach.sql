CREATE EXTENSION pgcorpus_reach;
SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension WHERE extname = 'pgcorpus_reach';
