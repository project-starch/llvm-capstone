CREATE EXTENSION fuzzystrmatch;
SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension WHERE extname = 'fuzzystrmatch';
