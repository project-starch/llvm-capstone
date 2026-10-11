CREATE EXTENSION pgcrypto;
SELECT 'PREFLIGHT-OK-' || extname AS marker FROM pg_extension WHERE extname = 'pgcrypto';
