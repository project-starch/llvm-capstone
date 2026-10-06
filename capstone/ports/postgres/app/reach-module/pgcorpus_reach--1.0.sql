-- complain if the script is sourced in psql rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgcorpus_reach" to load this file. \quit

CREATE FUNCTION corpus_ascii_truncated_lead(payload integer DEFAULT 1)
RETURNS integer
AS 'MODULE_PATHNAME', 'corpus_ascii_truncated_lead'
LANGUAGE C STRICT;

CREATE FUNCTION corpus_regexp_invalid_subject(payload integer DEFAULT 64)
RETURNS text[]
AS 'MODULE_PATHNAME', 'corpus_regexp_invalid_subject'
LANGUAGE C STRICT;

CREATE FUNCTION corpus_pgp_sesskey_overflow(msglen integer DEFAULT 64)
RETURNS integer
AS 'MODULE_PATHNAME', 'corpus_pgp_sesskey_overflow'
LANGUAGE C STRICT;
