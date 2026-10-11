-- complain if the script is sourced in psql rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgcorpus_lifetime" to load this file. \quit

CREATE FUNCTION lifetime_control(name text)
RETURNS text
AS 'MODULE_PATHNAME', 'lifetime_control'
LANGUAGE C STRICT;
