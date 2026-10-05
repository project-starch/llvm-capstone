CREATE FUNCTION sqljson_mystr_in(cstring) RETURNS sqljson_mystr AS 'textin' LANGUAGE internal IMMUTABLE STRICT;
CREATE FUNCTION sqljson_mystr_out(sqljson_mystr) RETURNS cstring AS 'textout' LANGUAGE internal IMMUTABLE STRICT;
CREATE TYPE sqljson_mystr ( INPUT = sqljson_mystr_in, OUTPUT = sqljson_mystr_out, LIKE = text, CATEGORY = 'S' );
SELECT '{"a":1}'::sqljson_mystr IS JSON;
