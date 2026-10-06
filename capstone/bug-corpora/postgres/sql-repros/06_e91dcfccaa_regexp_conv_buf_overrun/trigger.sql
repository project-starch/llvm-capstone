-- regexp conversion-buffer overrun (upstream e91dcfccaa), live at the 17.5 pin.
--
-- Requires a multibyte database encoding so that eml > 1 and regexp.c takes the
-- conv_buf path at all. convert_from(..., 'SQL_ASCII') is the standard way to
-- put bytes into a text value without the encoding check that a literal gets:
-- 0xBF is not a legal UTF-8 lead byte, so pg_mb2wchar_with_len (regexp.c:1444)
-- turns each one into its own pg_wchar, and re-encoding them costs more bytes
-- than they occupied on the way in.
SELECT getdatabaseencoding();

SELECT regexp_split_to_array(
           convert_from(repeat('\xbf'::bytea, 4096), 'SQL_ASCII'),
           'x');

SELECT regexp_match(
           convert_from(repeat('\xbf'::bytea, 4096), 'SQL_ASCII'),
           '(.*)');
