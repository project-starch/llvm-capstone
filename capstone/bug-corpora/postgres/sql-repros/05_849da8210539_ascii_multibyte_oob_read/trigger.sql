-- EXPECT-ERRORS: 4
select ascii(convert_from('\xc3'::bytea, 'SQL_ASCII'));
select ascii(convert_from('\xe282'::bytea, 'SQL_ASCII'));
select ascii(convert_from('\xf09f98'::bytea, 'SQL_ASCII'));
select ascii(convert_from('\xf8'::bytea, 'SQL_ASCII'));
