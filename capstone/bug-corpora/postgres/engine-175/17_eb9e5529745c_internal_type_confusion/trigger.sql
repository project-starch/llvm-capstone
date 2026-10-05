-- EXPECT-ERRORS: 3
select numeric_avg_accum(null, 1::numeric);
select array_agg_transfn(null, 1);
select int8_avg_accum(null, 1::int8);
