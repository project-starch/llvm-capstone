-- EXPECT-ERRORS: 2
-- EXPECT-ABSENT: array = "[0-9]
select array[]::oidvector;
select array[]::int2vector;
