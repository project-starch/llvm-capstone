-- EXPECT-ERRORS: 1
create table pt(a int);
insert into pt select generate_series(1,100);
prepare p1 as select * from pt;
declare c1 cursor for select * from pt;
close c1;
declare c1 cursor for execute p1;
fetch 1 from c1;
close c1;
