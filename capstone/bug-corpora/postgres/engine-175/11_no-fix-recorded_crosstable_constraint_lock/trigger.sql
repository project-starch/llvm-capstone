create table attbl(a int);
create table atref(b attbl check ((b).a is not null));
alter table attbl alter column a type numeric;
