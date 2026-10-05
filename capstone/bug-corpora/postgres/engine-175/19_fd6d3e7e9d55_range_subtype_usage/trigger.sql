-- EXPECT-ERRORS: 1
create role regress_subtype;
create type mytype as (a int, b int);
revoke usage on type mytype from public;
-- Without this the role cannot create anything and the run dies at
-- "permission denied for schema public" before reaching the defect.
grant create on schema public to regress_subtype;
set role regress_subtype;
create type myrange as range (subtype = mytype);
reset role;
drop type if exists myrange cascade;
drop type mytype cascade;
-- Revoke before dropping, or the drop fails with "privileges for schema
-- public" and that trailing error is counted as a rejection, masking the
-- differential: got=1 is then not < EXPECT-ERRORS=1 and the DIFF never fires.
revoke create on schema public from regress_subtype;
drop role regress_subtype;
