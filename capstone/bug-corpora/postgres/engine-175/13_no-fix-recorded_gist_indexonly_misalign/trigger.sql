-- Reachability probe, derived from the 2026-10-04 purecap run.
-- EXPLAIN confirms `Index Only Scan using gist_ios_idx`, so the index-only path is
-- taken; the select then fails with `type with OID 860 does not exist`. No correct
-- build emits a nonexistent type OID for this query -- that is what comes out of a
-- misaligned tuple descriptor, which is the defect. The host (x86) returned the
-- correct answer for the same SQL, so alignment is the variable, as the entry says.
-- NOT yet a confirmed reproduction: meta.toml records no fix commit, so there is no
-- fixed build to diff against. The directive asserts reachability, nothing more.
-- EXPECT-ABSENT: type with OID [0-9]+ does not exist
create temp table gist_ios_tupdesc (a inet, r numrange);
insert into gist_ios_tupdesc values ('::1', numrange(repeat('7', 200)::numeric, repeat('8', 200)::numeric));
create index gist_ios_idx on gist_ios_tupdesc using gist (a inet_ops, r);
vacuum analyze gist_ios_tupdesc;
set enable_seqscan = off;
set enable_bitmapscan = off;
explain (costs off) select lower(r) = repeat('7', 200)::numeric as lower_ok, upper(r) = repeat('8', 200)::numeric as upper_ok from gist_ios_tupdesc where r && numrange(null, null);
select lower(r) = repeat('7', 200)::numeric as lower_ok, upper(r) = repeat('8', 200)::numeric as upper_ok from gist_ios_tupdesc where r && numrange(null, null);
