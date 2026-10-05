-- 3704870b29 / af8f9248fb: postgres_fdw 异步请求. 需要 postmaster: 测试要两个指回
-- 自身的 foreign server, 单用户模式下没有监听者.
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE SERVER loopback FOREIGN DATA WRAPPER postgres_fdw OPTIONS (dbname 'postgres', host '/tmp', port '5432');
CREATE SERVER loopback2 FOREIGN DATA WRAPPER postgres_fdw OPTIONS (dbname 'postgres', host '/tmp', port '5432');
CREATE USER MAPPING FOR CURRENT_USER SERVER loopback;
CREATE USER MAPPING FOR CURRENT_USER SERVER loopback2;
ALTER SERVER loopback OPTIONS (ADD async_capable 'true');
ALTER SERVER loopback2 OPTIONS (ADD async_capable 'true');
CREATE TABLE async_pt (a int, b int, c text) PARTITION BY RANGE (a);
CREATE TABLE base_tbl1 (a int, b int, c text);
CREATE TABLE base_tbl2 (a int, b int, c text);
CREATE TABLE base_tbl3 (a int, b int, c text);
CREATE FOREIGN TABLE async_p1 PARTITION OF async_pt FOR VALUES FROM (1000) TO (2000) SERVER loopback OPTIONS (table_name 'base_tbl1');
CREATE FOREIGN TABLE async_p2 PARTITION OF async_pt FOR VALUES FROM (2000) TO (3000) SERVER loopback2 OPTIONS (table_name 'base_tbl2');
CREATE FOREIGN TABLE async_p3 PARTITION OF async_pt FOR VALUES FROM (3000) TO (4000) SERVER loopback2 OPTIONS (table_name 'base_tbl3');
INSERT INTO async_p1 SELECT 1000 + i, i, to_char(i, 'FM0000') FROM generate_series(0, 999, 5) i;
INSERT INTO async_p2 SELECT 2000 + i, i, to_char(i, 'FM0000') FROM generate_series(0, 999, 5) i;
INSERT INTO async_p3 SELECT 3000 + i, i, to_char(i, 'FM0000') FROM generate_series(0, 999, 5) i;
ANALYZE async_pt;
-- 这一条就是修复提交自己新增的触发语句
SELECT o.x FROM (VALUES (2505), (3505)) o(x), LATERAL (SELECT a FROM async_pt WHERE a = o.x OR a = 1505 LIMIT 1) s ORDER BY o.x;
