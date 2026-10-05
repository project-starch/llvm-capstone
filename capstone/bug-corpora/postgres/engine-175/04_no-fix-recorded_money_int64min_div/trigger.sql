-- Reachability probe for an arm with no sanitizer.
-- '-92233720368547758.08'::money is exactly INT64_MIN cents. Dividing it by -1
-- has no representable answer, so a correct build must raise an error; what it
-- can never do is return a NEGATIVE value, because negating a negative is
-- positive. On RISC-V the division does not trap (x86 raises SIGFPE, which is
-- how the host oracle saw it) and cash_div_int64 returns INT64_MIN unchanged.
-- So the original negative appearing in the output IS the defect executing.
-- EXPECT-ABSENT: -\$92,233,720,368,547,758
SELECT '-92233720368547758.08'::money / -1::int8;
