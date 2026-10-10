-- NEGATIVE CONTROL for trigger.sql.
--
-- The defect is a timezone abbreviation longer than the buffer to_char writes
-- it into. The trigger uses 60 characters and overflows by about 36, so the
-- buffer holds roughly two dozen; eight characters is well inside it and the
-- same two statements must then COMPLETE.
--
-- It exists because this arm reports a fault here and a fault is only this
-- case's result if it depends on the defect. If the control faults too, the
-- arm is reacting to the statement rather than to the overflow, and this row
-- must be withdrawn as case 03's was.
SET TIME ZONE '<ABCDEFGH>+0';
SELECT to_char(now(), 'TZ');
