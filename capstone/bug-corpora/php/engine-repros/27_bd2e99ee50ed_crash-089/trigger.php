<?php
// bug #28067: ent_uni_338_402 UTF-8 table is under-sized vs endchar=402
// html_entity_decode(all=1) iterates k=338..402 -> table[k-338] up to idx 64,
// but the table has only 63 entries -> OOB global read.
echo html_entity_decode("x", ENT_QUOTES, "UTF-8");
?>
