<?php
// bug #29199: ent_uni_punct[] has 66 entries but entity_map declares
// basechar=8194..endchar=8260 (67). The all-entities decode loop walks
// k=8194..8260 and reads entity_map[j].table[k-basechar]; at k=8260 it
// reads table[66] -> OOB read of the global entity_map table array.
// Any UTF-8 html_entity_decode with all-entities triggers the full walk.
var_dump(html_entity_decode('&frasl;', ENT_QUOTES, 'UTF-8'));
?>
