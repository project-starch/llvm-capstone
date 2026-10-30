<?php
mb_regex_encoding('UTF-8');
// backref \0 = whole match becomes the eval'd code, per match
// first match "1" evals OK (v=string, then zval_dtor frees v),
// second match "@" parse-errors -> eval FAILS -> v left dangling IS_STRING
var_dump(mb_ereg_replace('.', '\0', '1@@@@@@@@', 'e'));
