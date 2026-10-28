<?php
// Minimized trigger for spprintf.c:518 heap-use-after-free (ZE2, php-5.0.0).
// Root cause: Zend/zend_constants.c zend_register_constant() case-sensitive path.
// A duplicate define() makes zend_hash_add() FAIL; the error path does
//   free(c->name);                                  // line 320
//   zend_error(E_NOTICE,"Constant %s already defined", name);  // line 324
// For a case-sensitive constant name == c->name, so the E_NOTICE formats a
// %s pointer that was just free()'d -> strlen(freed) at spprintf.c:518.
define("C", 1);
define("C", 2);
