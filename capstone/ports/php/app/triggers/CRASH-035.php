function my_errorhandler($errno,$errormsg) { global $my_var; $my_var = 0; }
set_error_handler('my_errorhandler');
$my_var = str_repeat('A', $my_var[0]->errormsg = 'xyz');
echo 'ok';