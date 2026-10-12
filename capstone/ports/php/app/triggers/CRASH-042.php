class main { public static $dataAccessor = NULL; }
class relay { public function __get($name) { main::$dataAccessor = new relay; } }
main::$dataAccessor = new relay();
main::$dataAccessor->bar;
echo 'ok';