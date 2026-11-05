<?php
class Person {
	public $dad;
	public function __destruct() {
		$this->dad = null;
	}
}
class Dad extends Person {
	public $son;
	public function __construct() {
		$this->son = new Person;
		$this->son->dad = $this;
	}
	public function __destruct() {
		$this->son = null;
		parent::__destruct();
	}
}
$o = new Dad;
unset($o);
echo "ok\n";
?>
