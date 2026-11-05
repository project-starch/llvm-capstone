<?php
class Ks { var $r; function __destruct(){ $GLOBALS['gs']=&$this->r; } } $os=new Ks(); $os->r=&$sc; unset($os);
$sc = array(); $sc[0] = &$sc; unset($sc);
$sHTTP_RAW_POST_DATA = <<<EOF
xml version="1.0"
<env:Envelope xmlns:env="http://www.w3.org/2003/05/soap-envelope"
              xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <env:Body>
    <test:echoStringArray xmlns:test="http://example.org/ts-tests"
          env:encodingStyle="http://www.w3.org/2003/05/soap-encoding">
      <inputStringArray enc:itemType="xsd:string" 
                        xmlns:enc="http://www.w3.org/2003/05/soap-encoding">
        <item enc:id="data" xsi:type="xsd:string" enc:ref="#data">hello</item>
        <item>world</item>
      </inputStringArray>
    </test:echoStringArray>
  </env:Body>
</env:Envelope>
EOF;
include "soap12-test.inc";
class par {
	private $sid = "foo";

	function displayMe()
	{
		print $this->id;
	}
};

class chld extends par {
	public $sid = "bar";
	function displayHim()
	{
		parent::displayMe();
	}
};


$sobj = new chld();
$sobj->displayHim();
function global_func()
{
	echo __METHOD__ . "\n";
}

$sfunction = 'global_func';
$sfunction();

class foo
{
	static $smethod = 'global_func';
	
	static public function foo_func()
	{
		echo __METHOD__ . "\n";
	}
}

/* The following is a BC break with PHP 4 where it would 
 * call foo::fail. In PHP 5 we first evaluate static class 
 * properties and then do the function call.
 */
$smethod = 'foo_func';
foo::$smethod();
