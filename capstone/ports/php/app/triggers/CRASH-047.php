function compareByRef(&$first, &$second) { return $first === $second; }
class MyTree implements ArrayAccess {
  public $parent;
  public $children = array();
  public function offsetExists($offset) {}
  public function offsetUnset($offset) {}
  public function offsetSet($offset, $value) {
    $cannonicalName = strtolower($offset);
    $this->children[$cannonicalName] = $value;
    $value->parent = $this; }
  public function offsetGet($offset) {
    $cannonicalName = strtolower($offset);
    return $this->children[$cannonicalName]; }
}
$id = 'Test';
$root = new MyTree();
$child = new MyTree();
$root[$id] = $child;
var_dump(compareByRef($root[$id], $child));