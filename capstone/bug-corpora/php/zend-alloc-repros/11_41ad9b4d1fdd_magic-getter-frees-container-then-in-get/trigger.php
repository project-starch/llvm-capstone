<?php
class main {
   public static $dataAccessor = NULL ;
}
class relay {
   public function __get( $name ) {
       main::$dataAccessor = new relay;   // frees the old relay whose __get is running
   }
}
main::$dataAccessor = new relay();
main::$dataAccessor->bar;
echo "ok\n";
