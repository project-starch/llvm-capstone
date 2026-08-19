/* i128 is the capability carrier, so an ordinary ternary on __int128 is a
   select of two 128-bit values. Both arms constant is the shape JerryScript hits. */
unsigned __int128 g(int c) { return c ? (unsigned __int128) -4 : (unsigned __int128) 0; }
