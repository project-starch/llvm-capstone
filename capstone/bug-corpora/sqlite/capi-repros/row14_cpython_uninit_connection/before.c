#include <stdlib.h>
#include "sqlite3.h"

struct connection {
  sqlite3 *db;
};

int main(void) {
  struct connection *connection = calloc(1, sizeof(*connection));
  volatile unsigned char byte;

  byte = *(unsigned char *)connection->db;
  return byte;
}
