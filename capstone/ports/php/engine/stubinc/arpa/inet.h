/* Minimal stub: PHP main/streams pulls the socket headers in through php.h. A domain has
 * no network stack, so nothing declared here is ever called -- the declarations exist so
 * php.h parses. See sys/stat.h for the same rationale. */
#ifndef _ARPA_INET_H
#define _ARPA_INET_H 1
#include <netinet/in.h>
unsigned int   htonl(unsigned int);
unsigned short htons(unsigned short);
unsigned int   ntohl(unsigned int);
unsigned short ntohs(unsigned short);
in_addr_t inet_addr(const char *);
char *inet_ntoa(struct in_addr);
int inet_pton(int, const char *, void *);
const char *inet_ntop(int, const void *, char *, socklen_t);
#endif
