/* Minimal stub: PHP main/streams pulls the socket headers in through php.h. A domain has
 * no network stack, so nothing declared here is ever called -- the declarations exist so
 * php.h parses. See sys/stat.h for the same rationale. */
#ifndef _NETINET_IN_H
#define _NETINET_IN_H 1
#include <sys/socket.h>
typedef unsigned short in_port_t;
typedef unsigned int   in_addr_t;
struct in_addr  { in_addr_t s_addr; };
struct in6_addr { unsigned char s6_addr[16]; };
struct sockaddr_in  { sa_family_t sin_family; in_port_t sin_port;
                      struct in_addr sin_addr; char sin_zero[8]; };
struct sockaddr_in6 { sa_family_t sin6_family; in_port_t sin6_port; unsigned int sin6_flowinfo;
                      struct in6_addr sin6_addr; unsigned int sin6_scope_id; };
#define INADDR_ANY       ((in_addr_t)0x00000000)
#define INADDR_NONE      ((in_addr_t)0xffffffff)
#define INADDR_LOOPBACK  ((in_addr_t)0x7f000001)
#define IPPROTO_IP   0
#define IPPROTO_TCP  6
#define IPPROTO_UDP 17
#endif
