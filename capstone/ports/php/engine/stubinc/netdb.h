/* Minimal stub: PHP main/streams pulls the socket headers in through php.h. A domain has
 * no network stack, so nothing declared here is ever called -- the declarations exist so
 * php.h parses. See sys/stat.h for the same rationale. */
#ifndef _NETDB_H
#define _NETDB_H 1
#include <sys/socket.h>
struct hostent { char *h_name; char **h_aliases; int h_addrtype; int h_length; char **h_addr_list; };
struct servent { char *s_name; char **s_aliases; int s_port; char *s_proto; };
struct addrinfo { int ai_flags, ai_family, ai_socktype, ai_protocol;
                  socklen_t ai_addrlen; struct sockaddr *ai_addr;
                  char *ai_canonname; struct addrinfo *ai_next; };
struct hostent *gethostbyname(const char *);
struct servent *getservbyname(const char *, const char *);
int  getaddrinfo(const char *, const char *, const struct addrinfo *, struct addrinfo **);
void freeaddrinfo(struct addrinfo *);
#endif
