/*
 *  Some cool MapD License
 */

/*
 * File:   LdapServer.h
 * Author: michael
 *
 * Created on January 26, 2016, 11:50 PM
 */

#ifndef LDAPSERVER_H
#define LDAPSERVER_H

#include <string>
#include <glog/logging.h>
#include <ldap.h>

/*
 * @type LdapMetadata
 * @brief ldap data for using ldap server for authentication
 */
struct LdapMetadata {
  LdapMetadata(const std::string& uri, const std::string& dn) : uri(uri), distinguishedName(dn) {}
  LdapMetadata() {}
  int32_t port;
  std::string uri;
  std::string distinguishedName;
  std::string domainComp;
};

class LdapServer {
 public:
  LdapServer();
  LdapServer(const LdapMetadata& ldapMetadata);
  bool authenticate_user(const std::string& userName, const std::string& passwd);
  bool inUse();

 private:
  LdapMetadata ldapMetadata_;
  bool ldapInUse;
};

#endif /* LDAPSERVER_H */
