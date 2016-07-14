/*
 *  Some cool MapD License
 */

/*
 * File:   LdapServer.cpp
 * Author: michael
 *
 * Created on January 26, 2016, 11:50 PM
 */

#include "LdapServer.h"

LdapServer::LdapServer() {
  LOG(INFO) << "No LDAP server defined, will not attempt to authenticate via ldap";
  ldapInUse = false;
}

LdapServer::LdapServer(const LdapMetadata& ldapMetadata) {
  if (ldapMetadata.uri.empty()) {
    ldapInUse = false;
  } else {
    LOG(INFO) << "LDAP being used for Authentication, uri: " << ldapMetadata.uri << " ou: " << ldapMetadata.orgUnit;
    ldapInUse = true;
    ldapMetadata_ = ldapMetadata;
  }
}

bool LdapServer::inUse() {
  return ldapInUse;
}

bool LdapServer::authenticate_user(const std::string& userName, const std::string& passwd) {
#ifdef HAVE_LDAP
  LDAP* ldp;
  int rc, version;
  berval creds;
  int maxLength = userName.length() + ldapMetadata_.orgUnit.length() + 10;
  char bind_dn[maxLength];

  snprintf(bind_dn, maxLength, "cn=%s,%s", userName.c_str(), ldapMetadata_.orgUnit.c_str());
  LOG(INFO) << "User " << userName << " connecting as " << bind_dn;

  /* Open LDAP Connection */
  /* Get a handle to an LDAP connection. */
  rc = ldap_initialize(&ldp, ldapMetadata_.uri.c_str());
  if (rc != LDAP_SUCCESS) {
    LOG(ERROR) << "ldap_initialize failed " << ldap_err2string(rc);
    return false;
  }
  version = LDAP_VERSION3;

  ldap_set_option(ldp, LDAP_OPT_PROTOCOL_VERSION, &version);
  /* User authentication (bind) */
  std::vector<char> writable_passwd(passwd.begin(), passwd.end());
  writable_passwd.push_back('\0');
  creds.bv_val = &writable_passwd[0];
  creds.bv_len = passwd.length();
  rc = ldap_sasl_bind_s(ldp, bind_dn, NULL, &creds, NULL, NULL, NULL);
  if (rc != LDAP_SUCCESS) {
    LOG(ERROR) << "LDAP ldap_sasl_bind_s FAILURE: " << ldap_err2string(rc);
    return false;
  }
  LOG(INFO) << " User " << userName << " successfully logged in with LDAP authentication";
  ldap_unbind_ext_s(ldp, NULL, NULL);
  return true;
#else
  CHECK(false);
  return false;
#endif
}
