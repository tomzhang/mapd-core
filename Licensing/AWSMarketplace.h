#ifndef LICENSING_AWSMARKETPLACE_H
#define LICENSING_AWSMARKETPLACE_H

#ifdef HAVE_LICENSING_AWS
bool validate_server();
#else
bool validate_server() {
  return true;
};
#endif

#endif  // LICENSING_AWSMARKETPLACE_H
