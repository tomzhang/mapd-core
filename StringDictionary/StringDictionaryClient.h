#ifndef STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
#define STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H

#include "gen-cpp/RemoteStringDictionary.h"
#include "../LeafHostInfo.h"

#include <memory>
#include <mutex>

class StringDictionaryClient {
 public:
  StringDictionaryClient(const LeafHostInfo& server_host, const int dict_id);

  int32_t get(const std::string& str);

  void get_string(std::string& _return, const int32_t string_id);

 private:
  void setupClient();

  const LeafHostInfo server_host_;
  const int dict_id_;
  std::unique_ptr<RemoteStringDictionaryClient> client_;
  std::mutex client_mutex_;
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
