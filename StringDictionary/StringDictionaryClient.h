#ifndef STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
#define STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H

#include "gen-cpp/RemoteStringDictionary.h"
#include "LruCache.hpp"
#include "../LeafHostInfo.h"

#include <memory>
#include <mutex>

class StringDictionaryClient {
 public:
  StringDictionaryClient(const LeafHostInfo& server_host, const int dict_id, const bool with_timeout);

  void create(const int32_t dict_id, const int32_t db_id);

  int32_t get(const std::string& str);

  void get_string(std::string& _return, const int32_t string_id);

  int64_t storage_entry_count();

  std::vector<std::string> get_like(const std::string& pattern,
                                    const bool icase,
                                    const bool is_simple,
                                    const char escape,
                                    const int64_t generation);

  std::vector<std::string> get_regexp_like(const std::string& pattern, const char escape, const int64_t generation);

  void get_or_add_bulk(std::vector<int32_t>& string_ids, const std::vector<std::string>& strings);

  void translate_string_ids(std::vector<int32_t>& dest_ids,
                            const int32_t dest_dict_id,
                            const std::vector<int32_t>& source_ids,
                            const int32_t source_dict_id,
                            const int32_t dest_generation);

  bool checkpoint();

 private:
  void setupClient();

  const LeafHostInfo server_host_;
  LruCache<int32_t, std::string> id_to_string_cache_;
  const int dict_id_;
  const bool with_timeout_;
  std::unique_ptr<RemoteStringDictionaryClient> client_;
  std::mutex client_mutex_;
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
