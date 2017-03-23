#include "StringDictionaryClient.h"

#include <boost/make_shared.hpp>
#include <glog/logging.h>
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

using apache::thrift::protocol::TBinaryProtocol;
using apache::thrift::transport::TSocket;
using apache::thrift::transport::TTransport;
using apache::thrift::transport::TBufferedTransport;
using apache::thrift::TException;
using apache::thrift::transport::TTransportException;

StringDictionaryClient::StringDictionaryClient(const LeafHostInfo& server_host, const int dict_id)
    : server_host_(server_host), dict_id_(dict_id) {
  setupClient();
}

int32_t StringDictionaryClient::get(const std::string& str) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  try {
    return client_->get(str, dict_id_);
  } catch (const TTransportException&) {
    setupClient();
  }
  return client_->get(str, dict_id_);
}

void StringDictionaryClient::get_string(std::string& _return, const int32_t string_id) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  try {
    client_->get_string(_return, string_id, dict_id_);
    return;
  } catch (const TTransportException&) {
    setupClient();
  }
  client_->get_string(_return, string_id, dict_id_);
}

int64_t StringDictionaryClient::storage_entry_count() {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  try {
    return client_->storage_entry_count(dict_id_);
  } catch (const TTransportException&) {
    setupClient();
  }
  return client_->storage_entry_count(dict_id_);
}

std::vector<std::string> StringDictionaryClient::get_like(const std::string& pattern,
                                                          const bool icase,
                                                          const bool is_simple,
                                                          const char escape,
                                                          const int64_t generation) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  std::string escape_str(1, escape);
  std::vector<std::string> _return;
  try {
    client_->get_like(_return, pattern, icase, is_simple, escape_str, generation, dict_id_);
    return _return;
  } catch (const TTransportException&) {
    setupClient();
  }
  client_->get_like(_return, pattern, icase, is_simple, escape_str, generation, dict_id_);
  return _return;
}

std::vector<std::string> StringDictionaryClient::get_regexp_like(const std::string& pattern,
                                                                 const char escape,
                                                                 const int64_t generation) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  std::string escape_str(1, escape);
  std::vector<std::string> _return;
  try {
    client_->get_regexp_like(_return, pattern, escape_str, generation, dict_id_);
    return _return;
  } catch (const TTransportException&) {
    setupClient();
  }
  client_->get_regexp_like(_return, pattern, escape_str, generation, dict_id_);
  return _return;
}

void StringDictionaryClient::get_or_add_bulk(std::vector<int32_t>& string_ids,
                                             const std::vector<std::string>& strings) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  string_ids.resize(strings.size());
  try {
    client_->get_or_add_bulk(string_ids, strings, dict_id_);
    return;
  } catch (const TTransportException&) {
    setupClient();
  }
  client_->get_or_add_bulk(string_ids, strings, dict_id_);
}

bool StringDictionaryClient::checkpoint() {
  std::lock_guard<std::mutex> lock(client_mutex_);
  CHECK(client_);
  try {
    return client_->checkpoint(dict_id_);
  } catch (const TTransportException&) {
    setupClient();
  }
  return client_->checkpoint(dict_id_);
}

void StringDictionaryClient::setupClient() {
  const auto socket = boost::make_shared<TSocket>(server_host_.getHost(), server_host_.getPort());
  socket->setConnTimeout(5000);
  socket->setRecvTimeout(10000);
  socket->setSendTimeout(10000);
  const auto transport = boost::make_shared<TBufferedTransport>(socket);
  transport->open();
  const auto protocol = boost::make_shared<TBinaryProtocol>(transport);
  client_.reset(new RemoteStringDictionaryClient(protocol));
}
