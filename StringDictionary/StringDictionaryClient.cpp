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

void StringDictionaryClient::setupClient() {
  const auto socket = boost::make_shared<TSocket>(server_host_.getHost(), server_host_.getPort());
  const auto transport = boost::make_shared<TBufferedTransport>(socket);
  transport->open();
  const auto protocol = boost::make_shared<TBinaryProtocol>(transport);
  client_.reset(new RemoteStringDictionaryClient(protocol));
}
