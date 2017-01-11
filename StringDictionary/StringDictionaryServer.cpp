#include "StringDictionary.h"
#include "gen-cpp/RemoteStringDictionary.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/regex.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/smart_ptr/make_unique.hpp>
#include <glog/logging.h>
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>

#include <string>
#include <unordered_map>

namespace {

class RemoteStringDictionary : virtual public RemoteStringDictionaryIf {
 public:
  RemoteStringDictionary(const std::string& base_path) : base_path_(base_path) {
    boost::regex dict_folder_expr{R"(DB_(\d+)_DICT_(\d+))", boost::regex::extended};
    for (auto& dict_folder : boost::make_iterator_range(boost::filesystem::directory_iterator(base_path_))) {
      boost::smatch what;
      const auto dict_path = boost::filesystem::canonical(dict_folder.path());
      const auto dict_path_leaf = dict_path.filename();
      if (boost::regex_match(dict_path_leaf.string(), what, dict_folder_expr)) {
        CHECK(boost::filesystem::is_directory(dict_path));
        createDictionaryFromPath(std::stoi(what[2].str()), dict_path);
      }
    }
  }

  void create(const int32_t dict_id, const int32_t db_id) noexcept override {
    auto dict_path = base_path_ / ("DB_" + std::to_string(db_id) + "_DICT_" + std::to_string(dict_id));
    CHECK(boost::filesystem::create_directory(dict_path));
    createDictionaryFromPath(dict_id, dict_path);
  }

  int32_t get(const std::string& str, const int32_t dict_id) noexcept override {
    return getStringDictionary(dict_id)->get(str);
  }

  void get_string(std::string& _return, const int32_t string_id, const int32_t dict_id) noexcept override {
    _return = getStringDictionary(dict_id)->getString(string_id);
  }

 private:
  StringDictionary* getStringDictionary(const int32_t dict_id) const {
    mapd_shared_lock<mapd_shared_mutex> read_lock(string_dictionaries_mutex_);
    const auto it = string_dictionaries_.find(dict_id);
    CHECK(it != string_dictionaries_.end());
    return it->second.get();
  }

  void createDictionaryFromPath(const int32_t dict_id, const boost::filesystem::path& dict_path) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(string_dictionaries_mutex_);
    const auto it_ok = string_dictionaries_.emplace(dict_id, boost::make_unique<StringDictionary>(dict_path.string()));
    CHECK(it_ok.second);
  }

  boost::filesystem::path base_path_;
  std::unordered_map<int32_t, std::unique_ptr<StringDictionary>> string_dictionaries_;
  mutable mapd_shared_mutex string_dictionaries_mutex_;
};

}  // namespace

int main(int argc, char** argv) {
  int port = 10301;
  std::string base_path;

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("path", po::value<std::string>(&base_path), "Base path for dictionary storage");
  desc.add_options()("port", po::value<int>(&port)->default_value(port), "Port number");
  po::variables_map vm;

  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  auto handler = boost::make_shared<RemoteStringDictionary>(base_path);

  auto processor = boost::make_shared<RemoteStringDictionaryProcessor>(handler);

  using namespace ::apache::thrift;
  using namespace ::apache::thrift::concurrency;
  using namespace ::apache::thrift::protocol;
  using namespace ::apache::thrift::server;
  using namespace ::apache::thrift::transport;

  boost::shared_ptr<ThreadManager> thread_manager = ThreadManager::newSimpleThreadManager(8);
  thread_manager->threadFactory(boost::make_shared<PlatformThreadFactory>());
  thread_manager->start();

  auto server_socket = boost::make_shared<TServerSocket>(port);
  auto transport_factory = boost::make_shared<TBufferedTransportFactory>();
  auto protocol_factory = boost::make_shared<TBinaryProtocolFactory>();
  TThreadPoolServer server(processor, server_socket, transport_factory, protocol_factory, thread_manager);

  server.serve();

  return 0;
}
