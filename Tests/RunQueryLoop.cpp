#include "QueryRunner.h"

#include <boost/program_options.hpp>

int main(int argc, char** argv) {
  std::string db_path;
  std::string query;
  size_t iter;

  ExecutorDeviceType device_type{ExecutorDeviceType::GPU};

  boost::program_options::options_description desc("Options");
  desc.add_options()(
      "path", boost::program_options::value<std::string>(&db_path)->required(), "Directory path to Mapd catalogs")(
      "query", boost::program_options::value<std::string>(&query)->required(), "Query")(
      "iter", boost::program_options::value<size_t>(&iter), "Number of iterations")(
      "cpu", "Run on CPU (run on GPU by default)");

  boost::program_options::positional_options_description positionalOptions;
  positionalOptions.add("path", 1);
  positionalOptions.add("query", 1);

  boost::program_options::variables_map vm;

  try {
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    boost::program_options::notify(vm);
  } catch (boost::program_options::error& err) {
    LOG(ERROR) << err.what();
    return 1;
  }

  if (!vm.count("iter")) {
    iter = 100;
  }

  if (vm.count("cpu")) {
    device_type = ExecutorDeviceType::CPU;
  }

  std::unique_ptr<Catalog_Namespace::SessionInfo> session(get_session(db_path.c_str()));
  for (size_t i = 0; i < iter; ++i) {
    run_multiple_agg(query, session, device_type);
  }
  return 0;
}
