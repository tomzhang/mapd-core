#include <iostream>
#include <string>
#include <exception>
#include "boost/program_options.hpp"
#include <boost/filesystem.hpp>
#include "Catalog/Catalog.h"

int main(int argc, char* argv[]) {
  std::string base_path;
  bool force = false;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "path", po::value<std::string>(&base_path)->required(), "Directory path to Mapd catalogs")(
      "force,f", "Force overwriting of existing MapD instance");

  po::positional_options_description positionalOptions;
  positionalOptions.add("path", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    if (vm.count("help")) {
      std::cout << "Usage: initdb [-f] <catalog path>\n";
      return 0;
    }
    if (vm.count("force"))
      force = true;
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (!boost::filesystem::exists(base_path)) {
    std::cerr << "Catalog basepath " + base_path + " does not exist.\n";
    return 1;
  }
  std::string catalogs_path = base_path + "/mapd_catalogs";
  if (boost::filesystem::exists(catalogs_path)) {
    if (force)
      boost::filesystem::remove_all(catalogs_path);
    else {
      std::cerr << "MapD catalogs already initialized at " + base_path + ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  std::string data_path = base_path + "/mapd_data";
  if (boost::filesystem::exists(data_path)) {
    if (force)
      boost::filesystem::remove_all(data_path);
    else {
      std::cerr << "MapD data directory already exists at " + base_path + ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  std::string export_path = base_path + "/mapd_export";
  if (boost::filesystem::exists(export_path)) {
    if (force)
      boost::filesystem::remove_all(export_path);
    else {
      std::cerr << "MapD export directory already exists at " + base_path + ". Use -f to force reinitialization.\n";
      return 1;
    }
  }
  if (!boost::filesystem::create_directory(catalogs_path)) {
    std::cerr << "Cannot create mapd_catalogs subdirectory under " << base_path << std::endl;
  }
  if (!boost::filesystem::create_directory(export_path)) {
    std::cerr << "Cannot create mapd_export subdirectory under " << base_path << std::endl;
  }

  try {
    auto dummy = std::make_shared<Data_Namespace::DataMgr>(data_path, false);
    Catalog_Namespace::SysCatalog sys_cat(base_path, dummy, true);
    sys_cat.initDB();
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 0;
}
