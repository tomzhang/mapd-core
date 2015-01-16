#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "Catalog/Catalog.h"
#include "Parser/parser.h"
#include "Parser/ParserNode.h"
#include "Analyzer/Analyzer.h"
#include "Planner/Planner.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;

void
process_backslash_commands(const string &command, const Catalog &catalog)
{
	switch (command[1]) {
		/*
		case 'd':
			if (
		case 'l':
		*/
		case 'q':
			exit(0);
		default:
			throw runtime_error("Invalid backslash command.");
	}
}

int
main(int argc, char* argv[])
{
	string base_path;
	string db_name;
	string user_name;
	string passwd;
	bool debug = false;
	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<string>(&base_path)->required(), "Directory path to Mapd catalogs")
		("db", po::value<string>(&db_name), "Database name")
		("user,u", po::value<string>(&user_name)->required(), "User name")
		("passwd,p", po::value<string>(&passwd)->required(), "Password")
		("debug,d", "Verbose debug mode");

	po::positional_options_description positionalOptions;
	positionalOptions.add("path", 1);
	positionalOptions.add("db", 1);

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: mapd -u <user name> -p <password> <catalog path> [<database name>]\n";
			return 0;
		}
		if (vm.count("debug"))
			debug = true;
		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
		cerr << "Usage Error: " << e.what() << std::endl;
		return 1;
	}

	if (!vm.count("db"))
		db_name = MAPD_SYSTEM_DB;

	if (!boost::filesystem::exists(base_path)) {
		cerr << "Catalog path " + base_path + " does not exist.\n";
		return 1;
	}
	std::string system_db_file = base_path + "/mapd_catalogs/mapd";
	if (!boost::filesystem::exists(system_db_file)) {
		cerr << "MapD not initialized at " + base_path + "\nPlease run initdb first.\n";
		return 1;
	}

	SysCatalog sys_cat(base_path);
	UserMetadata user;
	if (!sys_cat.getMetadataForUser(user_name, user)) {
		cerr << "User " << user_name << " does not exist." << std::endl;
		return 1;
	}
	if (user.passwd != passwd) {
		cerr << "Invalid password for User " << user_name << std::endl;
		return 1;
	}
	DBMetadata db;
	if (!sys_cat.getMetadataForDB(db_name, db)) {
		cerr << "Database " << db_name << " does not exist." << std::endl;
		return 1;
	}
	if (!user.isSuper && user.userId != db.dbOwner) {
		cerr << "User " << user_name << " is not authorized to access database " << db_name << std::endl;
		return 1;
	}
	Catalog cat(base_path, user, db);
	while (true) {
		try {
			cout << "MapD > ";
			string input_str;
			getline(cin, input_str);
			if (cin.eof()) {
				cout << std::endl;
				break;
			}
			if (input_str[0] == '\\') {
				process_backslash_commands(input_str, cat);
				continue;
			}
			SQLParser parser;
			list<Parser::Stmt*> parse_trees;
			string last_parsed;
			int numErrors = parser.parse(input_str, parse_trees, last_parsed);
			if (numErrors > 0)
				throw runtime_error("Syntax error at: " + last_parsed);
			for (auto stmt : parse_trees) {
				unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
				Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
				if ( ddl != nullptr)
					ddl->execute(cat);
				else {
					Parser::DMLStmt *dml = dynamic_cast<Parser::DMLStmt*>(stmt);
					Query query;
					dml->analyze(cat, query);
					Optimizer optimizer(query, cat);
					RootPlan *plan = optimizer.optimize();
					unique_ptr<RootPlan> plan_ptr(plan); // make sure it's deleted
					if (debug) plan->print();
					// @TODO execute plan
				}
			}
		}
		catch (std::exception &e)
		{
			std::cerr << "Exception: " << e.what() << "\n";
		}
	}
	return 0;
}
