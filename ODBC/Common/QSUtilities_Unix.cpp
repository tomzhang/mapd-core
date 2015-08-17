//==================================================================================================
///  @file QSUtilities_Linux.cpp
///
///  Implementation of the class QSUtilities for Windows.
///
///  Copyright (C) 2008-2015 Simba Technologies Incorporated.
//==================================================================================================

#include "QSUtilities.h"

using namespace Simba::Quickstart;
using namespace std;

// Public ==========================================================================================
////////////////////////////////////////////////////////////////////////////////////////////////////
QSUtilities::QSUtilities(QuickstartSettings* in_settings) : m_settings(in_settings) {
  ;  // Do nothing.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
QSUtilities::~QSUtilities() {
  ;  // Do nothing.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSUtilities::DoesTableExist(const simba_wstring& in_tableName) const {
  MapDClientHandle client(9091);
  auto session = client->connect("mapd", "HyperInteractive", "mapd");
  std::vector<std::string> available_tables;
  client->get_tables(available_tables, session);
  for (const auto& table_name : available_tables) {
    if (simba_wstring(table_name) == in_tableName) {
      return true;
    }
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void QSUtilities::GetTables(std::vector<simba_wstring>& out_tables) const {
  MapDClientHandle client(9091);
  auto session = client->connect("mapd", "HyperInteractive", "mapd");
  std::vector<std::string> thrift_tables;
  client->get_tables(thrift_tables, session);
  for (const auto& table_name : thrift_tables) {
    out_tables.emplace_back(table_name);
  }
}
