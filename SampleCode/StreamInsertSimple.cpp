/**
 * @file    StreamInsert.cpp
 * @author  Wei Hong <wei@mapd.com>
 * @brief   Sample MapD Client code for inserting a stream of rows from stdin
 * to a MapD table.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <string>
#include <iostream>
#include <boost/tokenizer.hpp>

// include files for Thrift and MapD Thrift Services
#include "gen-cpp/MapD.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

// Thrift uses boost::shared_ptr instead of std::shared_ptr
using boost::shared_ptr;

namespace {
// anonymous namespace for private functions
const size_t INSERT_BATCH_SIZE = 10000;

// reads tab-delimited rows from std::cin and load them to
// table_name in batches of size INSERT_BATCH_SIZE until done
void stream_insert(MapDClient& client,
                   const TSessionId session,
                   const std::string& table_name,
                   const TTableDescriptor& table_desc,
                   const char* delimiter) {
  std::string line;
  std::vector<TStringRow> input_rows;
  TStringRow row;
  boost::char_separator<char> sep{delimiter, "", boost::keep_empty_tokens};
  while (std::getline(std::cin, line)) {
    {
      // free previous row's memory
      std::vector<TStringValue> empty;
      row.cols.swap(empty);
    }
    boost::tokenizer<boost::char_separator<char>> tok{line, sep};
    for (const auto& s : tok) {
      TStringValue ts;
      ts.str_val = s;
      ts.is_null = s.empty();
      row.cols.push_back(ts);
    }
    if (row.cols.size() != table_desc.size()) {
      std::cerr << "Incorrect number of columns: (" << row.cols.size() << " vs " << table_desc.size() << ") " << line
                << std::endl;
      continue;
    }
    input_rows.push_back(row);
    if (input_rows.size() >= INSERT_BATCH_SIZE) {
      try {
        client.load_table(session, table_name, input_rows);
      } catch (TMapDException& e) {
        std::cerr << e.error_msg << std::endl;
      }
      {
        // free rowset that has already been loaded
        std::vector<TStringRow> empty;
        input_rows.swap(empty);
      }
    }
  }
  // load remaining rowset if any
  if (input_rows.size() > 0)
    client.load_table(session, table_name, input_rows);
}
}

int main(int argc, char** argv) {
  std::string server_host("localhost");  // default to localohost
  int port = 9091;                       // default port number
  const char* delimiter = "\t";          // only support tab delimiter for now

  if (argc < 5) {
    std::cout << "Usage: <table> <database> <user> <password> [hostname[:port]]" << std::endl;
    return 1;
  }
  std::string table_name(argv[1]);
  std::string db_name(argv[2]);
  std::string user_name(argv[3]);
  std::string passwd(argv[4]);

  if (argc >= 6) {
    char* host = strtok(argv[5], ":");
    char* portno = strtok(NULL, ":");
    server_host = host;
    if (portno != NULL)
      port = atoi(portno);
  }

  shared_ptr<TTransport> socket(new TSocket(server_host, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  TSessionId session;
  try {
    transport->open();                                     // open transport
    session = client.connect(user_name, passwd, db_name);  // connect to mapd_server
    TTableDescriptor table_desc;
    client.get_table_descriptor(table_desc, session, table_name);
    stream_insert(client, session, table_name, table_desc, delimiter);
    client.disconnect(session);  // disconnect from mapd_server
    transport->close();          // close transport
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
    return 1;
  } catch (TException& te) {
    std::cerr << "Thrift error: " << te.what() << std::endl;
    return 1;
  }

  return 0;
}
