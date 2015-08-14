//==================================================================================================
///  @file QSUtilities.h
///
///  Definition of the class QSUtilities.
///
///  Copyright (C) 2009-2015 Simba Technologies Incorporated.
//==================================================================================================

#ifndef _SIMBA_QUICKSTART_QSUTILITIES_H_
#define _SIMBA_QUICKSTART_QSUTILITIES_H_

#include "gen-cpp/MapD.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "Quickstart.h"

#include <memory>
#include <vector>

class MapDClientHandle {
 public:
  MapDClientHandle(const int server_port) {
    socket_.reset(new apache::thrift::transport::TSocket("localhost", server_port));
    transport_.reset(new apache::thrift::transport::TBufferedTransport(socket_));
    protocol_.reset(new apache::thrift::protocol::TBinaryProtocol(transport_));
    client_.reset(new MapDClient(protocol_));
    transport_->open();
  }

  MapDClient* operator->() const { return client_.get(); }

  void reopen() { transport_->open(); }

 private:
  boost::shared_ptr<apache::thrift::transport::TSocket> socket_;
  boost::shared_ptr<apache::thrift::transport::TTransport> transport_;
  boost::shared_ptr<apache::thrift::protocol::TProtocol> protocol_;
  std::unique_ptr<MapDClient> client_;
};

namespace Simba {
namespace Quickstart {
/// @brief Quickstart platform specific utility functions.
class QSUtilities {
  // Public ======================================================================================
 public:
  /// @brief Constructor.
  ///
  /// @param in_settings              The connection settings to use. (NOT OWN)
  QSUtilities(QuickstartSettings* in_settings);

  /// @brief Destructor.
  ~QSUtilities();

  /// @brief Utility function to determine if a data file representing the table exists or
  /// not.
  ///
  /// @param in_tableName             The name of the table to check.
  ///
  /// @return True if the table exists; false otherwise.
  bool DoesTableExist(const simba_wstring& in_tableName) const;

  /// @brief Gets the table names in the DBF.
  /// @param out_tables               Vector of table names.
  void GetTables(std::vector<simba_wstring>& out_tables) const;

  // Private =====================================================================================
 private:
  // Struct containing connection settings and error codes. (NOT OWN)
  QuickstartSettings* m_settings;
};
}
}

#endif
