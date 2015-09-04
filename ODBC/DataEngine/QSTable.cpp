//==================================================================================================
///  @file QSTable.cpp
///
///  Implementation of the class QSTable.
///
///  Copyright (C) 2008-2014 Simba Technologies Incorporated.
//==================================================================================================

#include "QSUtilities.h"

#include "QSTable.h"

#include "DSIColumnMetadata.h"
#include "DSIResultSetColumn.h"
#include "DSIResultSetColumns.h"
#include "DSITypeUtilities.h"
#include "ErrorException.h"
#include "IColumn.h"
#include "IWarningListener.h"
#include "NumberConverter.h"
#include "SEInvalidArgumentException.h"
#include "SqlData.h"
#include "SqlDataTypeUtilities.h"
#include "SqlTypeMetadata.h"
#include "SqlTypeMetadataFactory.h"

#include <ctime>

using namespace Simba::Quickstart;
using namespace Simba::DSI;

// Public ==========================================================================================
////////////////////////////////////////////////////////////////////////////////////////////////////
QSTable::QSTable(QuickstartSettings* in_settings,
                 ILogger* in_log,
                 const simba_wstring& in_tableName,
                 IWarningListener* in_warningListener,
                 bool in_isODBCV3)
    : m_log(in_log),
      m_tableName(in_tableName),
      m_settings(in_settings),
      m_warningListener(in_warningListener),
      m_rowIdx(0) {
  SE_CHK_INVALID_ARG((NULL == in_settings) || (NULL == in_log));

  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "QSTable");

  InitializeColumns(in_isODBCV3);

  MapDClientHandle client(9091);
  auto session = client->connect("mapd", "HyperInteractive", "mapd");
  TQueryResult query_result;
  client->sql_execute(query_result,
                      session,
                      std::string("SELECT * FROM ") + m_tableName.GetAsPlatformString() + std::string(";"),
                      false);
  m_rowSet.Attach(new TRowSet(query_result.row_set));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
simba_unsigned_native QSTable::GetRowCount() {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "GetRowCount");

  if (m_settings->m_useCustomSQLStates) {
    // This is an example of posting a warning with a custom SQL state. If SQLRowCount() is
    // called, the driver will return SQL_SUCCESS_WITH_INFO, a row count of ROW_COUNT_UNKNOWN, a
    // SQL state of QS_TABLE_STATE (QS002) and a warning message that the row count is unknown.
    ErrorException e(QS_TABLE_STATE, QS_ERROR, L"QSRowCountUnknown");
    m_warningListener->PostWarning(e, QS_TABLE_STATE);
  }

  // Return ROW_COUNT_UNKNOWN if HasRowCount() returns false.
  return ROW_COUNT_UNKNOWN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
IColumns* QSTable::GetSelectColumns() {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "GetSelectColumns");
  return m_columns.Get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSTable::HasRowCount() {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "HasRowCount");
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSTable::IsOffsetSupported() {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "IsOffsetSupported");
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::GetCatalogName(simba_wstring& out_catalogName) {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "GetCatalogName");
  out_catalogName = QS_CATALOG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::GetSchemaName(simba_wstring& out_schemaName) {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "GetSchemaName");
  // QuickstartDSII does not support schemas.
  out_schemaName.Clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::GetTableName(simba_wstring& out_tableName) {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "GetTableName");
  out_tableName = m_tableName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSTable::RetrieveData(simba_uint16 in_column,
                           Simba::Support::SqlData* in_data,
                           simba_signed_native in_offset,
                           simba_signed_native in_maxSize) {
  DEBUG_ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "RetrieveData");
  assert(in_data);

  return ConvertData(in_column, in_data, in_offset, in_maxSize);
}

// Protected =======================================================================================
////////////////////////////////////////////////////////////////////////////////////////////////////
QSTable::~QSTable() {
  ;  // Do nothing.
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::DoCloseCursor() {
  ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "DoCloseCursor");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::MoveToBeforeFirstRow() {
  DEBUG_ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "MoveToBeforeFirstRow");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSTable::MoveToNextRow() {
  DEBUG_ENTRANCE_LOG(m_log, "Simba::Quickstart", "QSTable", "MoveToNextRow");
  bool isMoveSuccessful = m_rowIdx < m_rowSet->rows.size();
  ++m_rowIdx;
  return isMoveSuccessful;
}

// Private =========================================================================================
////////////////////////////////////////////////////////////////////////////////////////////////////
bool QSTable::ConvertData(simba_uint16 in_column,
                          SqlData* in_data,
                          simba_signed_native in_offset,
                          simba_signed_native in_maxSize) {
  size_t col_idx{0};
  auto cd = m_columns->GetColumn(in_column);
  simba_wstring col_name;
  cd->GetName(col_name);

  for (; col_idx < m_rowSet->row_desc.size(); ++col_idx) {
    if (simba_wstring(m_rowSet->row_desc[col_idx].col_name) == col_name) {
      break;
    }
  }

  if (col_idx == m_rowSet->row_desc.size()) {
    QSTHROWGEN(L"BadColumn");
  }

  // SqlData is a container class for the internal types that map to the SQL types. The type that
  // is contained is determined by the SQL type that was indicated for each column as defined
  // by the metadata returned by GetSelectColumns() for the result set. See SqlData.h for a list
  // of the contained types and how they map to the SQL types. To indicate NULL data, call
  // in_data->SetNull(true).
  auto col_datum = m_rowSet->rows[m_currentRow].cols[col_idx];
  if (col_datum.is_null) {
    in_data->SetNull(true);
    return false;
  }

  // Copy the data over depending on the data type.
  switch (in_data->GetMetadata()->GetTDWType()) {
    case TDW_SQL_CHAR:
    case TDW_SQL_VARCHAR:
    case TDW_SQL_LONGVARCHAR: {
      // This utility function will automatically handle chunking data back if the amount of
      // data in the column exceeds the amount of data requested, which will result in
      // multiple calls to retrieve the data with increasing offsets.
      return DSITypeUtilities::OutputVarCharStringData(col_datum.val.str_val, in_data, in_offset, in_maxSize);
      break;
    }

    case TDW_SQL_WCHAR:
    case TDW_SQL_WVARCHAR:
    case TDW_SQL_WLONGVARCHAR: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_UBIGINT: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_SBIGINT: {
      assert(!in_data->GetMetadata()->IsUnsigned());
      *reinterpret_cast<simba_int64*>(in_data->GetBuffer()) = col_datum.val.int_val;
      break;
    }

    case TDW_SQL_UINTEGER: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_SINTEGER: {
      assert(!in_data->GetMetadata()->IsUnsigned());
      *reinterpret_cast<simba_int32*>(in_data->GetBuffer()) = col_datum.val.int_val;
      break;
    }

    case TDW_SQL_USMALLINT: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_SSMALLINT: {
      assert(!in_data->GetMetadata()->IsUnsigned());
      *reinterpret_cast<simba_int16*>(in_data->GetBuffer()) = col_datum.val.int_val;
      break;
    }

    case TDW_SQL_UTINYINT: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_STINYINT: {
      assert(!in_data->GetMetadata()->IsUnsigned());
      *reinterpret_cast<simba_int8*>(in_data->GetBuffer()) = col_datum.val.int_val;
      break;
    }

    case TDW_SQL_BIT: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_REAL: {
      *reinterpret_cast<simba_double32*>(in_data->GetBuffer()) = static_cast<simba_double32>(col_datum.val.real_val);
      break;
    }

    case TDW_SQL_DOUBLE: {
      *reinterpret_cast<simba_double64*>(in_data->GetBuffer()) = col_datum.val.real_val;
      break;
    }

    case TDW_SQL_TYPE_TIME: {
      std::tm tm_struct;
      const auto ts = static_cast<time_t>(col_datum.val.int_val);
      gmtime_r(&ts, &tm_struct);
      *reinterpret_cast<TDWTime*>(in_data->GetBuffer()) =
          TDWTime(tm_struct.tm_hour, tm_struct.tm_min, tm_struct.tm_sec);
      break;
    }

    case TDW_SQL_TYPE_TIMESTAMP: {
      std::tm tm_struct;
      const auto ts = static_cast<time_t>(col_datum.val.int_val);
      gmtime_r(&ts, &tm_struct);
      *reinterpret_cast<TDWTimestamp*>(in_data->GetBuffer()) = TDWTimestamp(tm_struct.tm_year + 1900,
                                                                            tm_struct.tm_mon + 1,
                                                                            tm_struct.tm_mday,
                                                                            tm_struct.tm_hour,
                                                                            tm_struct.tm_min,
                                                                            tm_struct.tm_sec,
                                                                            0);
      break;
    }

    case TDW_SQL_TYPE_DATE: {
      std::tm tm_struct;
      const auto ts = static_cast<time_t>(col_datum.val.int_val);
      gmtime_r(&ts, &tm_struct);
      *reinterpret_cast<TDWDate*>(in_data->GetBuffer()) =
          TDWDate(tm_struct.tm_year + 1900, tm_struct.tm_mon + 1, tm_struct.tm_mday);
      break;
    }

    case TDW_SQL_BINARY:
    case TDW_SQL_VARBINARY:
    case TDW_SQL_LONGVARBINARY: {
      QSTHROWGEN(L"NotSupported");
      break;
    }

    case TDW_SQL_DECIMAL:
    case TDW_SQL_NUMERIC: {
      TDWExactNumericType* numeric = reinterpret_cast<TDWExactNumericType*>(in_data->GetBuffer());
      numeric->Set(std::to_string(col_datum.val.real_val));
      numeric->SetScale(2);  // TODO
      break;
    }

    default: { QSTHROWGEN1(L"QSUnknownType", in_data->GetMetadata()->GetLocalTypeName()); }
  }

  return false;
}

namespace {

simba_int16 from_thrift_type(const TTypeInfo& type_info) {
  switch (type_info.type) {
    case TDatumType::SMALLINT:
      return SQL_SMALLINT;
    case TDatumType::INT:
      return SQL_INTEGER;
    case TDatumType::BIGINT:
      return SQL_BIGINT;
    case TDatumType::FLOAT:
      return SQL_REAL;
    case TDatumType::DECIMAL:
      return SQL_DECIMAL;
    case TDatumType::DOUBLE:
      return SQL_DOUBLE;
    case TDatumType::STR:
      return SQL_VARCHAR;
    case TDatumType::TIME:
      return SQL_TIME;
    case TDatumType::TIMESTAMP:
      return SQL_TIMESTAMP;
    case TDatumType::DATE:
      return SQL_DATE;
    case 10:
      return SQL_TINYINT;
    default:
      break;
  }
  return SQL_UNKNOWN_TYPE;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////
void QSTable::InitializeColumns(bool in_isODBCV3) {
  auto columns = new DSIResultSetColumns();

  MapDClientHandle client(9091);

  auto session = client->connect("mapd", "HyperInteractive", "mapd");
  TTableDescriptor td;
  client->get_table_descriptor(td, session, m_tableName.GetAsPlatformString());

  for (const auto& kv : td) {
    AutoPtr<DSIColumnMetadata> columnMetadata(new DSIColumnMetadata());

    columnMetadata->m_name = kv.first;
    columnMetadata->m_schemaName.Clear();
    columnMetadata->m_tableName = m_tableName;
    columnMetadata->m_label = kv.second.col_name;
    columnMetadata->m_unnamed = false;
    columnMetadata->m_charOrBinarySize = m_settings->defaultMaxColumnSize;
    columnMetadata->m_nullable = kv.second.col_type.nullable ? DSI_NULLABLE : DSI_NO_NULLS;

    simba_int16 type = from_thrift_type(kv.second.col_type);

    AutoPtr<SqlTypeMetadata> sqlTypeMetadata(
        SqlTypeMetadataFactorySingleton::GetInstance()->CreateNewSqlTypeMetadata(type));

    if (sqlTypeMetadata->IsCharacterOrBinaryType()) {
      // Variable length columns need to have their lengths match the binary size.
      sqlTypeMetadata->SetLengthOrIntervalPrecision(columnMetadata->m_charOrBinarySize);
    } else if (sqlTypeMetadata->IsExactNumericType()) {
      // Exact numeric columns have a preset precision of 38 (default) and scale of 10.
      sqlTypeMetadata->SetScale(10);
    }

    columns->AddColumn(new DSIResultSetColumn(sqlTypeMetadata, columnMetadata));
  }

  m_columns.Attach(columns);
}
