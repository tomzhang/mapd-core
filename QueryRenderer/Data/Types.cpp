#include "Types.h"

namespace QueryRenderer {

std::string to_string(QueryDataType dataType) {
  switch (dataType) {
    case QueryDataType::UINT:
      return "UINT";
    case QueryDataType::INT:
      return "INT";
    case QueryDataType::FLOAT:
      return "FLOAT";
    case QueryDataType::DOUBLE:
      return "DOUBLE";
    case QueryDataType::UINT64:
      return "UINT64";
    case QueryDataType::INT64:
      return "INT64";
    case QueryDataType::COLOR:
      return "COLOR";
    case QueryDataType::STRING:
      return "STRING";
    case QueryDataType::POLYGON_DOUBLE:
      return "POLYGON_DOUBLE";
    default:
      return "<QueryDataType " + std::to_string(static_cast<int>(dataType)) + ">";
  }
  return "";
}

std::string to_string(QueryDataTableType dataTableType) {
  switch (dataTableType) {
    case QueryDataTableType::SQLQUERY:
      return "SQLQUERY";
    case QueryDataTableType::EMBEDDED:
      return "EMBEDDED";
    case QueryDataTableType::URL:
      return "URL";
    case QueryDataTableType::UNSUPPORTED:
      return "UNSUPPORTED";
    default:
      return "<QueryDataTableType " + std::to_string(static_cast<int>(dataTableType)) + ">";
  }
  return "";
}

std::string to_string(QueryDataTableBaseType dataTableBaseType) {
  switch (dataTableBaseType) {
    case QueryDataTableBaseType::BASIC_VBO:
      return "BASIC_VBO";
    case QueryDataTableBaseType::POLY:
      return "POLY";
    default:
      return "<QueryDataTableBaseType " + std::to_string(static_cast<int>(dataTableBaseType)) + ">";
  }
  return "";
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataType value) {
  os << QueryRenderer::to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataTableType value) {
  os << QueryRenderer::to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataTableBaseType value) {
  os << QueryRenderer::to_string(value);
  return os;
}
