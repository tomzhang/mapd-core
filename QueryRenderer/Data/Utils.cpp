#include "Utils.h"
#include "QueryDataTable.h"
#include "QueryPolyDataTable.h"
#include "../Utils/RapidJSONUtils.h"

namespace QueryRenderer {

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(),
                    RapidJSONUtils::getJsonParseErrorStr(obj, "A data object in the JSON must be an object."));

  rapidjson::Value::ConstMemberIterator itr;

  RUNTIME_EX_ASSERT((itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "A data object must contain a \"name\" property and it must be a string"));

  return itr->value.GetString();
}

std::pair<QueryDataTableBaseType, QueryDataTableType> getDataTableTypesFromJSONObj(const rapidjson::Value& obj) {
  QueryDataTableBaseType baseType = QueryDataTableBaseType::BASIC_VBO;
  QueryDataTableType tableType = QueryDataTableType::UNSUPPORTED;

  RUNTIME_EX_ASSERT(obj.IsObject(),
                    RapidJSONUtils::getJsonParseErrorStr(obj, "A data table in the JSON must be an object."));

  rapidjson::Value::ConstMemberIterator itr;

  if ((itr = obj.FindMember("format")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          itr->value, "The format of a data table is declared as a wrong type. It must be a string."));

    std::string format = itr->value.GetString();
    if (format == "polys") {
      baseType = QueryDataTableBaseType::POLY;
    } else {
      THROW_RUNTIME_EX(
          RapidJSONUtils::getJsonParseErrorStr(itr->value, "Unsupported table format type \"" + format + "\""));
    }
  }

  if ((itr = obj.FindMember("sql")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(
        itr->value.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(
            itr->value, "Cannot get data table's type - the sql property for a data table must be a string."));
    tableType = QueryDataTableType::SQLQUERY;
  } else if ((itr = obj.FindMember("values")) != obj.MemberEnd()) {
    tableType = QueryDataTableType::EMBEDDED;
  } else if ((itr = obj.FindMember("url")) != obj.MemberEnd()) {
    tableType = QueryDataTableType::URL;
  }

  RUNTIME_EX_ASSERT(tableType != QueryDataTableType::UNSUPPORTED,
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "Cannot get data table's type - the data table's type is not supported."));

  return std::make_pair(baseType, tableType);
}

QueryDataTableShPtr createDataTable(const rapidjson::Value& obj,
                                    const rapidjson::Pointer& objPath,
                                    const QueryRendererContextShPtr& ctx,
                                    const std::string& name) {
  std::string tableName(name);
  if (!tableName.length()) {
    tableName = getDataTableNameFromJSONObj(obj);
  } else {
    RUNTIME_EX_ASSERT(obj.IsObject(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          obj, "Cannot create data table - A data object in the JSON must be an object."));
  }

  RUNTIME_EX_ASSERT(tableName.length(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "Cannot create data table - The data table has an empty name. It must have a name."));

  auto tableTypes = getDataTableTypesFromJSONObj(obj);
  switch (tableTypes.second) {
    case QueryDataTableType::SQLQUERY:
      switch (tableTypes.first) {
        case QueryDataTableBaseType::BASIC_VBO:
          // return QueryDataTableVBOShPtr(
          //     new SqlQueryDataTable(ctx, tableName, obj, objPath, ctx->getQueryResultVertexBuffers()));
          return QueryDataTableVBOShPtr(new SqlQueryDataTable(ctx, tableName, obj, objPath));
        // case QueryDataTableBaseType::POLY:
        //   return QueryPolyDataTableShPtr(
        //       new SqlQueryPolyDataTable(ctx, tableName, obj, objPath, ctx->getQueryResultVertexBuffers()));

        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj,
                                                                "Cannot create data table \"" + tableName + "\". " +
                                                                    to_string(tableTypes.first) +
                                                                    " is not a supported table."));
      }
    case QueryDataTableType::EMBEDDED:
    case QueryDataTableType::URL:
      switch (tableTypes.first) {
        case QueryDataTableBaseType::BASIC_VBO:
          return QueryDataTableVBOShPtr(new DataTable(
              ctx, tableName, obj, objPath, tableTypes.second, ctx->doHitTest(), DataTable::VboType::INTERLEAVED));

        case QueryDataTableBaseType::POLY:
          return QueryPolyDataTableShPtr(new PolyDataTable(
              ctx, tableName, obj, objPath, tableTypes.second, ctx->doHitTest(), DataTable::VboType::INTERLEAVED));

        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj,
                                                                "Cannot create data table \"" + tableName + "\". " +
                                                                    to_string(tableTypes.first) +
                                                                    " is not a supported table."));
      }

    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj,
                                                            "Cannot create data table \"" + tableName +
                                                                "\". It is not a supported table. Supported tables "
                                                                "must have an \"sql\", \"values\" or \"url\" "
                                                                "property."));
  }

  return QueryDataTableVBOShPtr(nullptr);
}

}  // namespace QueryRenderer
