#ifndef QUERYRENDERER_DATA_UTILS_H_
#define QUERYRENDERER_DATA_UTILS_H_

#include "Types.h"
#include "../Types.h"
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj);
std::pair<QueryDataTableBaseType, QueryDataTableType> getDataTableTypesFromJSONObj(const rapidjson::Value& obj);
QueryDataTableShPtr createDataTable(const rapidjson::Value& obj,
                                    const rapidjson::Pointer& objPath,
                                    const QueryRendererContextShPtr& ctx,
                                    const std::string& name = "");

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_UTILS_H_
