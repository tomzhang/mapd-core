#include "BaseQueryDataTable.h"
#include "../QueryRendererContext.h"

namespace QueryRenderer {

BaseQueryDataTableJSON::BaseQueryDataTableJSON(const QueryRendererContextShPtr& ctx,
                                               const std::string& name,
                                               const rapidjson::Value& obj,
                                               const rapidjson::Pointer& objPath)
    : JSONRefObject(RefType::DATA, name, objPath), _ctx(ctx) {
}

bool BaseQueryDataTableJSON::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    _jsonPath = objPath;
    return false;
  }

  _updateFromJSONObj(obj, objPath);

  _jsonPath = objPath;

  return true;
}

std::string BaseQueryDataTableJSON::_printInfo() const {
  return to_string(_ctx->getUserWidgetIds()) + ", name: " + _name + ", json path: " +
         RapidJSONUtils::getPointerPath(_jsonPath);
}

BaseQueryDataTableSQLJSON::BaseQueryDataTableSQLJSON(const QueryRendererContextShPtr& ctx,
                                                     const std::string& name,
                                                     const rapidjson::Value& obj,
                                                     const rapidjson::Pointer& objPath,
                                                     bool isPolyQuery)
    : BaseQueryDataTableJSON(ctx, name, obj, objPath), BaseQueryDataTableSQL(), _isPolyQuery(isPolyQuery) {
}

::Rendering::GL::Resources::GLBufferLayoutShPtr BaseQueryDataTableSQLJSON::getGLBufferLayout() const {
  auto qlayout = getQueryDataLayout();
  if (qlayout) {
    return qlayout->getBufferLayout();
  }
  return nullptr;
}

void BaseQueryDataTableSQLJSON::_updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  rapidjson::Value::ConstMemberIterator itr;

  RUNTIME_EX_ASSERT(
      (itr = obj.FindMember("sql")) != obj.MemberEnd() && itr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(
          obj, "SQL data object \"" + _name + "\" must contain an \"sql\" property and it must be a string"));

  _sqlQueryStr = itr->value.GetString();

  // TODO(croot) -- should we validate the sql?

  // TODO(croot) - for backwards compatibility, the dbTableName doesn't have to be present
  // but should it be required? Or can we somehow extract it from the sql?
  if ((itr = obj.FindMember("dbTableName")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          itr->value, "SQL data object \"" + _name + "\" \"dbTableName\" property must be a string"));

    _tableName = itr->value.GetString();
  } else {
    _tableName = "";
  }

  _jsonPath = objPath;
}

bool BaseQueryDataTableSQLJSON::_executeQuery(const rapidjson::Value* dataObj) {
  // now execute the query
  // TODO(croot): should this be a lazy load? The only time executing here could
  // be wasteful to do here is if this data is not referenced at all when
  // performing a render In that case, it's probably better to throw a warning
  // or something instead to notify the user that there is unused data.

  // NOTE: "select x, y from tweets;" was a placeholder sql in the vega
  // Need to check for that for backwards compatibility.
  if (_sqlQueryStr.length() && _sqlQueryStr != "select x, y from tweets;") {
    auto executor = _ctx->getExecutor();

    if (executor) {
      auto executeCB = _ctx->getQueryExecutionFunc();
      if (executeCB) {
        auto renderTimerPtr = _ctx->getRenderTimer();
        const rapidjson::Value* obj = dataObj;
        if (!obj) {
          obj = _ctx->getJSONObj(_jsonPath);
        }
        CHECK(obj);
        auto render_info = executeCB(*renderTimerPtr, executor, _sqlQueryStr, obj, _isPolyQuery);
        auto& tables = render_info.first;

        // avoid a superfluous copy
        _queryDataLayoutPtr = (render_info.second ? std::move(render_info.second) : nullptr);

        RUNTIME_EX_ASSERT(tables.size() == 1, "The query " + _sqlQueryStr + " is too complex to render");

        LOG_IF(WARNING, _tableName.size() > 0 && _tableName != tables[0].second)
            << "The primary table to render \"" << tables[0].second
            << "\" does not match the dbTableName json attribute: " << _tableName
            << ". Using the table referenced in the query.";

        _tableId = tables[0].first;
        _tableName = tables[0].second;

        return true;
      }
    }
  } else {
    _tableId = -1;
  }

  return false;
}

}  // namespace QueryRenderer
