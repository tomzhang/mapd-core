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

BaseQueryDataTableSQL::BaseQueryDataTableSQL(const std::string& tableName, const std::string& sqlQueryStr)
    : _tableId(NonProjectionRenderQueryCacheMap::emptyTableId), _tableName(tableName), _sqlQueryStr(sqlQueryStr) {
}

int32_t BaseQueryDataTableSQL::getTableId() const {
  return (_queryCachePtr ? _queryCachePtr->tableId : _tableId);
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

bool BaseQueryDataTableSQLJSON::_executeQuery(const rapidjson::Value* dataObj, const std::string& sqlQueryOverride) {
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
      const std::string& sqlQueryToUse(sqlQueryOverride.empty() ? _sqlQueryStr : sqlQueryOverride);
      auto executeCB = _ctx->getQueryExecutionFunc();
      if (executeCB) {
        auto renderTimerPtr = _ctx->getRenderTimer();
        const rapidjson::Value* obj = dataObj;
        if (!obj) {
          obj = _ctx->getJSONObj(_jsonPath);
        }
        CHECK(obj);
        auto render_info = executeCB(*renderTimerPtr, executor, sqlQueryToUse, obj, _isPolyQuery);

        auto& tables = std::get<3>(render_info);
        CHECK(tables.size() > 0);

        LOG_IF(WARNING, !_tableName.empty() && _tableName != tables[0].second)
            << "The primary table to render \"" << tables[0].second
            << "\" does not match the dbTableName json attribute: " << _tableName
            << ". Using the table referenced in the query.";

        _tableName = tables[0].second;
        _tableId = tables[0].first;

        // avoid a superfluous copy
        _queryDataLayoutPtr = (std::get<4>(render_info) ? std::move(std::get<4>(render_info)) : nullptr);

        auto& rows = std::get<0>(render_info);
        if (tables.size() > 1) {
          // TODO(croot): is the number of table names used in a query good enough
          // in determining whether a query is a non-projection query?

          // should only have results returned from the query execute callback when
          // it's a non-projection query.
          auto baseTablePtr = dynamic_cast<BaseQueryDataTable*>(this);
          CHECK(baseTablePtr);

          auto rootCache = _ctx->getRootGpuCache();
          CHECK(rootCache);
          auto& queryCacheMap = rootCache->renderQueryCacheMap;

          if (std::get<2>(render_info) > NonProjectionRenderQueryCacheMap::maxQueryTime) {
            // cache query results if it took more than a max time
            // and there's enough memory to store the cache

            if (!queryCacheMap.hasQueryCache(sqlQueryToUse)) {
              if (queryCacheMap.canFitResults(rows)) {
                _queryCachePtr = queryCacheMap.addQueryResultToCache(
                    sqlQueryToUse, std::move(rows), std::move(std::get<1>(render_info)), std::move(tables));
              } else {
                _queryCachePtr = queryCacheMap.addQueryOnlyToCache(sqlQueryToUse, std::move(tables));
              }
            } else if (queryCacheMap.canFitUpdatedResults(sqlQueryToUse, rows)) {
              _queryCachePtr = queryCacheMap.updateQueryResultsInCache(
                  sqlQueryToUse, std::move(rows), std::move(std::get<1>(render_info)));
            } else {
              // cannot store the results in the cache, so we need to force
              // the query to run on hit-testing. That means removing the cached results
              _queryCachePtr = queryCacheMap.removeQueryResultsFromCache(sqlQueryToUse);
            }
          } else if (queryCacheMap.hasQueryCache(sqlQueryToUse)) {
            _queryCachePtr = queryCacheMap.removeQueryResultsFromCache(sqlQueryToUse);
          } else {
            _queryCachePtr = queryCacheMap.addQueryOnlyToCache(sqlQueryToUse, std::move(tables));
          }
        } else {
          _queryCachePtr = nullptr;
        }

        return true;
      }
    }
  } else {
    _tableId = NonProjectionRenderQueryCacheMap::emptyTableId;
    _queryCachePtr = nullptr;
  }

  return false;
}

}  // namespace QueryRenderer
