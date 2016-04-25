#include "BaseQueryDataTable.h"
#include "../QueryRendererContext.h"

namespace QueryRenderer {

BaseQueryDataTableJSON::BaseQueryDataTableJSON(const QueryRendererContextShPtr& ctx,
                                               const std::string& name,
                                               const rapidjson::Value& obj,
                                               const rapidjson::Pointer& objPath)
    : _ctx(ctx), _name(name), _jsonPath(objPath) {
}

void BaseQueryDataTableJSON::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    _jsonPath = objPath;
    return;
  }

  _updateFromJSONObj(obj, objPath);

  _jsonPath = objPath;
}

std::string BaseQueryDataTableJSON::_printInfo() const {
  return to_string(_ctx->getUserWidgetIds()) + ", name: " + _name + ", json path: " +
         RapidJSONUtils::getPointerPath(_jsonPath);
}

}  // namespace QueryRenderer
