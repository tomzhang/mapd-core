#ifndef QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
#define QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_

#include "Types.h"
#include "../Types.h"
#include "../QueryRendererContext.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

class BaseQueryDataTable {
 public:
  BaseQueryDataTable(const QueryRendererContextShPtr& ctx,
                     const std::string& name,
                     const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     QueryDataTableType type)
      : _ctx(ctx), _name(name), _type(type), _jsonPath(objPath) {}
  virtual ~BaseQueryDataTable() {}

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
    if (_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
      _jsonPath = objPath;
      return;
    }

    _updateFromJSONObj(obj, objPath);

    _jsonPath = objPath;
  }

  virtual int numRows(const GpuId& gpuId) = 0;

  std::string getName() { return _name; }
  QueryDataTableType getType() { return _type; }

  virtual operator std::string() const = 0;

 protected:
  std::string _printInfo() const { return to_string(_ctx->getUserWidgetIds()); }

  QueryRendererContextShPtr _ctx;
  std::string _name;
  QueryDataTableType _type;

  rapidjson::Pointer _jsonPath;

 private:
  virtual void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
