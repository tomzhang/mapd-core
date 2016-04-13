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
                     QueryDataTableType type,
                     QueryDataTableBaseType baseType)
      : _ctx(ctx), _name(name), _type(type), _baseType(baseType), _jsonPath(objPath) {}
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

  virtual bool hasAttribute(const std::string& attrName) = 0;
  virtual QueryBufferShPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) = 0;
  virtual std::map<GpuId, QueryBufferShPtr> getAttributeDataBuffers(const std::string& attrName) = 0;
  virtual QueryDataType getAttributeType(const std::string& attrName) = 0;

  std::string getName() const { return _name; }
  QueryDataTableType getType() const { return _type; }
  QueryDataTableBaseType getBaseType() const { return _baseType; }

  virtual operator std::string() const = 0;

 protected:
  std::string _printInfo() const { return to_string(_ctx->getUserWidgetIds()); }

  QueryRendererContextShPtr _ctx;
  std::string _name;
  QueryDataTableType _type;
  QueryDataTableBaseType _baseType;

  rapidjson::Pointer _jsonPath;

 private:
  virtual void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

  virtual void _initGpuResources(const QueryRendererContext* ctx,
                                 const std::unordered_set<GpuId>& unusedGpus = std::unordered_set<GpuId>(),
                                 bool initializing = true) = 0;

  friend class QueryRendererContext;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
