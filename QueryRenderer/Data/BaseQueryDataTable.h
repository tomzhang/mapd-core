#ifndef QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
#define QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_

#include "Types.h"
#include "../Types.h"
#include "../Interop/Types.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
#include <Rendering/Renderer/GL/Resources/Types.h>

#include <map>
#include <set>

namespace QueryRenderer {

class BaseQueryDataTable {
 public:
  BaseQueryDataTable(QueryDataTableType type, QueryDataTableBaseType baseType) : _type(type), _baseType(baseType) {}
  virtual ~BaseQueryDataTable() {}

  virtual int numRows(const GpuId& gpuId) = 0;
  virtual bool hasAttribute(const std::string& attrName) = 0;
  virtual QueryBufferShPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) = 0;
  virtual std::map<GpuId, QueryBufferShPtr> getAttributeDataBuffers(const std::string& attrName) = 0;
  virtual QueryDataType getAttributeType(const std::string& attrName) = 0;

  QueryDataTableType getType() const { return _type; }
  QueryDataTableBaseType getBaseType() const { return _baseType; }

  virtual std::set<GpuId> getUsedGpuIds() const = 0;

 protected:
  QueryDataTableType _type;
  QueryDataTableBaseType _baseType;

  std::string _printInfo() const { return "table type: " + to_string(_baseType) + ", data type: " + to_string(_type); }

 private:
  virtual void _initGpuResources(const RootCacheShPtr& qrmPerGpuDataPtr) = 0;

  friend class QueryRendererContext;
};

class BaseQueryDataTableJSON : public JSONRefObject {
 public:
  BaseQueryDataTableJSON(const QueryRendererContextShPtr& ctx,
                         const std::string& name,
                         const rapidjson::Value& obj,
                         const rapidjson::Pointer& objPath);
  virtual ~BaseQueryDataTableJSON() {}

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

  virtual operator std::string() const = 0;

 protected:
  QueryRendererContextShPtr _ctx;

  std::string _printInfo() const;

 private:
  virtual void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
};

class BaseQueryDataTableSQL {
 public:
  BaseQueryDataTableSQL(const std::string& tableName = "", const std::string& sqlQueryStr = "");
  virtual ~BaseQueryDataTableSQL() {}

  int32_t getTableId() const;
  std::string getTableName() const { return _tableName; }
  std::string getSqlQueryStr() const { return _sqlQueryStr; }

 protected:
  NPRQueryCacheShPtr _queryCachePtr;
  TableId _tableId;
  std::string _tableName;
  std::string _sqlQueryStr;

  std::string _printInfo() const { return "sqlQuery: " + _sqlQueryStr + ", table name: " + _tableName; }
};

class BaseQueryDataTableSQLJSON : public BaseQueryDataTableJSON, public BaseQueryDataTableSQL {
 public:
  BaseQueryDataTableSQLJSON(const QueryRendererContextShPtr& ctx,
                            const std::string& name,
                            const rapidjson::Value& obj,
                            const rapidjson::Pointer& objPath,
                            bool isPolyQuery);

  virtual ~BaseQueryDataTableSQLJSON() {}

  virtual QueryDataLayoutShPtr getQueryDataLayout() const { return _queryDataLayoutPtr; }
  ::Rendering::GL::Resources::GLBufferLayoutShPtr getGLBufferLayout() const;

 protected:
  std::string _printInfo() const {
    return BaseQueryDataTableJSON::_printInfo() + ", " + BaseQueryDataTableSQL::_printInfo();
  }

  bool _executeQuery(const rapidjson::Value* dataObj = nullptr, const std::string& sqlQueryOverride = "");
  virtual void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

  QueryDataLayoutShPtr _queryDataLayoutPtr;

 private:
  bool _isPolyQuery;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
