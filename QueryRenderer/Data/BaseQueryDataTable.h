#ifndef QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
#define QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_

#include "Types.h"
#include "../Types.h"
#include "../Interop/Types.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

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

class BaseQueryDataTableJSON {
 public:
  BaseQueryDataTableJSON(const QueryRendererContextShPtr& ctx,
                         const std::string& name,
                         const rapidjson::Value& obj,
                         const rapidjson::Pointer& objPath);
  virtual ~BaseQueryDataTableJSON() {}

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

  std::string getName() const { return _name; }

  virtual operator std::string() const = 0;

 protected:
  QueryRendererContextShPtr _ctx;
  std::string _name;
  rapidjson::Pointer _jsonPath;

  std::string _printInfo() const;

 private:
  virtual void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
};

class BaseQueryDataTableSQL {
 public:
  BaseQueryDataTableSQL(const std::string& tableName, const std::string& sqlQueryStr)
      : _tableName(tableName), _sqlQueryStr(sqlQueryStr) {}
  virtual ~BaseQueryDataTableSQL() {}

  std::string getTableName() const { return _tableName; }
  std::string getSqlQueryStr() const { return _sqlQueryStr; }

 protected:
  std::string _tableName;
  std::string _sqlQueryStr;

  std::string _printInfo() const { return "sqlQuery: " + _sqlQueryStr + ", table name: " + _tableName; }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_BASEQUERYDATATABLE_H_
