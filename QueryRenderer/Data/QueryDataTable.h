#ifndef QUERYRENDERER_QUERYDATATABLE_H_
#define QUERYRENDERER_QUERYDATATABLE_H_

#include "Types.h"
#include "BaseQueryDataTable.h"
#include "../PerGpuData.h"
#include "../QueryRendererContext.h"
#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Renderer/GL/Resources/GLVertexBuffer.h>

#include <boost/lexical_cast.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

#include <map>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

template <typename T>
QueryDataType getDataTypeForType();

template <>
QueryDataType getDataTypeForType<unsigned int>();

template <>
QueryDataType getDataTypeForType<int>();

template <>
QueryDataType getDataTypeForType<float>();

template <>
QueryDataType getDataTypeForType<double>();

template <>
QueryDataType getDataTypeForType<Rendering::Objects::ColorRGBA>();

template <>
QueryDataType getDataTypeForType<std::string>();

class BaseQueryDataTableVBO : public BaseQueryDataTable {
 public:
  BaseQueryDataTableVBO(const QueryRendererContextShPtr& ctx,
                        const std::string& name,
                        const rapidjson::Value& obj,
                        const rapidjson::Pointer& objPath,
                        QueryDataTableType type)
      : BaseQueryDataTable(ctx, name, obj, objPath, type) {
    _initGpuResources(ctx.get());
  }
  explicit BaseQueryDataTableVBO(const QueryRendererContextShPtr& ctx,
                                 const std::string& name,
                                 const rapidjson::Value& obj,
                                 const rapidjson::Pointer& objPath,
                                 QueryDataTableType type,
                                 const std::map<GpuId, QueryVertexBufferShPtr>& vboMap)
      : BaseQueryDataTableVBO(ctx, name, obj, objPath, type) {
    _initVBOs(vboMap);
  }
  virtual ~BaseQueryDataTableVBO() {}

  virtual bool hasColumn(const std::string& columnName) = 0;
  virtual QueryVertexBufferShPtr getColumnDataVBO(const GpuId& gpuId, const std::string& columnName) = 0;
  virtual std::map<GpuId, QueryVertexBufferShPtr> getColumnDataVBOs(const std::string& columnName) = 0;
  virtual QueryDataType getColumnType(const std::string& columnName) = 0;

 protected:
  struct PerGpuData : BasePerGpuData {
    QueryVertexBufferShPtr vbo;

    PerGpuData() : BasePerGpuData(), vbo(nullptr) {}
    explicit PerGpuData(const BasePerGpuData& data, const QueryVertexBufferShPtr& vbo = nullptr)
        : BasePerGpuData(data), vbo(vbo) {}
    PerGpuData(const PerGpuData& data) : BasePerGpuData(data), vbo(data.vbo) {}
    PerGpuData(PerGpuData&& data) : BasePerGpuData(std::move(data)), vbo(std::move(data.vbo)) {}

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;

 private:
  void _initGpuResources(const QueryRendererContext* ctx,
                         const std::unordered_set<GpuId>& unusedGpus = std::unordered_set<GpuId>(),
                         bool initializing = true);

  void _initVBOs(const std::map<GpuId, QueryVertexBufferShPtr>& vboMap);

  friend class QueryRendererContext;
};

class SqlQueryDataTable : public BaseQueryDataTableVBO {
 public:
  SqlQueryDataTable(const QueryRendererContextShPtr& ctx,
                    const std::string& name,
                    const rapidjson::Value& obj,
                    const rapidjson::Pointer& objPath,
                    const std::map<GpuId, QueryVertexBufferShPtr>& vboMap,
                    const std::string& sqlQueryStr = "")
      : BaseQueryDataTableVBO(ctx, name, obj, objPath, QueryDataTableType::SQLQUERY, vboMap),
        _sqlQueryStr(sqlQueryStr),
        _tableName() {
    _initFromJSONObj(obj, objPath);
  }
  ~SqlQueryDataTable() {}

  bool hasColumn(const std::string& columnName);

  QueryVertexBufferShPtr getColumnDataVBO(const GpuId& gpuId, const std::string& columnName);

  std::map<GpuId, QueryVertexBufferShPtr> getColumnDataVBOs(const std::string& columnName) final;

  QueryDataType getColumnType(const std::string& columnName);
  int numRows(const GpuId& gpuId);

  std::string getTableName() { return _tableName; }

  operator std::string() const final;

 private:
  std::string _sqlQueryStr;
  std::string _tableName;

  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool forceUpdate = false);
  void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;
};

struct TypelessColumnData {
  void* data;
  size_t numItems;
  size_t numBytesPerItem;
};

class DataColumn {
 public:
  enum class InitType { ROW_MAJOR = 0, COLUMN_MAJOR };

  std::string columnName;

  explicit DataColumn(const std::string& name) : columnName(name) {}
  virtual ~DataColumn() {}

  virtual int size() = 0;

  virtual QueryDataType getColumnType() = 0;
  virtual TypelessColumnData getTypelessColumnData() = 0;
  virtual void push_back(const std::string& val) = 0;

  // for multi_index_container tag
  struct ColumnName {};
};

typedef std::unique_ptr<DataColumn> DataColumnUqPtr;
typedef std::shared_ptr<DataColumn> DataColumnShPtr;

template <typename T>
class TDataColumn : public DataColumn {
 public:
  TDataColumn(const std::string& name, int size = 0) : DataColumn(name), _columnDataPtr(new std::vector<T>(size)) {}
  TDataColumn(const std::string& name, const rapidjson::Value& dataArrayObj, InitType initType)
      : DataColumn(name), _columnDataPtr(new std::vector<T>()) {
    if (initType == DataColumn::InitType::ROW_MAJOR) {
      _initFromRowMajorJSONObj(dataArrayObj);
    } else {
      _initFromColMajorJSONObj(dataArrayObj);
    }
  }
  ~TDataColumn() {}

  T& operator[](unsigned int i) { return (*_columnDataPtr)[i]; }

  // void push_back(const T& val) { _columnDataPtr->push_back(val); }

  void push_back(const std::string& val) {
    // TODO(croot): this would throw a boost::bad_lexical_cast error
    // if the conversion can't be done.. I may need to throw
    // a MapD-compliant exception
    _columnDataPtr->push_back(boost::lexical_cast<T>(val));
  }

  QueryDataType getColumnType() { return getDataTypeForType<T>(); }

  std::shared_ptr<std::vector<T>> getColumnData() { return _columnDataPtr; }
  TypelessColumnData getTypelessColumnData() {
    return TypelessColumnData({static_cast<void*>(&(*_columnDataPtr)[0]), _columnDataPtr->size(), sizeof(T)});
  };

  std::pair<T, T> getExtrema() {
    auto result = std::minmax_element(_columnDataPtr->begin(), _columnDataPtr->end());

    RUNTIME_EX_ASSERT(result.first != _columnDataPtr->end() && result.second != _columnDataPtr->end(),
                      std::string(*this) + " getExtrema(): cannot find the extrema of the column.");

    return std::make_pair(*result.first, *result.second);
  }

  int size() { return _columnDataPtr->size(); }

  operator std::string() const {
    return "TDataColumn<" + std::string(typeid(T).name()) + ">(column name: " + columnName + ")";
  }

 private:
  std::shared_ptr<std::vector<T>> _columnDataPtr;
  // TypeGL<T> _typeGL;

  void _initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
    RUNTIME_EX_ASSERT(dataArrayObj.IsArray(),
                      RapidJSONUtils::getJsonParseErrorStr(dataArrayObj, "Row-major data object is not an array."));

    rapidjson::Value::ConstValueIterator vitr;
    rapidjson::Value::ConstMemberIterator mitr;

    for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
      RUNTIME_EX_ASSERT(
          vitr->IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(dataArrayObj,
                                               "Item " + std::to_string(vitr - dataArrayObj.Begin()) +
                                                   "in data array must be an object for row-major-defined data."));
      RUNTIME_EX_ASSERT((mitr = vitr->FindMember(columnName.c_str())) != vitr->MemberEnd(),
                        RapidJSONUtils::getJsonParseErrorStr(*vitr,
                                                             "column \"" + columnName +
                                                                 "\" does not exist in row-major-defined data item " +
                                                                 std::to_string(vitr - dataArrayObj.Begin())));

      _columnDataPtr->push_back(RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value));
    }
  }

  void _initFromColMajorJSONObj(const rapidjson::Value& dataArrayObj) {
    THROW_RUNTIME_EX("Column-major data is not yet supported.");
  }
};

template <>
void TDataColumn<::Rendering::Objects::ColorRGBA>::push_back(const std::string& val);

template <>
void TDataColumn<::Rendering::Objects::ColorRGBA>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj);

class DataTable : public BaseQueryDataTableVBO {
 public:
  enum class VboType { SEQUENTIAL = 0, INTERLEAVED, INDIVIDUAL };
  static const std::string defaultIdColumnName;

  static DataColumnUqPtr createDataColumnFromRowMajorObj(const std::string& columnName,
                                                         const rapidjson::Value& rowItem,
                                                         const rapidjson::Value& dataArray);

  static DataColumnUqPtr createColorDataColumnFromRowMajorObj(const std::string& columnName,
                                                              const rapidjson::Value& rowItem,
                                                              const rapidjson::Value& dataArray);

  DataTable(const QueryRendererContextShPtr& ctx,
            const std::string& name,
            const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            QueryDataTableType type,
            bool buildIdColumn = false,
            VboType vboType = VboType::SEQUENTIAL);
  ~DataTable() {}

  template <typename C1, typename C2>
  std::pair<C1, C2> getExtrema(const std::string& column);

  bool hasColumn(const std::string& columnName) {
    ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
    return (nameLookup.find(columnName) != nameLookup.end());
  }

  QueryDataType getColumnType(const std::string& columnName);
  DataColumnShPtr getColumn(const std::string& columnName);

  QueryVertexBufferShPtr getColumnDataVBO(const GpuId& gpuId, const std::string& columnName);
  std::map<GpuId, QueryVertexBufferShPtr> getColumnDataVBOs(const std::string& columnName) final;

  int numRows(const GpuId& gpuId) { return _numRows; }

  operator std::string() const final;

 private:
  VboType _vboType;
  int _numRows;

  typedef boost::multi_index_container<
      DataColumnShPtr,
      boost::multi_index::indexed_by<boost::multi_index::random_access<>,

                                     // hashed on name
                                     boost::multi_index::hashed_unique<
                                         boost::multi_index::tag<DataColumn::ColumnName>,
                                         boost::multi_index::member<DataColumn, std::string, &DataColumn::columnName>>>>
      ColumnMap;

  typedef ColumnMap::index<DataColumn::ColumnName>::type ColumnMap_by_name;

  ColumnMap _columns;

  void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;
  void _buildColumnsFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool buildIdColumn);
  void _populateColumnsFromJSONObj(const rapidjson::Value& obj);
  void _readDataFromFile(const std::string& filename);
  void _readFromCsvFile(const std::string& filename);

  std::pair<::Rendering::GL::Resources::GLBufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>>
  _createVBOData();
};

typedef std::unique_ptr<DataTable> DataTableUqPtr;
typedef std::shared_ptr<DataTable> DataTableShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYDATATABLE_H_
