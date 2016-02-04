#ifndef QUERYRENDERER_QUERYDATATABLE_H_
#define QUERYRENDERER_QUERYDATATABLE_H_

#include "Types.h"
#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Renderer/GL/Resources/GLVertexBuffer.h>
// #include "QueryRendererError.h"
// #include "VertexBuffer.h"
// #include "TypeGL.h"
// #include "Color.h"

#include <boost/lexical_cast.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

// #include <string>
// #include <memory>
// #include <algorithm>  // minmax_element

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

class BaseQueryDataTableVBO {
 public:
  BaseQueryDataTableVBO(const QueryRendererContextShPtr& ctx,
                        const std::string& name,
                        const rapidjson::Value& obj,
                        const rapidjson::Pointer& objPath,
                        QueryDataTableType type)
      : _ctx(ctx), _name(name), _vbo(nullptr), _type(type), _jsonPath(objPath) {}
  explicit BaseQueryDataTableVBO(const QueryRendererContextShPtr& ctx,
                                 const std::string& name,
                                 const rapidjson::Value& obj,
                                 const rapidjson::Pointer& objPath,
                                 QueryDataTableType type,
                                 const Rendering::GL::Resources::GLVertexBufferShPtr& vbo)
      : _ctx(ctx), _name(name), _vbo(vbo), _type(type), _jsonPath(objPath) {}
  virtual ~BaseQueryDataTableVBO() {}

  virtual void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
  virtual bool hasColumn(const std::string& columnName) = 0;
  virtual ::Rendering::GL::Resources::GLVertexBufferShPtr getColumnDataVBO(const std::string& columnName) = 0;
  virtual QueryDataType getColumnType(const std::string& columnName) = 0;
  virtual int numRows() = 0;

  std::string getName() { return _name; }
  QueryDataTableType getType() { return _type; }

 protected:
  QueryRendererContextShPtr _ctx;
  std::string _name;
  ::Rendering::GL::Resources::GLVertexBufferShPtr _vbo;
  QueryDataTableType _type;
  rapidjson::Pointer _jsonPath;
};

class SqlQueryDataTable : public BaseQueryDataTableVBO {
 public:
  SqlQueryDataTable(const QueryRendererContextShPtr& ctx,
                    const std::string& name,
                    const rapidjson::Value& obj,
                    const rapidjson::Pointer& objPath,
                    const ::Rendering::GL::Resources::GLVertexBufferShPtr& vbo,
                    const std::string& sqlQueryStr = "")
      : BaseQueryDataTableVBO(ctx, name, obj, objPath, QueryDataTableType::SQLQUERY, vbo),
        _sqlQueryStr(sqlQueryStr),
        _tableName() {
    _initFromJSONObj(obj, objPath);
  }
  ~SqlQueryDataTable() {}

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  bool hasColumn(const std::string& columnName) { return _vbo->hasAttribute(columnName); }
  ::Rendering::GL::Resources::GLVertexBufferShPtr getColumnDataVBO(const std::string& columnName) {
    RUNTIME_EX_ASSERT(_vbo->hasAttribute(columnName),
                      "SqlQueryDataTable::hasColumn(): column \"" + columnName + "\" does not exist.");
    return _vbo;
  }
  QueryDataType getColumnType(const std::string& columnName);
  int numRows() { return _vbo->numItems(); }

  std::string getTableName() { return _tableName; }

 private:
  std::string _sqlQueryStr;
  std::string _tableName;

  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool forceUpdate = false);
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
  TDataColumn(const std::string& name, const rapidjson::Value& dataArrayObj, InitType initType);
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
                      "TDataColumn::getExtrema(): cannot find the extrema of the column.");

    return std::make_pair(*result.first, *result.second);
  }

  int size() { return _columnDataPtr->size(); }

 private:
  std::shared_ptr<std::vector<T>> _columnDataPtr;
  // TypeGL<T> _typeGL;

  void _initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj);
  void _initFromColMajorJSONObj(const rapidjson::Value& dataArrayObj);
};

template <>
void TDataColumn<::Rendering::Objects::ColorRGBA>::push_back(const std::string& val);

template <>
void TDataColumn<::Rendering::Objects::ColorRGBA>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj);

class DataTable : public BaseQueryDataTableVBO {
 public:
  enum class VboType { SEQUENTIAL = 0, INTERLEAVED, INDIVIDUAL };
  static const std::string defaultIdColumnName;

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

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

  bool hasColumn(const std::string& columnName) {
    ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
    return (nameLookup.find(columnName) != nameLookup.end());
  }

  QueryDataType getColumnType(const std::string& columnName);
  DataColumnShPtr getColumn(const std::string& columnName);

  ::Rendering::GL::Resources::GLVertexBufferShPtr getColumnDataVBO(const std::string& columnName);

  int numRows() { return _numRows; }

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

  void _buildColumnsFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool buildIdColumn);
  void _populateColumnsFromJSONObj(const rapidjson::Value& obj);
  void _readDataFromFile(const std::string& filename);
  void _readFromCsvFile(const std::string& filename);
};

typedef std::unique_ptr<DataTable> DataTableUqPtr;
typedef std::shared_ptr<DataTable> DataTableShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYDATATABLE_H_
