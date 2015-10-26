#ifndef DATA_TABLE_H_
#define DATA_TABLE_H_

#include "VertexBuffer.h"
#include "TypeGL.h"
#include "Color.h"

#include <boost/lexical_cast.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

#include <string>
#include <memory>
#include <algorithm>  // minmax_element

#include "rapidjson/document.h"

namespace MapD_Renderer {

enum class DataType { UINT = 0, INT, FLOAT, DOUBLE, COLOR };

class BaseDataTableVBO {
 public:
  enum class DataTableType { SQLQUERY = 0, OTHER };

  BaseDataTableVBO(const std::string& name = "", DataTableType type = DataTableType::OTHER)
      : _name(name), _vbo(nullptr), _type(type) {}
  explicit BaseDataTableVBO(const std::string& name, DataTableType type, const BaseVertexBufferShPtr& vbo)
      : _name(name), _vbo(vbo), _type(type) {}
  virtual ~BaseDataTableVBO() {}

  virtual bool hasColumn(const std::string& columnName) = 0;
  virtual BaseVertexBufferShPtr getColumnDataVBO(const std::string& columnName) = 0;
  virtual DataType getColumnType(const std::string& columnName) = 0;
  virtual int numRows() = 0;

  std::string getName() { return _name; }
  DataTableType getType() { return _type; }

 protected:
  std::string _name;
  BaseVertexBufferShPtr _vbo;
  DataTableType _type;
};

typedef std::unique_ptr<BaseDataTableVBO> DataVBOUqPtr;
typedef std::shared_ptr<BaseDataTableVBO> DataVBOShPtr;

class SqlQueryDataTable : public BaseDataTableVBO {
 public:
  SqlQueryDataTable(const std::string& name, const BaseVertexBufferShPtr& vbo, const std::string& sqlQueryStr)
      : BaseDataTableVBO(name, BaseDataTableVBO::DataTableType::SQLQUERY, vbo), _sqlQueryStr(sqlQueryStr) {}
  ~SqlQueryDataTable() {}

  bool hasColumn(const std::string& columnName) { return _vbo->hasAttribute(columnName); }
  BaseVertexBufferShPtr getColumnDataVBO(const std::string& columnName) {
    // TODO(croot): throw/log error instead of assert
    assert(_vbo->hasAttribute(columnName));
    return _vbo;
  }
  DataType getColumnType(const std::string& columnName) {
    BufferAttrType attrType = _vbo->getAttributeType(columnName);
    switch (attrType) {
      case BufferAttrType::UINT:
        return DataType::UINT;
      case BufferAttrType::INT:
        return DataType::INT;
      case BufferAttrType::FLOAT:
        return DataType::FLOAT;
      case BufferAttrType::DOUBLE:
        return DataType::DOUBLE;
      case BufferAttrType::VEC4F:
        return DataType::COLOR;
      default:
        // TODO(croot): throw/log error
        assert(false);
    }
  }
  int numRows() { return _vbo->size(); }

 private:
  std::string _sqlQueryStr;
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

  virtual DataType getColumnType() = 0;
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

  void push_back(const T& val) { _columnDataPtr->push_back(val); }

  void push_back(const std::string& val) {
    // TODO: this would throw a boost::bad_lexical_cast error
    // if the conversion can't be done.. I may need to throw
    // a MapD-compliant exception
    _columnDataPtr->push_back(boost::lexical_cast<T>(val));
  }

  DataType getColumnType();

  std::shared_ptr<std::vector<T>> getColumnData() { return _columnDataPtr; }
  TypelessColumnData getTypelessColumnData() {
    return TypelessColumnData({static_cast<void*>(&(*_columnDataPtr)[0]), _columnDataPtr->size(), sizeof(T)});
  };

  std::pair<T, T> getExtrema() {
    auto result = std::minmax_element(_columnDataPtr->begin(), _columnDataPtr->end());

    assert(result.first != _columnDataPtr->end() && result.second != _columnDataPtr->end());

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
DataType TDataColumn<unsigned int>::getColumnType();

template <>
DataType TDataColumn<int>::getColumnType();

template <>
DataType TDataColumn<float>::getColumnType();

template <>
DataType TDataColumn<double>::getColumnType();

template <>
DataType TDataColumn<ColorRGBA>::getColumnType();

template <>
void TDataColumn<ColorRGBA>::push_back(const std::string& val);

template <>
void TDataColumn<ColorRGBA>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj);

class DataTable : public BaseDataTableVBO {
 public:
  enum class VboType { SEQUENTIAL = 0, INTERLEAVED, INDIVIDUAL };
  static const std::string defaultIdColumnName;

  DataTable(const std::string& name,
            const rapidjson::Value& obj,
            bool buildIdColumn = false,
            VboType vboType = VboType::SEQUENTIAL);
  ~DataTable() {}

  template <typename C1, typename C2>
  std::pair<C1, C2> getExtrema(const std::string& column);

  bool hasColumn(const std::string& columnName) {
    ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
    return (nameLookup.find(columnName) != nameLookup.end());
  }

  DataType getColumnType(const std::string& columnName);
  DataColumnShPtr getColumn(const std::string& columnName);

  BaseVertexBufferShPtr getColumnDataVBO(const std::string& columnName);

  int numRows() { return _numRows; }

 private:
  VboType _vboType;
  int _numRows;

  typedef multi_index_container<
      DataColumnShPtr,
      indexed_by<random_access<>,

                 // hashed on name
                 hashed_unique<tag<DataColumn::ColumnName>, member<DataColumn, std::string, &DataColumn::columnName>>>>
      ColumnMap;

  typedef ColumnMap::index<DataColumn::ColumnName>::type ColumnMap_by_name;

  ColumnMap _columns;

  void _buildColumnsFromJSONObj(const rapidjson::Value& obj, bool buildIdColumn);
  void _populateColumnsFromJSONObj(const rapidjson::Value& obj);
  void _readDataFromFile(const std::string& filename);
  void _readFromCsvFile(const std::string& filename);
};

typedef std::unique_ptr<DataTable> DataTableUqPtr;
typedef std::shared_ptr<DataTable> DataTableShPtr;

}  // namespace MapD_Renderer

#endif  // DATA_TABLE_H_
