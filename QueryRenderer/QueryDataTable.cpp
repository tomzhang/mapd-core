#include "QueryDataTable.h"
#include "QueryRenderer.h"
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>

// #include "DataTable.h"
// #include "QueryRenderer.h"
// #include "RapidJSONUtils.h"
// #include <limits>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
// #include <fstream>
// #include <regex>

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;
using ::Rendering::GL::Resources::GLBufferAttrType;
using ::Rendering::GL::Resources::GLVertexBufferShPtr;

using ::Rendering::GL::GLRenderer;

using ::Rendering::GL::Resources::GLSequentialBufferLayout;
using ::Rendering::GL::Resources::GLInterleavedBufferLayout;
using ::Rendering::GL::Resources::GLBufferLayoutShPtr;

template <>
QueryDataType getDataTypeForType<unsigned int>() {
  return QueryDataType::UINT;
}

template <>
QueryDataType getDataTypeForType<int>() {
  return QueryDataType::INT;
}

template <>
QueryDataType getDataTypeForType<float>() {
  return QueryDataType::FLOAT;
}

template <>
QueryDataType getDataTypeForType<double>() {
  return QueryDataType::DOUBLE;
}

template <>
QueryDataType getDataTypeForType<ColorRGBA>() {
  return QueryDataType::COLOR;
}

template <>
QueryDataType getDataTypeForType<std::string>() {
  return QueryDataType::STRING;
}

DataColumnUqPtr createDataColumnFromRowMajorObj(const std::string& columnName,
                                                const rapidjson::Value& rowItem,
                                                const rapidjson::Value& dataArray) {
  if (rowItem.IsInt()) {
    return DataColumnUqPtr(new TDataColumn<int>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  } else if (rowItem.IsUint()) {
    return DataColumnUqPtr(new TDataColumn<unsigned int>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  } else if (rowItem.IsDouble()) {
    // TODO(croot): How do we properly handle floats?
    return DataColumnUqPtr(new TDataColumn<double>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));

    // double val = rowItem.GetDouble();
    // if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
    //   return DataColumnUqPtr(new TDataColumn<float>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    // } else {
    //   return DataColumnUqPtr(new TDataColumn<double>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    // }
  } else {
    THROW_RUNTIME_EX("Cannot create data column for column \"" + columnName +
                     "\". The JSON data for the column is not supported.");
  }
}

DataColumnUqPtr createColorDataColumnFromRowMajorObj(const std::string& columnName,
                                                     const rapidjson::Value& rowItem,
                                                     const rapidjson::Value& dataArray) {
  RUNTIME_EX_ASSERT(rowItem.IsString(),
                    "Cannot create color column \"" + columnName + "\". Colors must be defined as strings.");

  return DataColumnUqPtr(new TDataColumn<ColorRGBA>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
}

void SqlQueryDataTable::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    // need to update the path in case the path has changed in the json,
    // but the data is the same.
    _jsonPath = objPath;
    return;
  }

  // force an initialization
  _initFromJSONObj(obj, objPath, true);
}

QueryDataType SqlQueryDataTable::getColumnType(const std::string& columnName) {
  // all vbos should have the same set of columns, so only need to check the first one.
  auto itr = _perGpuData.begin();
  CHECK(itr != _perGpuData.end());

  GLBufferAttrType attrType = itr->second.vbo->getAttributeType(columnName);
  switch (attrType) {
    case GLBufferAttrType::UINT:
      return QueryDataType::UINT;
    case GLBufferAttrType::INT:
      return QueryDataType::INT;
    case GLBufferAttrType::FLOAT:
      return QueryDataType::FLOAT;
    case GLBufferAttrType::DOUBLE:
      return QueryDataType::DOUBLE;
    case GLBufferAttrType::VEC4F:
      return QueryDataType::COLOR;
    default:
      THROW_RUNTIME_EX("SqlQueryDataTable::getColumnType(): Vertex buffer attribute type: " +
                       std::to_string(static_cast<int>(attrType)) + " is not a supported type.");
  }
}

void SqlQueryDataTable::_initFromJSONObj(const rapidjson::Value& obj,
                                         const rapidjson::Pointer& objPath,
                                         bool forceUpdate) {
  rapidjson::Value::ConstMemberIterator itr;
  if (forceUpdate || !_sqlQueryStr.length()) {
    RUNTIME_EX_ASSERT((itr = obj.FindMember("sql")) != obj.MemberEnd() && itr->value.IsString(),
                      "SQL data object \"" + _name + "\" must contain an \"sql\" property and it must be a string");

    _sqlQueryStr = itr->value.GetString();
  }

  // TODO(croot) -- should we validate the sql?

  // TODO(croot) - for backwards compatibility, the dbTableName doesn't have to be present
  // but should it be required? Or can we somehow extract it from the sql?
  if ((forceUpdate || !_tableName.length()) && (itr = obj.FindMember("dbTableName")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(),
                      "SQL data object \"" + _name + "\" \"dbTableName\" property must be a string");

    _tableName = itr->value.GetString();
  }

  _jsonPath = objPath;
}

const std::string DataTable::defaultIdColumnName = "rowid";

DataTable::DataTable(const QueryRendererContextShPtr& ctx,
                     const std::string& name,
                     const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     QueryDataTableType type,
                     bool buildIdColumn,
                     VboType vboType)
    : BaseQueryDataTableVBO(ctx, name, obj, objPath, type), _vboType(vboType), _numRows(0) {
  _buildColumnsFromJSONObj(obj, objPath, buildIdColumn);
}

void DataTable::_readFromCsvFile(const std::string& filename) {
  // // typedef boost::escaped_list_separator<char> char_separator;
  // // typedef boost::tokenizer<char_separator> tokenizer;

  // typedef std::regex_token_iterator<std::string::iterator> tokenizer;

  // // static const std::regex sep("\\b\\s*,*\\s*\\b");
  // static const std::regex sep("\\b[\\s,]+");
  // // static const std::regex sep("\\s+");

  // std::string line;
  // std::ifstream inFile(filename.c_str());

  // // TODO: check for errors and throw exceptions on bad reads, eofs, etc.

  // // get the first line. There needs to be header info in the first line:
  // std::getline(inFile, line);

  // tokenizer tok_itr, tok_end;

  // // TODO: use a set in order to error on same column name
  // std::vector<std::string> colNames;

  // for (tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr != tok_end; ++tok_itr) {
  //   colNames.push_back(*tok_itr);
  // }

  // // Now iterate through the first line of data to determine types
  // std::getline(inFile, line);

  // int idx = 0;
  // for (idx = 0, tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr != tok_end; ++tok_itr, ++idx) {
  //   // TODO: what if there are not enough or too many tokens in this line?
  //   _columns.push_back(createDataColumnFromString(colNames[idx], *tok_itr));
  // }

  // // now get the rest of the data
  // int linecnt = 2;
  // while (std::getline(inFile, line)) {
  //   for (idx = 0, tok_itr = tokenizer(line.begin(), line.end(), sep, -1); tok_itr != tok_end; ++tok_itr, ++idx) {
  //     // TODO: what if there are not enough or too many tokens in this line?
  //     _columns[idx]->push_back(*tok_itr);
  //   }
  //   ++linecnt;

  //   if (linecnt % 5000 == 0) {
  //     std::cout << "line cnt update: " << linecnt << std::endl;
  //   }
  // }

  // inFile.close();
}

void DataTable::_readDataFromFile(const std::string& filename) {
  boost::filesystem::path p(filename);  // avoid repeated path construction below

  RUNTIME_EX_ASSERT(boost::filesystem::exists(p), "File: " + filename + " does not exist.");

  RUNTIME_EX_ASSERT(boost::filesystem::is_regular_file(p),
                    "File: " + filename + " is not a regular file. Cannot read contents to build a data table.");

  RUNTIME_EX_ASSERT(p.has_extension(),
                    "File: " + filename + " does not have an extension. Cannot read contents to build a data table.");

  std::string ext = p.extension().string();
  boost::to_lower(ext);

  if (ext == ".csv") {
    _readFromCsvFile(filename);
  } else {
    THROW_RUNTIME_EX("File: " + filename + " with extension \"" + ext + "\" is not a supported data file.");
  }
}

void DataTable::_buildColumnsFromJSONObj(const rapidjson::Value& obj,
                                         const rapidjson::Pointer& objPath,
                                         bool buildIdColumn) {
  RUNTIME_EX_ASSERT(obj.IsObject(),
                    "JSON data parse error - data must be an object. Cannot build data table from JSON.");

  rapidjson::Value::ConstMemberIterator mitr1, mitr2;

  bool isObject;

  if ((mitr1 = obj.FindMember("values")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT((isObject = mitr1->value.IsObject()) || mitr1->value.IsArray(),
                      "JSON data parse error - \"values\" property in the json must be an object or an array.");

    if (isObject) {
      // in column format
      // TODO: fill out
      THROW_RUNTIME_EX("JSON data parse error - column format not supported yet.");
    } else {
      // in row format in an array

      // TODO(croot) - should we just log a warning if no data is supplied instead?
      RUNTIME_EX_ASSERT(!mitr1->value.Empty(), "JSON data parse error - there is no data in defined.");

      const rapidjson::Value& item = mitr1->value[0];

      RUNTIME_EX_ASSERT(item.IsObject(),
                        "JSON data parse error - every row of JSON data must be defined as an object.");

      for (mitr2 = item.MemberBegin(); mitr2 != item.MemberEnd(); ++mitr2) {
        // TODO: Support strings? bools? Anything else?
        if (mitr2->value.IsNumber()) {
          _columns.push_back(createDataColumnFromRowMajorObj(mitr2->name.GetString(), mitr2->value, mitr1->value));
        } else if (mitr2->value.IsString()) {
          std::string val = mitr2->value.GetString();
          if (ColorRGBA::isColorString(val)) {
            _columns.push_back(
                createColorDataColumnFromRowMajorObj(mitr2->name.GetString(), mitr2->value, mitr1->value));
          }
        }
      }
    }
  } else if ((mitr1 = obj.FindMember("url")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr1->value.IsString(), "JSON data parse error - \"url\" property must be a string.");

    _readDataFromFile(mitr1->value.GetString());
  } else {
    THROW_RUNTIME_EX("JSON data parse error - JSON data object must contain either a \"values\" or \"url\" property.");
  }

  // TODO(croot) - throw a warning instead if no data?
  RUNTIME_EX_ASSERT(_columns.size(), "JSON data parse error - there are not columns in the data table.");
  _numRows = (*_columns.begin())->size();

  if (buildIdColumn) {
    TDataColumn<unsigned int>* idColumn = new TDataColumn<unsigned int>(defaultIdColumnName, _numRows);

    for (int i = 0; i < _numRows; ++i) {
      (*idColumn)[i] = i;
    }

    _columns.push_back(DataColumnUqPtr(idColumn));
  }
}

QueryDataType DataTable::getColumnType(const std::string& columnName) {
  ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = nameLookup.find(columnName)) != nameLookup.end(),
                    "DataTable::getColumnType(): column \"" + columnName + "\" does not exist.");

  return (*itr)->getColumnType();
}

DataColumnShPtr DataTable::getColumn(const std::string& columnName) {
  ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  RUNTIME_EX_ASSERT((itr = nameLookup.find(columnName)) != nameLookup.end(),
                    "DataTable::getColumn(): column \"" + columnName + "\" does not exist.");

  return *itr;
}

std::pair<GLBufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> DataTable::_createVBOData() {
  GLBufferLayoutShPtr vboLayoutPtr;
  QueryVertexBufferShPtr vbo;

  switch (_vboType) {
    case VboType::SEQUENTIAL: {
      GLSequentialBufferLayout* vboLayout = new GLSequentialBufferLayout();

      ColumnMap::iterator itr;
      int numBytes = 0;
      int numBytesPerItem = 0;

      // build up the layout of the vertex buffer
      for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
        switch ((*itr)->getColumnType()) {
          case QueryDataType::UINT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::UINT);
            break;
          case QueryDataType::INT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::INT);
            break;
          case QueryDataType::FLOAT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::FLOAT);
            break;
          case QueryDataType::DOUBLE:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::DOUBLE);
            break;
          case QueryDataType::COLOR:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::VEC4F);
            break;
          default:
            THROW_RUNTIME_EX("Column type for column \"" + (*itr)->columnName + "\" in data table \"" + _name +
                             "\" is not supported. Cannot build vertex buffer.");
            break;
        }

        TypelessColumnData data = (*itr)->getTypelessColumnData();
        numBytes += data.numItems * data.numBytesPerItem;
        numBytesPerItem += data.numBytesPerItem;
      }

      // now cpy the column data into one big buffer, sequentially, and
      // buffer it all to the gpu via the VBO.
      // char byteData[numBytes];
      std::unique_ptr<char[]> byteDataPtr(new char[numBytes]);
      char* byteData = byteDataPtr.get();

      // float byteData[numBytes/sizeof(float)];
      // memset(byteData, 0x0, numBytes);
      // memset(&byteData[0], 0x0, numBytes);
      memset(byteData, 0x0, numBytes);

      int startIdx = 0;
      for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
        TypelessColumnData data = (*itr)->getTypelessColumnData();
        memcpy(&byteData[startIdx], data.data, data.numItems * data.numBytesPerItem);

        startIdx += data.numItems * data.numBytesPerItem;
        // startIdx += data.numItems;
      }

      vboLayoutPtr.reset(vboLayout);

      return std::make_pair(vboLayoutPtr, std::make_pair(std::move(byteDataPtr), numBytesPerItem));

    } break;

    case VboType::INTERLEAVED: {
      GLInterleavedBufferLayout* vboLayout = new GLInterleavedBufferLayout();

      ColumnMap::iterator itr;
      int numBytes = 0;

      std::vector<TypelessColumnData> columnData(_columns.size());

      // build up the layout of the vertex buffer
      for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
        switch ((*itr)->getColumnType()) {
          case QueryDataType::UINT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::UINT);
            break;
          case QueryDataType::INT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::INT);
            break;
          case QueryDataType::FLOAT:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::FLOAT);
            break;
          case QueryDataType::DOUBLE:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::DOUBLE);
            break;
          case QueryDataType::COLOR:
            vboLayout->addAttribute((*itr)->columnName, GLBufferAttrType::VEC4F);
            break;
          default:
            THROW_RUNTIME_EX("Column type for column \"" + (*itr)->columnName + "\" in data table \"" + _name +
                             "\" is not supported. Cannot build vertex buffer.");
            break;
        }

        int idx = itr - _columns.begin();
        columnData[idx] = (*itr)->getTypelessColumnData();
        numBytes += columnData[idx].numItems * columnData[idx].numBytesPerItem;
      }

      // now cpy the column data into one big buffer, interleaving the data, and
      // buffer it all to the gpu via the VBO.
      // char byteData[numBytes];
      std::unique_ptr<char[]> byteDataPtr(new char[numBytes]);
      char* byteData = byteDataPtr.get();
      memset(byteData, 0x0, numBytes);

      int startIdx = 0;
      for (int i = 0; i < _numRows; ++i) {
        for (size_t j = 0; j < columnData.size(); ++j) {
          int bytesPerItem = columnData[j].numBytesPerItem;
          memcpy(&byteData[startIdx], static_cast<char*>(columnData[j].data) + (i * bytesPerItem), bytesPerItem);
          startIdx += bytesPerItem;
        }
      }

      vboLayoutPtr.reset(vboLayout);

      return std::make_pair(vboLayoutPtr, std::make_pair(std::move(byteDataPtr), vboLayout->getNumBytesPerVertex()));

    } break;
    case VboType::INDIVIDUAL:
      // TODO: What kind of data structure should we do in this case? Should we do
      // one big unordered map, but there's only one item in the SEQUENTIAL &
      // INTERLEAVED case?
      break;
  }

  return std::make_pair(nullptr, std::make_pair(nullptr, 0));
}

QueryVertexBufferShPtr DataTable::getColumnDataVBO(const GpuId& gpuId, const std::string& columnName) {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(), "Cannot get column data vbo for gpu " + std::to_string(gpuId));

  if (itr->second.vbo == nullptr) {
    CHECK(itr->second.qrmGpuData && itr->second.qrmGpuData->rendererPtr);
    GLRenderer* renderer = dynamic_cast<GLRenderer*>(itr->second.qrmGpuData->rendererPtr.get());
    CHECK(renderer != nullptr);

    std::pair<GLBufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> vboData = _createVBOData();

    itr->second.vbo.reset(new QueryVertexBuffer(renderer, vboData.first));
    itr->second.vbo->bufferData(vboData.second.first.get(), _numRows, vboData.second.second);
  }

  RUNTIME_EX_ASSERT(itr->second.vbo->hasAttribute(columnName),
                    "DataTable::getColumnVBO(): column \"" + columnName + "\" does not exist.");

  return itr->second.vbo;
}

std::map<GpuId, QueryVertexBufferShPtr> DataTable::getColumnDataVBOs(const std::string& columnName) {
  std::map<GpuId, QueryVertexBufferShPtr> rtn;
  std::pair<GLBufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> vboData;

  for (auto& itr : _perGpuData) {
    if (itr.second.vbo == nullptr) {
      itr.second.makeActiveOnCurrentThread();

      GLRenderer* renderer = dynamic_cast<GLRenderer*>(itr.second.qrmGpuData->rendererPtr.get());
      CHECK(renderer != nullptr);

      if (vboData.first == nullptr) {
        vboData = _createVBOData();
      }

      itr.second.vbo.reset(new QueryVertexBuffer(renderer, vboData.first));
      itr.second.vbo->bufferData(vboData.second.first.get(), _numRows, vboData.second.second);
    }

    rtn.insert({itr.first, itr.second.vbo});
  }

  return rtn;
}

void DataTable::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    // need to update the path in case the path has changed in the json,
    // but the data is the same.
    _jsonPath = objPath;
    return;
  }

  _jsonPath = objPath;

  THROW_RUNTIME_EX("Updating a JSON-embedded data table is not yet supported.");
}

template <typename T>
TDataColumn<T>::TDataColumn(const std::string& name, const rapidjson::Value& dataArrayObj, InitType initType)
    : DataColumn(name), _columnDataPtr(new std::vector<T>()) {
  if (initType == DataColumn::InitType::ROW_MAJOR) {
    _initFromRowMajorJSONObj(dataArrayObj);
  } else {
    _initFromColMajorJSONObj(dataArrayObj);
  }
}

template <>
void TDataColumn<ColorRGBA>::push_back(const std::string& val) {
  _columnDataPtr->push_back(ColorRGBA(val));
}

template <typename T>
void TDataColumn<T>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
  RUNTIME_EX_ASSERT(dataArrayObj.IsArray(), "JSON data parse error: Row-major data object is not an array.");

  rapidjson::Value::ConstValueIterator vitr;
  rapidjson::Value::ConstMemberIterator mitr;

  for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
    RUNTIME_EX_ASSERT(vitr->IsObject(),
                      "JSON data parse error: Item " + std::to_string(vitr - dataArrayObj.Begin()) +
                          "in data array must be an object for row-major-defined data.");
    RUNTIME_EX_ASSERT((mitr = vitr->FindMember(columnName.c_str())) != vitr->MemberEnd(),
                      "JSON data parse error: column \"" + columnName +
                          "\" does not exist in row-major-defined data item " +
                          std::to_string(vitr - dataArrayObj.Begin()));

    _columnDataPtr->push_back(RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value));
  }
}

template <>
void TDataColumn<ColorRGBA>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
  RUNTIME_EX_ASSERT(dataArrayObj.IsArray(), "JSON data parse error: Row-major data object is not an array.");

  rapidjson::Value::ConstValueIterator vitr;
  rapidjson::Value::ConstMemberIterator mitr;

  ColorRGBA color;

  for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
    RUNTIME_EX_ASSERT(vitr->IsObject(),
                      "JSON data parse error: Item " + std::to_string(vitr - dataArrayObj.Begin()) +
                          "in data array must be an object for row-major-defined data.");
    RUNTIME_EX_ASSERT((mitr = vitr->FindMember(columnName.c_str())) != vitr->MemberEnd(),
                      "JSON data parse error: column \"" + columnName +
                          "\" does not exist in row-major-defined data item " +
                          std::to_string(vitr - dataArrayObj.Begin()));

    _columnDataPtr->push_back(color.initFromCSSString(mitr->value.GetString()));
  }
}

template <typename T>
void TDataColumn<T>::_initFromColMajorJSONObj(const rapidjson::Value& dataArrayObj) {
}

// template <typename T>
// std::pair<T, T> TDataColumn<T>::getExtrema() {
//     T min, max;

//     return std::make_pair(min, max);
// }

}  // namespace QueryRenderer
