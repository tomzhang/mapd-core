#include "DataTable.h"
#include "RapidJSONUtils.h"
#include <limits>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <regex>

using namespace MapD_Renderer;

DataColumnUqPtr createDataColumnFromRowMajorObj(const std::string& columnName,
                                                const rapidjson::Value& rowItem,
                                                const rapidjson::Value& dataArray) {
  if (rowItem.IsInt()) {
    return DataColumnUqPtr(new TDataColumn<int>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  } else if (rowItem.IsUint()) {
    return DataColumnUqPtr(new TDataColumn<unsigned int>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  } else if (rowItem.IsDouble()) {
    double val = rowItem.GetDouble();
    if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
      return DataColumnUqPtr(new TDataColumn<float>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    } else {
      return DataColumnUqPtr(new TDataColumn<double>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    }
  } else {
    assert(false);
  }
}

DataColumnUqPtr createColorDataColumnFromRowMajorObj(const std::string& columnName,
                                                     const rapidjson::Value& rowItem,
                                                     const rapidjson::Value& dataArray) {
  assert(rowItem.IsString());

  return DataColumnUqPtr(new TDataColumn<ColorRGBA>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
}

const std::string DataTable::defaultIdColumnName = "__id__";

DataTable::DataTable(const rapidjson::Value& obj, bool buildIdColumn, VboType vboType)
    : _vboType(vboType), _numRows(0) {
  _buildColumnsFromJSONObj(obj, buildIdColumn);
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

  // TODO: throw exceptions instead of asserts
  assert(boost::filesystem::exists(p));

  assert(boost::filesystem::is_regular_file(p));

  assert(p.has_extension());

  std::string ext = p.extension().string();
  boost::to_lower(ext);

  if (ext == ".csv") {
    _readFromCsvFile(filename);
  } else {
    assert(false);
  }
}

void DataTable::_buildColumnsFromJSONObj(const rapidjson::Value& obj, bool buildIdColumn) {
  // TODO: Throw exception in place of asserts
  assert(obj.IsObject());

  rapidjson::Value::ConstMemberIterator mitr1, mitr2;
  assert((mitr1 = obj.FindMember("name")) != obj.MemberEnd() && mitr1->value.IsString());
  _name = mitr1->value.GetString();

  bool isObject;

  if ((mitr1 = obj.FindMember("values")) != obj.MemberEnd()) {
    assert((isObject = mitr1->value.IsObject()) || mitr1->value.IsArray());

    if (isObject) {
      // in column format
      // TODO: fill out
    } else {
      // in row format in an array
      assert(!mitr1->value.Empty());

      const rapidjson::Value& item = mitr1->value[0];

      assert(item.IsObject());

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
    assert(mitr1->value.IsString());

    _readDataFromFile(mitr1->value.GetString());
  } else {
    assert(false);
  }

  assert(_columns.size());
  _numRows = (*_columns.begin())->size();

  if (buildIdColumn) {
    TDataColumn<unsigned int>* idColumn = new TDataColumn<unsigned int>(defaultIdColumnName, _numRows);

    for (int i = 0; i < _numRows; ++i) {
      (*idColumn)[i] = i + 1;
    }

    _columns.push_back(DataColumnUqPtr(idColumn));
  }
}

DataType DataTable::getColumnType(const std::string& columnName) {
  // TODO: throw exception for assert
  ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  assert((itr = nameLookup.find(columnName)) != nameLookup.end());

  return (*itr)->getColumnType();
}

DataColumnShPtr DataTable::getColumn(const std::string& columnName) {
  ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();

  ColumnMap_by_name::iterator itr;
  assert((itr = nameLookup.find(columnName)) != nameLookup.end());

  return *itr;
}

VertexBufferShPtr DataTable::getColumnDataVBO(const std::string& columnName) {
  if (_vbo == nullptr) {
    VertexBuffer::BufferLayoutShPtr vboLayoutPtr;
    switch (_vboType) {
      case VboType::SEQUENTIAL: {
        SequentialBufferLayout* vboLayout = new SequentialBufferLayout();

        ColumnMap::iterator itr;
        int numBytes = 0;
        int numBytesPerItem = 0;

        // build up the layout of the vertex buffer
        for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
          switch ((*itr)->getColumnType()) {
            case DataType::UINT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::UINT);
              break;
            case DataType::INT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::INT);
              break;
            case DataType::FLOAT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::FLOAT);
              break;
            case DataType::DOUBLE:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::DOUBLE);
              break;
            case DataType::COLOR:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::VEC4F);
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
        // // vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout *>(vboLayout));

        VertexBuffer* vbo = new VertexBuffer(vboLayoutPtr);
        vbo->bufferData(byteData, _numRows, numBytesPerItem);
        _vbo.reset(vbo);

      } break;

      case VboType::INTERLEAVED: {
        InterleavedBufferLayout* vboLayout = new InterleavedBufferLayout();

        ColumnMap::iterator itr;
        int numBytes = 0;

        std::vector<TypelessColumnData> columnData(_columns.size());

        // build up the layout of the vertex buffer
        for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
          switch ((*itr)->getColumnType()) {
            case DataType::UINT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::UINT);
              break;
            case DataType::INT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::INT);
              break;
            case DataType::FLOAT:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::FLOAT);
              break;
            case DataType::DOUBLE:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::DOUBLE);
              break;
            case DataType::COLOR:
              vboLayout->addAttribute((*itr)->columnName, BufferAttrType::VEC4F);
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
        for (size_t i = 0; i < _numRows; ++i) {
          for (size_t j = 0; j < columnData.size(); ++j) {
            int bytesPerItem = columnData[j].numBytesPerItem;
            memcpy(&byteData[startIdx], static_cast<char*>(columnData[j].data) + (i * bytesPerItem), bytesPerItem);
            startIdx += bytesPerItem;
          }
        }

        vboLayoutPtr.reset(vboLayout);
        VertexBuffer* vbo = new VertexBuffer(vboLayoutPtr);
        vbo->bufferData(byteData, _numRows, vboLayout->getBytesPerVertex());
        _vbo.reset(vbo);

      } break;
      case VboType::INDIVIDUAL:
        // TODO: What kind of data structure should we do in this case? Should we do
        // one big unordered map, but there's only one item in the SEQUENTIAL &
        // INTERLEAVED case?
        break;
    }
  }

  assert(_vbo->hasAttribute(columnName));

  return _vbo;
}

template <typename T>
TDataColumn<T>::TDataColumn(const std::string& name, int size)
    : DataColumn(name), _columnDataPtr(new std::vector<T>(size)) {
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

template <typename T>
void TDataColumn<T>::push_back(const std::string& val) {
  // TODO: this would throw a boost::bad_lexical_cast error
  // if the conversion can't be done.. I may need to throw
  // a MapD-compliant exception
  _columnDataPtr->push_back(boost::lexical_cast<T>(val));
}

template <>
void TDataColumn<ColorRGBA>::push_back(const std::string& val) {
  _columnDataPtr->push_back(ColorRGBA(val));
}

template <typename T>
void TDataColumn<T>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
  assert(dataArrayObj.IsArray());

  rapidjson::Value::ConstValueIterator vitr;
  rapidjson::Value::ConstMemberIterator mitr;

  for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
    assert(vitr->IsObject());
    assert((mitr = vitr->FindMember(columnName.c_str())) != vitr->MemberEnd());

    _columnDataPtr->push_back(getNumValFromJSONObj<T>(mitr->value));
  }
}

template <>
void TDataColumn<ColorRGBA>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
  assert(dataArrayObj.IsArray());

  rapidjson::Value::ConstValueIterator vitr;
  rapidjson::Value::ConstMemberIterator mitr;

  ColorRGBA color;

  for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
    assert(vitr->IsObject());
    assert((mitr = vitr->FindMember(columnName.c_str())) != vitr->MemberEnd());

    _columnDataPtr->push_back(color.initFromCSSString(mitr->value.GetString()));
    // _columnDataPtr->push_back(color);
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

template <>
DataType TDataColumn<unsigned int>::getColumnType() {
  return DataType::UINT;
}

template <>
DataType TDataColumn<int>::getColumnType() {
  return DataType::INT;
}

template <>
DataType TDataColumn<float>::getColumnType() {
  return DataType::FLOAT;
}

template <>
DataType TDataColumn<double>::getColumnType() {
  return DataType::DOUBLE;
}

template <>
DataType TDataColumn<ColorRGBA>::getColumnType() {
  return DataType::COLOR;
}
