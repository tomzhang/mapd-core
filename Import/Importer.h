/*
 * @file Importer.h
 * @author Wei Hong < wei@mapd.com>
 * @brief Importer class for table import from file
 */
#ifndef _IMPORTER_H_
#define _IMPORTER_H_

#include <string>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <boost/noncopyable.hpp>
#include <glog/logging.h>
#include "../Catalog/TableDescriptor.h"
#include "../Catalog/Catalog.h"
#include "../Fragmenter/Fragmenter.h"
#include "../StringDictionary/StringDictionary.h"

namespace Importer_NS {

class TypedImportBuffer : boost::noncopyable {
public:
  TypedImportBuffer(
    const ColumnDescriptor* col_desc,
    StringDictionary* string_dict)
    : column_desc_(col_desc),
    string_dict_(string_dict) {
    switch (col_desc->columnType.get_type()) {
    case kBOOLEAN:
      bool_buffer_ = new std::vector<int8_t>();
      break;
    case kSMALLINT:
      smallint_buffer_ = new std::vector<int16_t>();
      break;
    case kINT:
      int_buffer_ = new std::vector<int32_t>();
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      bigint_buffer_ = new std::vector<int64_t>();
      break;
    case kFLOAT:
      float_buffer_ = new std::vector<float>();
      break;
    case kDOUBLE:
      double_buffer_ = new std::vector<double>();
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      string_buffer_ = new std::vector<std::string>();
      if (col_desc->columnType.get_compression() == kENCODING_DICT) {
        string_dict_buffer_ = new std::vector<int32_t>();
      } else if (col_desc->columnType.get_compression() == kENCODING_TOKDICT)
        CHECK(false);
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      time_buffer_ = new std::vector<time_t>();
      break;
    default:
      CHECK(false);
    }
  }

  ~TypedImportBuffer() {
    switch (column_desc_->columnType.get_type()) {
    case kBOOLEAN:
      delete bool_buffer_;
      break;
    case kSMALLINT:
      delete smallint_buffer_;
      break;
    case kINT:
      delete int_buffer_;
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      delete bigint_buffer_;
      break;
    case kFLOAT:
      delete float_buffer_;
      break;
    case kDOUBLE:
      delete double_buffer_;
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      delete string_buffer_;
      if (column_desc_->columnType.get_compression() == kENCODING_DICT) {
        delete string_dict_buffer_;
      }
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      delete time_buffer_;
      break;
    default:
      CHECK(false);
    }
  }

  void addBoolean(const int8_t v) {
    bool_buffer_->push_back(v);
  }

  void addSmallint(const int16_t v) {
    smallint_buffer_->push_back(v);
  }

  void addInt(const int32_t v) {
    int_buffer_->push_back(v);
  }

  void addBigint(const int64_t v) {
    bigint_buffer_->push_back(v);
  }

  void addFloat(const float v) {
    float_buffer_->push_back(v);
  }

  void addDouble(const double v) {
    double_buffer_->push_back(v);
  }

  void addString(const std::string& v) {
    string_buffer_->push_back(v);
  }

  void addTime(const time_t v) {
    time_buffer_->push_back(v);
  }

  void addDictEncodedString(const std::vector<std::string> &stringVec) {
    CHECK(string_dict_);
    string_dict_->addBulk(stringVec, *string_dict_buffer_);
  }

  const SQLTypeInfo &getTypeInfo() const {
    return column_desc_->columnType;
  }

  const ColumnDescriptor *getColumnDesc() const {
    return column_desc_;
  }

  int8_t* getAsBytes() const {
    switch (column_desc_->columnType.get_type()) {
    case kBOOLEAN:
      return reinterpret_cast<int8_t*>(&((*bool_buffer_)[0]));
    case kSMALLINT:
      return reinterpret_cast<int8_t*>(&((*smallint_buffer_)[0]));
    case kINT:
      return reinterpret_cast<int8_t*>(&((*int_buffer_)[0]));
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      return reinterpret_cast<int8_t*>(&((*bigint_buffer_)[0]));
    case kFLOAT:
      return reinterpret_cast<int8_t*>(&((*float_buffer_)[0]));
    case kDOUBLE:
      return reinterpret_cast<int8_t*>(&((*double_buffer_)[0]));
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return reinterpret_cast<int8_t*>(&((*time_buffer_)[0]));
    default:
      CHECK(false);
    }
  }

  std::vector<std::string>* getStringBuffer() const {
    return string_buffer_;
  }

  int8_t* getStringDictBuffer() const {
    return reinterpret_cast<int8_t*>(&((*string_dict_buffer_)[0]));
  }

  bool stringDictCheckpoint() {
    if (string_dict_ == nullptr)
      return true;
    return string_dict_->checkpoint();
  }

  void clear() {
    switch (column_desc_->columnType.get_type()) {
    case kBOOLEAN: {
      bool_buffer_->clear();
      break;
    }
    case kSMALLINT: {
      smallint_buffer_->clear();
      break;
    }
    case kINT: {
      int_buffer_->clear();
      break;
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
      bigint_buffer_->clear();
      break;
    }
    case kFLOAT: {
      float_buffer_->clear();
      break;
    }
    case kDOUBLE: {
      double_buffer_->clear();
      break;
    }
    case kTEXT: 
    case kVARCHAR:
    case kCHAR: {
      string_buffer_->clear();
      if (column_desc_->columnType.get_compression() == kENCODING_DICT) {
        string_dict_buffer_->clear();
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      time_buffer_->clear();
      break;
    }
    default:
      CHECK(false);
    }
  }
private:
  union {
    std::vector<int8_t>* bool_buffer_;
    std::vector<int16_t>* smallint_buffer_;
    std::vector<int32_t>* int_buffer_;
    std::vector<int64_t>* bigint_buffer_;
    std::vector<float>* float_buffer_;
    std::vector<double>* double_buffer_;
    std::vector<time_t>* time_buffer_;
    std::vector<std::string>* string_buffer_;
  };
  std::vector<int32_t>* string_dict_buffer_;
  const ColumnDescriptor *column_desc_;
  StringDictionary *string_dict_;
};

struct CopyParams {
  char delimiter;
  std::string null_str;
  bool has_header;
  bool quoted; // does the input have any quoted fields, default to false
  char quote;
  char escape;
  char line_delim;
  int threads;

  CopyParams() : delimiter(','), null_str("\\N"), has_header(true), quoted(false), quote('"'), escape('"'), line_delim('\n'), threads(0) {}
};

class Loader {
  public:
    Loader(const Catalog_Namespace::Catalog &c, const TableDescriptor *t) : catalog(c), table_desc(t), column_descs(c.getAllColumnMetadataForTable(t->tableId)), load_failed(false) {};
    void init();
    const Catalog_Namespace::Catalog &get_catalog() const { return catalog; }
    const TableDescriptor *get_table_desc() const { return table_desc; }
    const std::list<const ColumnDescriptor *> &get_column_descs() const { return column_descs; }
    const Fragmenter_Namespace::InsertData &get_insert_data() const { return insert_data; }
    StringDictionary *get_string_dict(const ColumnDescriptor *cd) const { 
      if (!cd->columnType.is_string() || cd->columnType.get_compression() != kENCODING_DICT)
        return nullptr;
      return dict_map.at(cd->columnId);
    }
    void load(const std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers, size_t row_count);
    bool get_load_failed() const { return load_failed; }
    void set_load_failed(bool f) { load_failed = f; }
  private:
    const Catalog_Namespace::Catalog &catalog;
    const TableDescriptor *table_desc;
    std::list <const ColumnDescriptor *> column_descs;
    Fragmenter_Namespace::InsertData insert_data;
    std::map<int, StringDictionary*> dict_map;
    bool load_failed;
};

class Importer {
  public:
    Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p);
    ~Importer();
    void import();
    const CopyParams &get_copy_params() const { return copy_params; }
    const std::list<const ColumnDescriptor *> &get_column_descs() const { return loader.get_column_descs(); }
    void load(const std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers, size_t row_count) { loader.load(import_buffers, row_count); }
    std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> &get_import_buffers_vec() { return import_buffers_vec; }
    std::vector<std::unique_ptr<TypedImportBuffer>> &get_import_buffers(int i) { return import_buffers_vec[i]; }
  private:
    const std::string &file_path;
    const CopyParams &copy_params;
    size_t file_size;
    int max_threads;
    FILE *p_file;
    char *buffer[2];
    int which_buf;
    std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
    Loader loader;
};

};
#endif // _IMPORTER_H_
