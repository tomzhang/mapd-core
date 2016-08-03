#ifndef QUERYRENDERER_SCALES_SCALEDOMAINRANGEDATA_H_
#define QUERYRENDERER_SCALES_SCALEDOMAINRANGEDATA_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "../Data/QueryDataTable.h"
#include "../QueryRendererContext.h"
#include <Rendering/Renderer/GL/TypeGL.h>

#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

class BaseScaleDomainRangeData {
 public:
  BaseScaleDomainRangeData(const std::string& name, bool useString = false) : _name(name), _useString(useString) {}
  virtual ~BaseScaleDomainRangeData() {}
  virtual int size() const = 0;
  virtual QueryDataType getType() const = 0;
  virtual const ::Rendering::GL::TypeGLShPtr& getTypeGL() = 0;
  virtual const std::type_info& getTypeInfo() = 0;

  std::string getName() { return _name; }
  bool useString() { return _useString; }

  virtual operator std::string() const = 0;

 protected:
  std::string _name;
  bool _useString;
};

template <typename T>
class ScaleDomainRangeData : public BaseScaleDomainRangeData {
 public:
  static const QueryDataType dataType;
  ScaleDomainRangeData(const std::string& name, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(nullptr), _cachedTypeGL(nullptr) {}
  ScaleDomainRangeData(const std::string& name, size_t size, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(new std::vector<T>(size)), _cachedTypeGL(nullptr) {}
  ~ScaleDomainRangeData() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryRendererContextShPtr& ctx,
                             ScaleType type,
                             bool& sizeChanged,
                             bool& valsChanged) {
    sizeChanged = false;
    valsChanged = false;

    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    bool isObject = false, isString = false;
    QueryDataTableShPtr tablePtr;
    DataColumnShPtr columnPtr;

    mitr = obj.FindMember(_name.c_str());
    RUNTIME_EX_ASSERT(mitr != obj.MemberEnd(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          ctx->getUserWidgetIds(), obj, "scale objects must have a \"" + _name + "\" property."));

    if (!ctx->isJSONCacheUpToDate(_jsonPath, mitr->value)) {
      if (_useString) {
        RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
                           (mitr->value.IsArray() && mitr->value.Size())),
                          RapidJSONUtils::getJsonParseErrorStr(
                              ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
      } else {
        RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())),
                          RapidJSONUtils::getJsonParseErrorStr(
                              ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
      }

      const rapidjson::Value& jsonObj = mitr->value;

      int prevSz = 0;
      if (_vectorPtr) {
        prevSz = _vectorPtr->size();
      }
      if (isObject) {
        RUNTIME_EX_ASSERT(
            (mitr = jsonObj.FindMember("data")) != jsonObj.MemberEnd() && mitr->value.IsString(),
            RapidJSONUtils::getJsonParseErrorStr(
                ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"data\" string property."));
        tablePtr = ctx->getDataTable(mitr->value.GetString());

        RUNTIME_EX_ASSERT(
            (mitr = jsonObj.FindMember("field")) != jsonObj.MemberEnd() && mitr->value.IsString(),
            RapidJSONUtils::getJsonParseErrorStr(
                ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"field\" string property."));

        // Only supports hand-written data right now.
        // TODO(croot): Support query result vbo -- this is somewhat
        // tricky, because in order to do so we'd have to use compute
        // shaders/cuda to do min/max/other math stuff, and uniform buffers or
        // storage buffers to send the values as uniforms
        RUNTIME_EX_ASSERT(tablePtr->getType() == QueryDataTableType::EMBEDDED,
                          RapidJSONUtils::getJsonParseErrorStr(
                              ctx->getUserWidgetIds(),
                              jsonObj,
                              "scale data references only support JSON-embedded data tables currently."));

        auto dataTablePtr = dynamic_cast<DataTable*>(tablePtr.get());
        std::string columnName = mitr->value.GetString();
        if (dataTablePtr) {
          columnPtr = dataTablePtr->getColumn(columnName);
        } else {
          auto polyDataTablePtr = dynamic_cast<PolyDataTable*>(tablePtr.get());

          RUNTIME_EX_ASSERT(
              polyDataTablePtr != nullptr,
              RapidJSONUtils::getJsonParseErrorStr(ctx->getUserWidgetIds(),
                                                   jsonObj,
                                                   "Unsupported data reference table type. Data reference is not a "
                                                   "vertex or poly buffer-based data table."));

          columnPtr = polyDataTablePtr->getColumn(columnName);
        }

        auto dataColumnPtr = dynamic_cast<TDataColumn<T>*>(columnPtr.get());

        RUNTIME_EX_ASSERT(dataColumnPtr != nullptr,
                          RapidJSONUtils::getJsonParseErrorStr(
                              ctx->getUserWidgetIds(),
                              jsonObj,
                              "Data column " + columnName + " is of type " + to_string(columnPtr->getColumnType()) +
                                  " is not compatible with domain/range data of type " + to_string(getType())));

        _vectorPtr = dataColumnPtr->getColumnData();

        _updateVectorDataByType(dataColumnPtr, type);
      } else if (isString) {
        _setFromStringValue(mitr->value.GetString(), type);
      } else {
        _vectorPtr.reset(new std::vector<T>(jsonObj.Capacity()));

        // gather all the items
        size_t idx = 0;
        for (vitr = jsonObj.Begin(); vitr != jsonObj.End(); ++vitr, ++idx) {
          _setItem(idx, *vitr);
        }
      }

      int newSz = 0;
      if (_vectorPtr) {
        newSz = _vectorPtr->size();
      }

      // only need to regenerate the shader if the size of the
      // scale data changed, but we need to update any coerced
      // values if the scale's values changed, which means
      // we need to distinguish between when the size of the
      // data changed vs only the values changed.
      valsChanged = true;
      sizeChanged = (prevSz != newSz);
    }

    updateJSONPath(objPath);
  }

  void updateJSONPath(const rapidjson::Pointer& objPath) { _jsonPath = objPath.Append(_name.c_str(), _name.length()); }

  int size() const { return (_vectorPtr == nullptr ? 0 : _vectorPtr->size()); }

  double getDifference(const double divisor = 1) const {
    int sz = size();
    RUNTIME_EX_ASSERT(sz > 0, "Cannot get difference from an empty domain/range.");
    return double((*_vectorPtr)[sz - 1] - (*_vectorPtr)[0]) / divisor;
  }

  std::vector<T>& getVectorDataRef() { return *_vectorPtr; }
  std::vector<T> getVectorData() { return *_vectorPtr; }

  QueryDataType getType() const final { return dataType; }

  const ::Rendering::GL::TypeGLShPtr& getTypeGL() {
    if (!_cachedTypeGL) {
      _cachedTypeGL.reset(new ::Rendering::GL::TypeGL<T>());
    }
    return _cachedTypeGL;
  }

  inline const std::type_info& getTypeInfo() { return typeid(T); }

  static T getDataValueFromJSONObj(const rapidjson::Value& obj);

  operator std::string() const { return "ScaleDomainRangeData<T>(name: " + _name + ")"; }

  std::string getTypeStr() const;

 private:
  std::shared_ptr<std::vector<T>> _vectorPtr;
  ::Rendering::GL::TypeGLShPtr _cachedTypeGL;
  rapidjson::Pointer _jsonPath;

  void _pushItem(const rapidjson::Value& obj) { _vectorPtr->push_back(getDataValueFromJSONObj(obj)); }

  void _setItem(size_t idx, const rapidjson::Value& obj) {
    (*_vectorPtr)[idx] = static_cast<T>(getDataValueFromJSONObj(obj));
  }

  void _setFromStringValue(const std::string& strVal, ScaleType type) {
    if (type == ScaleType::LINEAR && (strVal == "width" || strVal == "height")) {
      // TODO(croot): Handle actual width/height vals
      _vectorPtr.reset(new std::vector<T>({0, 1}));

      // need to put the width/height range from -1 to 1. Didn't do that
      // in the previous line because the compiler was complaining that
      // you cannot convert -1 to an unsigned int, tho if range is
      // an unsigned int, you should never get here. Doing a subtraction here
      // to get a -1 to 1 range without the compilation error.

      (*_vectorPtr)[0] -= 1;
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                            to_string(type) + " for domain/range is invalid."));
    }
  }

  void _updateVectorDataByType(TDataColumn<T>* dataColumnPtr, ScaleType type) {
    if (type == ScaleType::LINEAR || type == ScaleType::QUANTIZE) {
      std::pair<T, T> minmaxDomain = dataColumnPtr->getExtrema();

      _vectorPtr.reset(new std::vector<T>({minmaxDomain.first, minmaxDomain.second}));
    }
  }
};

template <>
double ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getDifference(const double divisor) const;

template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getTypeGL();

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL();

template <>
unsigned int ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
int ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
float ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
double ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
::Rendering::Objects::ColorRGBA ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::_setFromStringValue(const std::string& strVal,
                                                                                ScaleType type);

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, ScaleType type);

template <>
void ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::_updateVectorDataByType(
    TDataColumn<::Rendering::Objects::ColorRGBA>* dataColumnPtr,
    ScaleType type);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALEDOMAINRANGEDATA_H_
