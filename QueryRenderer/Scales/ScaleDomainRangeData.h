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
  BaseScaleDomainRangeData(const QueryRendererContextShPtr& ctx, const std::string& name, bool useString = false)
      : _ctx(ctx), _name(name), _useString(useString) {}
  virtual ~BaseScaleDomainRangeData() {}
  virtual int size() const = 0;
  virtual QueryDataType getType() const = 0;
  virtual const ::Rendering::GL::TypeGLShPtr& getTypeGL() = 0;
  virtual const std::type_info& getTypeInfo() const = 0;

  std::string getName() { return _name; }
  bool useString() { return _useString; }

  virtual operator std::string() const = 0;

 protected:
  QueryRendererContextShPtr _ctx;
  std::string _name;
  bool _useString;
};

template <typename T>
class ScaleDomainRangeData : public BaseScaleDomainRangeData {
 public:
  static const QueryDataType dataType;
  ScaleDomainRangeData(const QueryRendererContextShPtr& ctx, const std::string& name, bool useString = false)
      : BaseScaleDomainRangeData(ctx, name, useString), _vectorPtr(nullptr), _cachedTypeGL(nullptr) {}
  ScaleDomainRangeData(const QueryRendererContextShPtr& ctx,
                       const std::string& name,
                       size_t size,
                       bool useString = false)
      : BaseScaleDomainRangeData(ctx, name, useString), _vectorPtr(new std::vector<T>(size)), _cachedTypeGL(nullptr) {}
  ~ScaleDomainRangeData() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             ScaleType type,
                             bool& sizeChanged,
                             bool& valsChanged,
                             std::function<void(T&)> valConvert) {
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
                          _ctx->getUserWidgetIds(), obj, "scale objects must have a \"" + _name + "\" property."));

    if (!_ctx->isJSONCacheUpToDate(_jsonPath, mitr->value)) {
      if (_useString) {
        RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
                           (mitr->value.IsArray() && mitr->value.Size())),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
      } else {
        RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
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
                _ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"data\" string property."));
        tablePtr = _ctx->getDataTable(mitr->value.GetString());

        RUNTIME_EX_ASSERT(
            (mitr = jsonObj.FindMember("field")) != jsonObj.MemberEnd() && mitr->value.IsString(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"field\" string property."));

        // Only supports hand-written data right now.
        // TODO(croot): Support query result vbo -- this is somewhat
        // tricky, because in order to do so we'd have to use compute
        // shaders/cuda to do min/max/other math stuff, and uniform buffers or
        // storage buffers to send the values as uniforms

        // TODO(croot): if we support query result vbos, then we need to set
        // up callbacks for data replace/update/removes
        RUNTIME_EX_ASSERT(tablePtr->getType() == QueryDataTableType::EMBEDDED,
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
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
              RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                   jsonObj,
                                                   "Unsupported data reference table type. Data reference is not a "
                                                   "vertex or poly buffer-based data table."));

          columnPtr = polyDataTablePtr->getColumn(columnName);
        }

        auto dataColumnPtr = dynamic_cast<TDataColumn<T>*>(columnPtr.get());

        RUNTIME_EX_ASSERT(dataColumnPtr != nullptr,
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
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
          _setItem(idx, *vitr, valConvert);
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
  std::vector<T> getVectorData() const { return *_vectorPtr; }

  QueryDataType getType() const final { return dataType; }
  static T getNullValue() {
    THROW_RUNTIME_EX("Nulls of domain/range type " + to_string(dataType) + " are not currently supported.");
    return T();
  }

  const ::Rendering::GL::TypeGLShPtr& getTypeGL() {
    if (!_cachedTypeGL) {
      auto rootGpuCache = _ctx->getRootGpuCache();
      CHECK(rootGpuCache);
      _cachedTypeGL.reset(new ::Rendering::GL::TypeGL<T, 1>(rootGpuCache->supportedExtensions));
    }
    return _cachedTypeGL;
  }

  inline const std::type_info& getTypeInfo() const { return typeid(T); }

  static T getDataValueFromJSONObj(const rapidjson::Value& obj) { return RapidJSONUtils::getNumValFromJSONObj<T>(obj); }

  operator std::string() const { return "ScaleDomainRangeData<" + std::string(typeid(T).name()) + ">(" + _name + ")"; }

 private:
  std::shared_ptr<std::vector<T>> _vectorPtr;
  ::Rendering::GL::TypeGLShPtr _cachedTypeGL;
  rapidjson::Pointer _jsonPath;

  void _pushItem(const rapidjson::Value& obj) { _vectorPtr->push_back(getDataValueFromJSONObj(obj)); }

  void _setItem(size_t idx, const rapidjson::Value& obj, std::function<void(T&)> valConvertFunc) {
    (*_vectorPtr)[idx] = getDataValueFromJSONObj(obj);
    valConvertFunc((*_vectorPtr)[idx]);
  }

  void _setFromStringValue(const std::string& strVal, ScaleType type) {
    if (isQuantitativeScale(type) && (strVal == "width" || strVal == "height")) {
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

/*
 * RGBA specializations
 */
template <>
double ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>::getTypeGL();

template <>
::Rendering::Colors::ColorRGBA ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>::_setFromStringValue(const std::string& strVal,
                                                                               ScaleType type);
template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>::_updateVectorDataByType(
    TDataColumn<::Rendering::Colors::ColorRGBA>* dataColumnPtr,
    ScaleType type);

/*
 * HSL specializations
 */
template <>
double ScaleDomainRangeData<::Rendering::Colors::ColorHSL>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Colors::ColorHSL>::getTypeGL();

template <>
::Rendering::Colors::ColorHSL ScaleDomainRangeData<::Rendering::Colors::ColorHSL>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorHSL>::_setFromStringValue(const std::string& strVal,
                                                                              ScaleType type);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorHSL>::_updateVectorDataByType(
    TDataColumn<::Rendering::Colors::ColorHSL>* dataColumnPtr,
    ScaleType type);

/*
 * LAB specializations
 */
template <>
double ScaleDomainRangeData<::Rendering::Colors::ColorLAB>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Colors::ColorLAB>::getTypeGL();

template <>
::Rendering::Colors::ColorLAB ScaleDomainRangeData<::Rendering::Colors::ColorLAB>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorLAB>::_setFromStringValue(const std::string& strVal,
                                                                              ScaleType type);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorLAB>::_updateVectorDataByType(
    TDataColumn<::Rendering::Colors::ColorLAB>* dataColumnPtr,
    ScaleType type);

/*
 * HCL specializations
 */
template <>
double ScaleDomainRangeData<::Rendering::Colors::ColorHCL>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Colors::ColorHCL>::getTypeGL();

template <>
::Rendering::Colors::ColorHCL ScaleDomainRangeData<::Rendering::Colors::ColorHCL>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorHCL>::_setFromStringValue(const std::string& strVal,
                                                                              ScaleType type);

template <>
void ScaleDomainRangeData<::Rendering::Colors::ColorHCL>::_updateVectorDataByType(
    TDataColumn<::Rendering::Colors::ColorHCL>* dataColumnPtr,
    ScaleType type);

/*
 * string specializations
 */
template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const;

template <>
const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL();

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, ScaleType type);

/*
 * int specializations
 */
template <>
int ScaleDomainRangeData<int>::getNullValue();

/*
 * float specializations
 */
template <>
float ScaleDomainRangeData<float>::getNullValue();

/*
 * double specializations
 */
template <>
double ScaleDomainRangeData<double>::getNullValue();

/*
 * int64 specializations
 */
template <>
int64_t ScaleDomainRangeData<int64_t>::getNullValue();

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALEDOMAINRANGEDATA_H_
