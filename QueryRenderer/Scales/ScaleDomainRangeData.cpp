#include "ScaleDomainRangeData.h"
#include "Scale.h"
#include <Rendering/Colors/ColorUnion.h>
#include <Shared/sqltypes.h>

namespace QueryRenderer {

using ::Rendering::GL::TypeGLShPtr;
using ::Rendering::Colors::ColorRGBA;
using ::Rendering::Colors::ColorHSL;
using ::Rendering::Colors::ColorLAB;
using ::Rendering::Colors::ColorHCL;
using ::Rendering::Colors::ColorUnion;

/*
 * RGBA specializations
 */
template <>
double ScaleDomainRangeData<ColorRGBA>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Mathematical operations on color objects aren't currently supported.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
  if (!_cachedTypeGL) {
    auto rootGpuCache = _ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    _cachedTypeGL = ColorRGBA::getTypeGLPtr(rootGpuCache->supportedExtensions);
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<ColorRGBA>::dataType = QueryDataType::COLOR;

template <>
ColorRGBA ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorRGBA(ColorUnion(obj.GetString()));
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " RGBA color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA>* dataColumnPtr, ScaleType type) {
  if (type == ScaleType::LINEAR) {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of RGBA colors in the domain/range is unsupported for scale type" + to_string(type) +
        "."));
  }
}

/*
 * HSL specializations
 */
template <>
double ScaleDomainRangeData<ColorHSL>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Mathematical operations on color objects aren't currently supported.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<ColorHSL>::getTypeGL() {
  if (!_cachedTypeGL) {
    auto rootGpuCache = _ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    _cachedTypeGL = ColorHSL::getTypeGLPtr(rootGpuCache->supportedExtensions);
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<ColorHSL>::dataType = QueryDataType::COLOR;

template <>
ColorHSL ScaleDomainRangeData<ColorHSL>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorHSL(ColorUnion(obj.GetString()));
}

template <>
void ScaleDomainRangeData<ColorHSL>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " HSL color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<ColorHSL>::_updateVectorDataByType(TDataColumn<ColorHSL>* dataColumnPtr, ScaleType type) {
  if (type == ScaleType::LINEAR) {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of HSL colors in the domain/range is unsupported for scale type" + to_string(type) + "."));
  }
}

/*
 * LAB specializations
 */
template <>
double ScaleDomainRangeData<ColorLAB>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Mathematical operations on color objects aren't currently supported.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<ColorLAB>::getTypeGL() {
  if (!_cachedTypeGL) {
    auto rootGpuCache = _ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    _cachedTypeGL = ColorLAB::getTypeGLPtr(rootGpuCache->supportedExtensions);
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<ColorLAB>::dataType = QueryDataType::COLOR;

template <>
ColorLAB ScaleDomainRangeData<ColorLAB>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorLAB(ColorUnion(obj.GetString()));
}

template <>
void ScaleDomainRangeData<ColorLAB>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " LAB color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<ColorLAB>::_updateVectorDataByType(TDataColumn<ColorLAB>* dataColumnPtr, ScaleType type) {
  if (type == ScaleType::LINEAR) {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of LAB colors in the domain/range is unsupported for scale type" + to_string(type) + "."));
  }
}

/*
 * HCL specializations
 */
template <>
double ScaleDomainRangeData<ColorHCL>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Mathematical operations on color objects aren't currently supported.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<ColorHCL>::getTypeGL() {
  if (!_cachedTypeGL) {
    auto rootGpuCache = _ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    _cachedTypeGL = ColorHCL::getTypeGLPtr(rootGpuCache->supportedExtensions);
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<ColorHCL>::dataType = QueryDataType::COLOR;

template <>
ColorHCL ScaleDomainRangeData<ColorHCL>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorHCL(ColorUnion(obj.GetString()));
}

template <>
void ScaleDomainRangeData<ColorHCL>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " HCL color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<ColorHCL>::_updateVectorDataByType(TDataColumn<ColorHCL>* dataColumnPtr, ScaleType type) {
  if (type == ScaleType::LINEAR) {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of HCL colors in the domain/range is unsupported for scale type" + to_string(type) + "."));
  }
}

/*
 * string specializations
 */

template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Cannot run mathematical operations on string objects.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL() {
  if (_cachedTypeGL) {
    // NOTE: we should never get in here, but just in case :)
    _cachedTypeGL = nullptr;
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<std::string>::dataType = QueryDataType::STRING;

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetString();
}

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(
      RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " + to_string(type) +
                                           " for string domain/range is invalid. It must be an array of strings."));
}

/*
 * unsigned int specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<unsigned int>::dataType = QueryDataType::UINT;

/*
 * int specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<int>::dataType = QueryDataType::INT;

template <>
int ScaleDomainRangeData<int>::getNullValue() {
  return NULL_INT;
}

/*
 * float specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<float>::dataType = QueryDataType::FLOAT;

template <>
float ScaleDomainRangeData<float>::getNullValue() {
  return NULL_FLOAT;
}

/*
 * double specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<double>::dataType = QueryDataType::DOUBLE;

template <>
double ScaleDomainRangeData<double>::getNullValue() {
  return NULL_DOUBLE;
}

/*
 * int64_t specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<int64_t>::dataType = QueryDataType::INT64;

template <>
int64_t ScaleDomainRangeData<int64_t>::getNullValue() {
  return NULL_INT;
}

/*
 * uint64_t specializations
 */
template <>
const QueryDataType ScaleDomainRangeData<uint64_t>::dataType = QueryDataType::UINT64;

}  // namespace QueryRenderer
