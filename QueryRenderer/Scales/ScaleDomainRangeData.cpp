#include "ScaleDomainRangeData.h"
#include "Scale.h"

namespace QueryRenderer {

using ::Rendering::GL::TypeGLShPtr;
using ::Rendering::Objects::ColorRGBA;

template <>
double ScaleDomainRangeData<ColorRGBA>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Mathematical operations on color objects aren't currently supported.");
}

template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Cannot run mathematical operations on string objects.");
}

template <>
const TypeGLShPtr& ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
  // TODO(croot): combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions
  if (!_cachedTypeGL) {
    _cachedTypeGL.reset(new ::Rendering::GL::TypeGL<float, 4>());
  }
  return _cachedTypeGL;
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
const QueryDataType ScaleDomainRangeData<unsigned int>::dataType = QueryDataType::UINT;

template <>
unsigned int ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetUint();
}

template <>
const QueryDataType ScaleDomainRangeData<int>::dataType = QueryDataType::INT;

template <>
int ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetInt();
}

template <>
const QueryDataType ScaleDomainRangeData<float>::dataType = QueryDataType::FLOAT;

template <>
float ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return static_cast<float>(obj.GetDouble());
}

template <>
const QueryDataType ScaleDomainRangeData<double>::dataType = QueryDataType::DOUBLE;

template <>
double ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetDouble();
}

template <>
const QueryDataType ScaleDomainRangeData<ColorRGBA>::dataType = QueryDataType::COLOR;

template <>
ColorRGBA ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorRGBA(obj.GetString());
}

template <>
const QueryDataType ScaleDomainRangeData<std::string>::dataType = QueryDataType::STRING;

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetString();
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, ScaleType type) {
  THROW_RUNTIME_EX(
      RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " + to_string(type) +
                                           " for string domain/range is invalid. It must be an array of strings."));
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA>* dataColumnPtr, ScaleType type) {
  if (type == ScaleType::LINEAR) {
    // TODO(croot): update ColorRGBA to be supported by the <,>,etc. operators
    // so that the getExtrema() call will work to get the min and max colors
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of colors in the domain/range is unsupported for scale type" + to_string(type) + "."));
  }
}

}  // namespace QueryRenderer
