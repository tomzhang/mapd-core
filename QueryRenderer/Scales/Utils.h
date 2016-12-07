#ifndef QUERYRENDERER_SCALES_UTILS_H_
#define QUERYRENDERER_SCALES_UTILS_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

#include <Rendering/RenderError.h>
#include <Rendering/Colors/Types.h>

#include <boost/any.hpp>

namespace QueryRenderer {

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj);
ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj,
                                                const QueryRendererContextShPtr& ctx,
                                                const ScaleType scaleType);
QueryDataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj,
                                               const QueryRendererContextShPtr& ctx,
                                               const ScaleType scaleType);
std::pair<::Rendering::Colors::ColorType, ScaleInterpType> getScaleRangeColorTypeFromJSONObj(
    const rapidjson::Value& obj);

ScaleShPtr createScale(const rapidjson::Value& obj,
                       const rapidjson::Pointer& objPath,
                       const QueryRendererContextShPtr& ctx,
                       const std::string& name = "",
                       // TODO(croot): expose default as a constant somewhere
                       ScaleType type = ScaleType::UNDEFINED);

QueryDataType getHigherPriorityDataType(const QueryDataType baseDataType, const QueryDataType checkDataType);
bool isScaleDomainCompatible(const ScaleType scaleType, const QueryDataType domainType);
bool isScaleRangeCompatible(const ScaleType scaleType, const QueryDataType rangeType);
bool areTypesCompatible(const QueryDataType srcType, const QueryDataType inType);
bool areTypesCompatible(const std::type_info& srcType, const std::type_info& inType);

QueryDataType convertTypeIdToDataType(const std::type_info& srcTypeId);

template <typename T>
T convertType(const QueryDataType type, const boost::any& value) {
  switch (type) {
    case QueryDataType::UINT: {
      unsigned int val = boost::any_cast<unsigned int>(value);
      return static_cast<T>(val);
    }
    case QueryDataType::INT: {
      int val = boost::any_cast<int>(value);
      return static_cast<T>(val);
    }
    case QueryDataType::FLOAT: {
      float val = boost::any_cast<float>(value);
      return static_cast<T>(val);
    }
    case QueryDataType::DOUBLE: {
      double val = boost::any_cast<double>(value);
      return static_cast<T>(val);
    }
    default:
      THROW_RUNTIME_EX("Converting " + to_string(type) + " to " + typeid(T).name() + " is currently unsupported.");
  }
}

template <>
Rendering::Colors::ColorRGBA convertType(const QueryDataType type, const boost::any& value);

template <>
Rendering::Colors::ColorHSL convertType(const QueryDataType type, const boost::any& value);

template <>
Rendering::Colors::ColorLAB convertType(const QueryDataType type, const boost::any& value);

template <>
Rendering::Colors::ColorHCL convertType(const QueryDataType type, const boost::any& value);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_UTILS_H_
