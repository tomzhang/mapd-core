#ifndef QUERYRENDERER_SCALES_UTILS_H_
#define QUERYRENDERER_SCALES_UTILS_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj);
ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj,
                                                const QueryRendererContextShPtr& ctx,
                                                const ScaleType scaleType);
QueryDataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj,
                                               const QueryRendererContextShPtr& ctx,
                                               const ScaleType scaleType);
ScaleShPtr createScale(const rapidjson::Value& obj,
                       const rapidjson::Pointer& objPath,
                       const QueryRendererContextShPtr& ctx,
                       const std::string& name = "",
                       // TODO(croot): expose default as a constant somewhere
                       ScaleType type = ScaleType::UNDEFINED);

QueryDataType getHigherPriorityDataType(const QueryDataType baseDataType, const QueryDataType checkDataType);
bool isScaleDomainCompatible(const ScaleType scaleType, const QueryDataType domainType);
bool isScaleRangeCompatible(const ScaleType scaleType, const QueryDataType rangeType);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_UTILS_H_
