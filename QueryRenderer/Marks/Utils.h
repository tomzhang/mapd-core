#ifndef QUERYRENDERER_MARKS_UTILS_H_
#define QUERYRENDERER_MARKS_UTILS_H_

#include "Types.h"
#include "../Types.h"

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

GeomType getMarkTypeFromJSONObj(const rapidjson::Value& obj);
GeomConfigShPtr createMark(const rapidjson::Value& obj,
                           const rapidjson::Pointer& objPath,
                           const QueryRendererContextShPtr& ctx);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_UTILS_H_
