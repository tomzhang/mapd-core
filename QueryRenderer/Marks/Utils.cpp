#include "Utils.h"
#include "../Utils/RapidJSONUtils.h"
#include "PointMark.h"

namespace QueryRenderer {

GeomType getMarkTypeFromJSONObj(const rapidjson::Value& obj) {
  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("type")) != obj.MemberEnd() && itr->value.IsString(),
                    RapidJSONUtils::getJsonParseErrorStr(obj, "a mark object must have a \"type\" string property."));

  std::string strGeomType(itr->value.GetString());

  if (strGeomType == "points") {
    return GeomType::POINTS;
  } else {
    THROW_RUNTIME_EX(
        RapidJSONUtils::getJsonParseErrorStr(obj, "a mark of type \"" + strGeomType + "\" is unsupported."));
  }
}

GeomConfigShPtr createMark(const rapidjson::Value& obj,
                           const rapidjson::Pointer& objPath,
                           const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "marks must be objects."));

  switch (getMarkTypeFromJSONObj(obj)) {
    case GeomType::POINTS:
      return GeomConfigShPtr(new PointMark(obj, objPath, ctx));
      break;
  }

  return GeomConfigShPtr(nullptr);
}

}  // namespace QueryRenderer
