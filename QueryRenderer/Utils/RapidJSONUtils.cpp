#include "RapidJSONUtils.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <Rendering/Objects/ColorRGBA.h>

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;

std::string RapidJSONUtils::getPointerPath(const rapidjson::Pointer& ptr) {
  rapidjson::StringBuffer sb;
  ptr.Stringify(sb);
  return sb.GetString();
}

std::string RapidJSONUtils::getObjAsString(const rapidjson::Value& obj) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  obj.Accept(writer);
  return std::string(sb.GetString());
}

const std::string RapidJSONUtils::jsonErrorStrPrefix = "JSON parse error";
std::string RapidJSONUtils::getJsonParseErrorStr(const std::string& errStr) {
  return jsonErrorStrPrefix + ": " + errStr;
}

std::string RapidJSONUtils::getJsonParseErrorStr(const rapidjson::Value& obj, const std::string& errStr) {
  return jsonErrorStrPrefix + " obj: " + getObjAsString(obj) + ": " + errStr;
}

std::string RapidJSONUtils::getJsonParseErrorStr(const UserWidgetIdPair& userWidget, const std::string& errStr) {
  return jsonErrorStrPrefix + " " + to_string(userWidget) + ": " + errStr;
}

std::string RapidJSONUtils::getJsonParseErrorStr(const UserWidgetIdPair& userWidget,
                                                 const rapidjson::Value& obj,
                                                 const std::string& errStr) {
  return jsonErrorStrPrefix + " " + to_string(userWidget) + " obj: " + getObjAsString(obj) + ": " + errStr;
}

QueryDataType RapidJSONUtils::getDataTypeFromJSONObj(const rapidjson::Value& obj, bool supportString) {
  QueryDataType rtn(QueryDataType::INT);
  rapidjson::Type type = obj.GetType();

  switch (type) {
    case rapidjson::kNumberType:
      if (obj.IsInt()) {
        rtn = QueryDataType::INT;
      } else if (obj.IsUint()) {
        rtn = QueryDataType::UINT;
      } else if (obj.IsDouble()) {
        rtn = QueryDataType::DOUBLE;

        // double val = obj.GetDouble();
        // TODO(croot): how do we handle floats?
        // if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
        //   rtn = QueryDataType::FLOAT;
        // } else {
        //   rtn = QueryDataType::DOUBLE;
        // }
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "RapidJSON number type is not supported."));
      }
      break;
    case rapidjson::kStringType: {
      std::string val = obj.GetString();
      if (supportString) {
        rtn = QueryDataType::STRING;
      } else if (ColorRGBA::isColorString(val)) {
        rtn = QueryDataType::COLOR;
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "non-color strings are not a supported type."));
      }
    } break;
    case rapidjson::kTrueType:
    case rapidjson::kFalseType:
      rtn = QueryDataType::BOOL;
      break;
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "type from JSON is unsupported."));
  }

  return rtn;
}

}  // namespace QueryRenderer
