#include "RapidJSONUtils.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

std::string QueryRenderer::RapidJSONUtils::getPointerPath(const rapidjson::Pointer& ptr) {
  rapidjson::StringBuffer sb;
  ptr.Stringify(sb);
  return sb.GetString();
}

std::string QueryRenderer::RapidJSONUtils::getObjAsString(const rapidjson::Value& obj) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  obj.Accept(writer);
  return std::string(sb.GetString());
}

const std::string QueryRenderer::RapidJSONUtils::jsonErrorStrPrefix = "JSON parse error";
std::string QueryRenderer::RapidJSONUtils::getJsonParseErrorStr(const std::string& errStr) {
  return jsonErrorStrPrefix + ": " + errStr;
}

std::string QueryRenderer::RapidJSONUtils::getJsonParseErrorStr(const rapidjson::Value& obj,
                                                                const std::string& errStr) {
  return jsonErrorStrPrefix + " obj: " + getObjAsString(obj) + ": " + errStr;
}

std::string QueryRenderer::RapidJSONUtils::getJsonParseErrorStr(const UserWidgetIdPair& userWidget,
                                                                const std::string& errStr) {
  return jsonErrorStrPrefix + " " + to_string(userWidget) + ": " + errStr;
}

std::string QueryRenderer::RapidJSONUtils::getJsonParseErrorStr(const UserWidgetIdPair& userWidget,
                                                                const rapidjson::Value& obj,
                                                                const std::string& errStr) {
  return jsonErrorStrPrefix + " " + to_string(userWidget) + " obj: " + getObjAsString(obj) + ": " + errStr;
}
