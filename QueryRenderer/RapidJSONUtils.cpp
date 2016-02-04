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
