#include "RapidJSONUtils.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

std::string MapD_Renderer::RapidJSONUtils::getPointerPath(const rapidjson::Pointer& ptr) {
  rapidjson::StringBuffer sb;
  ptr.Stringify(sb);
  return sb.GetString();
}

std::string MapD_Renderer::RapidJSONUtils::getObjAsString(const rapidjson::Value& obj) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  obj.Accept(writer);
  return std::string(sb.GetString());
}
