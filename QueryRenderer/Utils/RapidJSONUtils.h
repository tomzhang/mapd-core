#ifndef QUERYRENDERER_RAPIDJSONUTILS_H_
#define QUERYRENDERER_RAPIDJSONUTILS_H_

#include "../Types.h"
#include "../Data/Types.h"

#include <Rendering/RenderError.h>

#include <memory>
#include <string>
#include <boost/any.hpp>
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

struct RapidJSONUtils {
  typedef std::shared_ptr<rapidjson::Value> JsonCachePtr;

  static std::string getPointerPath(const rapidjson::Pointer& ptr);

  static std::string getObjAsString(const rapidjson::Value& obj);

  static const std::string jsonErrorStrPrefix;
  static std::string getJsonParseErrorStr(const std::string& errStr);
  static std::string getJsonParseErrorStr(const rapidjson::Value& obj, const std::string& errStr);
  static std::string getJsonParseErrorStr(const UserWidgetIdPair& userWidget, const std::string& errStr);
  static std::string getJsonParseErrorStr(const UserWidgetIdPair& userWidget,
                                          const rapidjson::Value& obj,
                                          const std::string& errStr);

  static QueryDataType getDataTypeFromJSONObj(const rapidjson::Value& obj, bool supportString = false);

  template <typename T>
  static T getNumValFromJSONObj(const rapidjson::Value& obj) {
    RUNTIME_EX_ASSERT(obj.IsNumber() || obj.IsBool(),
                      getJsonParseErrorStr(
                          obj, "getNumValFromJSONObj(): rapidjson object is not a number. Cannot get a number value."));

    T rtn(0);

    // TODO: do min/max checks?
    // How would we do this? implicit conversions apparently
    // take place in >=< operations and therefore
    // min/max checks here wouldn't work. For now, implicitly
    // coverting between possible types, but it could result
    // in undefined/unexpected behavior.
    // One way is to have a template specialization for
    // each basic type. Might be the only way to fix.
    // T max = std::numeric_limits<T>::max();
    // T min = std::numeric_limits<T>::lowest();

    if (obj.IsDouble()) {
      double val = obj.GetDouble();
      rtn = static_cast<T>(val);
    } else if (obj.IsInt()) {
      int val = obj.GetInt();
      rtn = static_cast<T>(val);
    } else if (obj.IsUint()) {
      unsigned int val = obj.GetUint();
      rtn = static_cast<T>(val);
    } else if (obj.IsBool()) {
      bool val = obj.GetBool();
      rtn = static_cast<T>(val);
    }  // else if (obj.IsInt64()) {
    // } else if (obj.IsUInt64()) {
    // }

    return rtn;
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RAPIDJSONUTILS_H_
