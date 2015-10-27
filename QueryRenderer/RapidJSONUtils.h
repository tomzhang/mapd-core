#ifndef RAPID_JSON_UTILS_H_
#define RAPID_JSON_UTILS_H_

#include <stdexcept>
#include <string>
#include "rapidjson/document.h"

namespace MapD_Renderer {

class RJMapDException : public std::runtime_error {
 public:
  explicit RJMapDException(const std::string& what_arg) : std::runtime_error("Invalid json config: " + what_arg) {}
};

template <typename T>
T getNumValFromJSONObj(const rapidjson::Value& obj) {
  assert(obj.IsNumber());

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
  }  // else if (obj.IsInt64()) {
  // } else if (obj.IsUInt64()) {
  // }

  return rtn;
}

}  // namespace MapD_Renderer

#endif  // RAPID_JSON_UTILS_H_
