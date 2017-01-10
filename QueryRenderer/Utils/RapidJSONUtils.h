#ifndef QUERYRENDERER_RAPIDJSONUTILS_H_
#define QUERYRENDERER_RAPIDJSONUTILS_H_

#include "../Types.h"
#include "../Data/Types.h"

#include <Rendering/RenderError.h>
#include <Rendering/Colors/Types.h>
#include <Rendering/Colors/ColorUnion.h>

#include <memory>
#include <string>
#include <boost/any.hpp>
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

template <class T>
struct is_std_string
    : std::integral_constant<bool, std::is_same<std::string, typename std::remove_cv<T>::type>::value> {};

class AnyDataType {
 private:
  QueryDataType _type;
  boost::any _val;

 public:
  AnyDataType() : _type(QueryDataType::INT), _val(int(0)) {}
  AnyDataType(const QueryDataType type, const boost::any& val) : _type(type), _val(val) {}
  ~AnyDataType() {}

  void set(const QueryDataType type, const boost::any& val) {
    _type = type;
    _val = val;
  }

  QueryDataType getType() const { return _type; }

  template <typename T>
  T getVal() const {
    if (_val.empty()) {
      return T();
    }

    try {
      switch (_type) {
        case QueryDataType::INT:
          return static_cast<T>(boost::any_cast<int>(_val));
        case QueryDataType::UINT:
          return static_cast<T>(boost::any_cast<unsigned int>(_val));
        case QueryDataType::FLOAT:
          return static_cast<T>(boost::any_cast<float>(_val));
        case QueryDataType::INT64:
          return static_cast<T>(boost::any_cast<int64_t>(_val));
        case QueryDataType::UINT64:
          return static_cast<T>(boost::any_cast<uint64_t>(_val));
        case QueryDataType::DOUBLE:
          return static_cast<T>(boost::any_cast<double>(_val));
        default:
          break;
      }
    } catch (boost::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of sync. It's labeled as a " +
          to_string(_type) + " but its value is not.");
    }

    THROW_RUNTIME_EX("Conversion of the val of AnyDataType type " + std::string(_val.type().name()) + " to type " +
                     typeid(T()).name() + " is not supported");
    return T();
  }

  template <typename T, typename std::enable_if<::Rendering::Colors::is_color_union<T>::value>::type* = nullptr>
  const T& getColorRef() const {
    RUNTIME_EX_ASSERT(_type == QueryDataType::COLOR,
                      "Cannot get reference to a color. The AnyDataType is marked as a " + to_string(_type) + ".");

    RUNTIME_EX_ASSERT(!_val.empty(), "Cannot get reference to a color. The AnyDataType object is not initialized");

    try {
      return boost::any_cast<const ::Rendering::Colors::ColorUnion&>(_val);
    } catch (boost::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of sync. It's labeled as a " +
          to_string(_type) + " but its value is not.");
    }
  }

  template <typename T, typename std::enable_if<::Rendering::Colors::is_color_union<T>::value>::type* = nullptr>
  T& getColorRef() {
    return const_cast<T&>(static_cast<const AnyDataType&>(*this).getColorRef<T>());
  }

  std::string getStringVal() const {
    // TODO(croot): support converting any type to a string
    RUNTIME_EX_ASSERT(
        _type == QueryDataType::STRING,
        "Cannot extract a string from the AnyDataType object. It is marked as a " + to_string(_type) + ".");
    if (_val.empty()) {
      return "";
    }

    try {
      return boost::any_cast<std::string>(_val);
    } catch (boost::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get a string value of the AnyDataType. Its internals have gotten out of sync. It's labeled as a " +
          to_string(_type) + " but its value is not.");
    }
  }

  operator std::string() const {
    std::string rtn = "type: " + to_string(_type) + " val: ";
    if (_val.empty()) {
      rtn += "empty";
    }

    try {
      switch (_type) {
        case QueryDataType::INT:
          rtn += std::to_string(boost::any_cast<int>(_val));
          return rtn;
        case QueryDataType::UINT:
          rtn += std::to_string(boost::any_cast<unsigned int>(_val));
          return rtn;
        case QueryDataType::FLOAT:
          rtn += std::to_string(boost::any_cast<float>(_val));
          return rtn;
        case QueryDataType::INT64:
          rtn += std::to_string(boost::any_cast<int64_t>(_val));
          return rtn;
        case QueryDataType::UINT64:
          rtn += std::to_string(boost::any_cast<uint64_t>(_val));
          return rtn;
        case QueryDataType::DOUBLE:
          rtn += std::to_string(boost::any_cast<double>(_val));
          return rtn;
        case QueryDataType::COLOR:
          rtn += std::string(boost::any_cast<::Rendering::Colors::ColorUnion>(_val));
        case QueryDataType::STRING:
          rtn += std::string(boost::any_cast<std::string>(_val));
        default:
          break;
      }
    } catch (boost::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot convert AnyDataType to a string. Its internals have gotten out of sync. It's labeled as a " +
          to_string(_type) + " but its value is not.");
    }

    THROW_RUNTIME_EX("Conversion of AnyDataType type " + to_string(_type) + " to a string is not yet supported");
    return rtn;
  }
};

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
  static AnyDataType getAnyDataFromJSONObj(const rapidjson::Value& obj, bool supportString = false);
  static QueryDataType getHigherPriorityDataType(const AnyDataType& baseDataType, const AnyDataType& checkDataType);

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

    if (obj.IsBool()) {
      auto val = obj.GetBool();
      rtn = static_cast<T>(val);
    } else if (obj.IsInt()) {
      auto val = obj.GetInt();
      rtn = static_cast<T>(val);
    } else if (obj.IsUint()) {
      auto val = obj.GetUint();
      rtn = static_cast<T>(val);
    } else if (obj.IsInt64()) {
      auto val = obj.GetInt64();
      rtn = static_cast<T>(val);
    } else if (obj.IsUint64()) {
      auto val = obj.GetUint64();
      rtn = static_cast<T>(val);
    } else if (obj.IsDouble()) {
      auto val = obj.GetDouble();
      rtn = static_cast<T>(val);
    }

    return rtn;
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RAPIDJSONUTILS_H_
