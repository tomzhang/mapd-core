#include "RapidJSONUtils.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <Rendering/Colors/Utils.h>
#include <Rendering/Colors/ColorUnion.h>

namespace QueryRenderer {

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
  return getAnyDataFromJSONObj(obj, supportString).getType();
}

AnyDataType RapidJSONUtils::getAnyDataFromJSONObj(const rapidjson::Value& obj, bool supportString) {
  AnyDataType rtn;
  rapidjson::Type type = obj.GetType();

  switch (type) {
    case rapidjson::kNumberType:
      if (obj.IsInt()) {
        rtn.set(QueryDataType::INT, obj.GetInt());
      } else if (obj.IsUint()) {
        rtn.set(QueryDataType::UINT, obj.GetUint());
      } else if (obj.IsInt64()) {
        rtn.set(QueryDataType::INT64, obj.GetInt64());
      } else if (obj.IsUint64()) {
        rtn.set(QueryDataType::UINT64, obj.GetUint64());
      } else if (obj.IsDouble()) {
        double val = obj.GetDouble();

        // will be a float if it can be losslessly converted to a float
        // NOTE: version 1.1.0 and greater of rapidjson have functions
        // that do this check, IsLosslessFloat()
        double check = static_cast<double>(static_cast<float>(val));
        if (val >= check && val <= check) {
          rtn.set(QueryDataType::FLOAT, static_cast<float>(val));
        } else {
          rtn.set(QueryDataType::DOUBLE, val);
        }
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "RapidJSON number type is not supported."));
      }
      break;
    case rapidjson::kStringType: {
      std::string val = obj.GetString();
      if (supportString) {
        rtn.set(QueryDataType::STRING, val);
      } else if (::Rendering::Colors::isColorString(val)) {
        rtn.set(QueryDataType::COLOR, ::Rendering::Colors::ColorUnion(val));
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "non-color strings are not a supported type."));
      }
    } break;
    case rapidjson::kTrueType:
    case rapidjson::kFalseType:
      rtn.set(QueryDataType::BOOL, obj.GetBool());
      break;
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "type from JSON is unsupported."));
  }

  return rtn;
}

template <typename FromType,
          typename ToType,
          typename std::enable_if<std::is_integral<FromType>::value>::type* = nullptr>
bool isLosslessIntegralConversion(const AnyDataType& dataType) {
  auto val = dataType.getVal<FromType>();
  return static_cast<FromType>(static_cast<ToType>(val)) == val;
}

template <typename FromType,
          typename ToType,
          typename std::enable_if<std::is_floating_point<FromType>::value>::type* = nullptr>
bool isLosslessFloatingPtConversion(const AnyDataType& dataType) {
  auto val = dataType.getVal<FromType>();
  auto convval = static_cast<FromType>(static_cast<ToType>(val)) == val;
  return convval <= val && convval >= val;
}

QueryDataType RapidJSONUtils::getHigherPriorityDataType(const AnyDataType& baseDataType,
                                                        const AnyDataType& checkDataType) {
  auto baseType = baseDataType.getType();
  auto checkType = checkDataType.getType();

  if (baseType == checkType) {
    return baseType;
  }

  switch (baseType) {
    case QueryDataType::FLOAT:
      if (checkType == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, float>(checkDataType)) {
          return QueryDataType::FLOAT;
        } else {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::UINT) {
        if (isLosslessIntegralConversion<unsigned int, float>(checkDataType)) {
          return QueryDataType::FLOAT;
        } else {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, float>(checkDataType)) {
          return QueryDataType::FLOAT;
        }
      } else if (checkType == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<uint64_t, float>(checkDataType)) {
          return QueryDataType::FLOAT;
        }
      } else if (checkType == QueryDataType::DOUBLE) {
        return QueryDataType::DOUBLE;
      }
    // NOTE: intentionally not breaking the FLOAT case to let
    // INT64/UINT64 convertible to double pass thru
    case QueryDataType::DOUBLE:
      if (checkType == QueryDataType::UINT || checkType == QueryDataType::INT || checkType == QueryDataType::FLOAT) {
        return QueryDataType::DOUBLE;
      } else if (checkType == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, double>(checkDataType)) {
          return QueryDataType::INT64;
        } else {
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<uint64_t, double>(checkDataType)) {
          return QueryDataType::UINT64;
        } else {
          return QueryDataType::DOUBLE;
        }
      }
      break;
    case QueryDataType::INT:
      if (checkType == QueryDataType::UINT) {
        if (isLosslessIntegralConversion<int, unsigned int>(baseDataType)) {
          return QueryDataType::UINT;
        } else {
          // the int < 0, so int64 will hold both
          return QueryDataType::INT64;
        }
      } else if (checkType == QueryDataType::FLOAT) {
        if (isLosslessIntegralConversion<int, float>(baseDataType)) {
          return QueryDataType::FLOAT;
        } else {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<int, uint64_t>(baseDataType)) {
          return QueryDataType::UINT64;
        } else if (isLosslessIntegralConversion<uint64_t, double>(baseDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::INT64 || checkType == QueryDataType::DOUBLE) {
        return checkType;
      }
      break;
    case QueryDataType::UINT:
      if (checkType == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, unsigned int>(checkDataType)) {
          return QueryDataType::UINT;
        } else {
          // the int < 0, so int64 will hold both
          return QueryDataType::INT64;
        }
      } else if (checkType == QueryDataType::FLOAT) {
        if (isLosslessIntegralConversion<unsigned int, float>(baseDataType)) {
          return QueryDataType::FLOAT;
        } else {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      } else if (checkType == QueryDataType::INT64 || checkType == QueryDataType::UINT64 ||
                 checkType == QueryDataType::DOUBLE) {
        return checkType;
      }
      break;
    case QueryDataType::INT64:
      if (checkType == QueryDataType::INT || checkType == QueryDataType::UINT) {
        return QueryDataType::INT64;
      } else if (checkType == QueryDataType::FLOAT || checkType == QueryDataType::DOUBLE) {
        if (isLosslessIntegralConversion<int64_t, double>(baseDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
        return QueryDataType::INT64;
      } else if (checkType == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<int64_t, uint64_t>(baseDataType)) {
          return QueryDataType::UINT64;
        } else if (isLosslessIntegralConversion<int64_t, double>(baseDataType) &&
                   isLosslessIntegralConversion<uint64_t, double>(checkDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
        // let pass thru to throw an error
      }
      break;
    case QueryDataType::UINT64:
      if (checkType == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, uint64_t>(checkDataType)) {
          return QueryDataType::UINT64;
        } else if (isLosslessIntegralConversion<uint64_t, double>(baseDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
        // let pass thru to throw an error
      } else if (checkType == QueryDataType::UINT) {
        return QueryDataType::UINT64;
      } else if (checkType == QueryDataType::FLOAT || checkType == QueryDataType::DOUBLE) {
        if (isLosslessIntegralConversion<uint64_t, double>(baseDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
        if (checkDataType.getVal<double>() >= 0) {
          return QueryDataType::UINT64;
        }
        // let pass thru to throw an error
      } else if (checkType == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, uint64_t>(checkDataType)) {
          return QueryDataType::UINT64;
        } else if (isLosslessIntegralConversion<uint64_t, double>(baseDataType) &&
                   isLosslessIntegralConversion<int64_t, double>(checkDataType)) {
          // double will hold both
          return QueryDataType::DOUBLE;
        }
      }
      break;

    default:
      break;
  }

  THROW_RUNTIME_EX(std::string(baseDataType) + " is incompatible with " + std::string(checkDataType));
  return baseType;
}

}  // namespace QueryRenderer
