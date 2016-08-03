#include "Utils.h"
#include "Scale.h"
#include "../Utils/RapidJSONUtils.h"
#include <Rendering/Objects/ColorRGBA.h>

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;

static QueryDataType getDataTypeFromJSONObj(const rapidjson::Value& obj, bool supportString = false) {
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
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "type from JSON is unsupported."));
  }

  return rtn;
}

static QueryDataType getDataTypeFromDataRefJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "data reference is not a JSON object."));

  rapidjson::Value::ConstMemberIterator mitr;
  RUNTIME_EX_ASSERT(
      (mitr = obj.FindMember("data")) != obj.MemberEnd() && mitr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "data reference object doesn't contain a \"data\" string property."));
  std::string dataTableName = mitr->value.GetString();
  QueryDataTableShPtr tablePtr = ctx->getDataTable(dataTableName);

  RUNTIME_EX_ASSERT(tablePtr != nullptr,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "data table \"" + dataTableName + "\" doesn't exist."));

  RUNTIME_EX_ASSERT(
      (mitr = obj.FindMember("field")) != obj.MemberEnd() && mitr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "data reference object must contain a \"field\" string property."));

  return tablePtr->getAttributeType(mitr->value.GetString());
}

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "scale items must be JSON objects."));

  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT(
      (itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "scale objects must contain a \"name\" string property."));

  return itr->value.GetString();
}

ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj) {
  // TODO(croot): expose default as a static attr.
  ScaleType rtn = ScaleType::LINEAR;

  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "scale items must be JSON objects."));

  rapidjson::Value::ConstMemberIterator itr;
  if ((itr = obj.FindMember("type")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(
        itr->value.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(itr->value, "\"type\" property in scale objects must be a string."));
    std::string strScaleType(itr->value.GetString());

    if (strScaleType == "linear") {
      rtn = ScaleType::LINEAR;
    } else if (strScaleType == "ordinal") {
      rtn = ScaleType::ORDINAL;
    } else if (strScaleType == "quantize") {
      rtn = ScaleType::QUANTIZE;
    } else {
      THROW_RUNTIME_EX(
          RapidJSONUtils::getJsonParseErrorStr(obj, "scale type \"" + strScaleType + "\" is not a supported type."));
    }
  }

  return rtn;
}

QueryDataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;

  // TODO(croot): expose "domain" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (itr->value.IsArray() && itr->value.Size())),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "\"domain\" property for scales must exist and must be an object or an array."));

  QueryDataType domainType, itemType;

  if (isObject) {
    domainType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else {
    // TODO(croot): Probably need to have specific classes to correspond
    // to the different scales. For example, ordinal/categorical scales
    // can support strings for domain values. Others shouldn't.
    // Will allow all domains to accept all strings for now.
    domainType = getDataTypeFromJSONObj(itr->value[0], true);

    for (size_t i = 1; i < itr->value.Size(); ++i) {
      itemType = RapidJSONUtils::getDataTypeFromJSONObj(itr->value[i], true);

      try {
        domainType = getHigherPriorityDataType(domainType, itemType);
      } catch (const std::runtime_error& e) {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
            obj,
            "scale domain item at index " + std::to_string(i) + " has an incompatible type. " + to_string(domainType) +
                " is not compatible with " + to_string(itemType)));
      }
    }
  }

  return domainType;
}

QueryDataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;
  bool isString;

  // TODO(croot): expose "range" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("range")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (isString = itr->value.IsString()) ||
                         (itr->value.IsArray() && itr->value.Size())),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "\"range\" property for scales must exist and must be an object or a string."));

  QueryDataType rangeType, itemType;
  if (isObject) {
    rangeType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else if (isString) {
    std::string strVal = itr->value.GetString();
    RUNTIME_EX_ASSERT(strVal == "width" || strVal == "height",
                      RapidJSONUtils::getJsonParseErrorStr(
                          itr->value,
                          "invalid \"range\" string property for scales. Only string values supported are "
                          "\"width\" and \"height\""));

    // TODO(croot): should we actually use width/height values in the scale?
    // The easy way is just to use the -1 to 1 NDC range, but what
    // if someone uses "height" as the range, but uses that scale
    // for an x coordinate or some other property? Should use the
    // true 0-height in that case.
    rangeType = QueryDataType::FLOAT;
  } else {
    rangeType = getDataTypeFromJSONObj(itr->value[0]);

    for (size_t i = 1; i < itr->value.Size(); ++i) {
      itemType = RapidJSONUtils::getDataTypeFromJSONObj(itr->value[i]);

      try {
        rangeType = getHigherPriorityDataType(rangeType, itemType);
      } catch (const std::runtime_error& e) {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj,
                                                              "scale range item at index " + std::to_string(i) +
                                                                  " has an incompatible type. " + to_string(rangeType) +
                                                                  " is not compatible with " + to_string(itemType)));
      }
    }
  }

  return rangeType;
}

template <typename DomainType, typename RangeType>
ScaleShPtr createScalePtr(const rapidjson::Value& obj,
                          const rapidjson::Pointer& objPath,
                          const QueryRendererContextShPtr& ctx,
                          const std::string& scaleName,
                          ScaleType scaleType) {
  switch (scaleType) {
    case ScaleType::LINEAR:
      return ScaleShPtr(new LinearScale<DomainType, RangeType>(obj, objPath, ctx, scaleName));
    case ScaleType::ORDINAL:
      return ScaleShPtr(new OrdinalScale<DomainType, RangeType>(obj, objPath, ctx, scaleName));
    case ScaleType::QUANTIZE:
      return ScaleShPtr(new QuantizeScale<DomainType, RangeType>(obj, objPath, ctx, scaleName));
    default:
      THROW_RUNTIME_EX(
          RapidJSONUtils::getJsonParseErrorStr(obj, "Scale type " + to_string(scaleType) + " is unsupported."));
  }
}

ScaleShPtr createScale(const rapidjson::Value& obj,
                       const rapidjson::Pointer& objPath,
                       const QueryRendererContextShPtr& ctx,
                       const std::string& name,
                       ScaleType type) {
  std::string scaleName(name);
  if (!scaleName.length()) {
    scaleName = getScaleNameFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleName.length() > 0,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "Scales must have a \"name\" property"));

  ScaleType scaleType(type);
  if (scaleType == ScaleType::UNDEFINED) {
    scaleType = getScaleTypeFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleType != ScaleType::UNDEFINED,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "Scale type for \"" + scaleName + "\" is undefined."));

  rapidjson::Value::ConstMemberIterator itr;

  QueryDataType domainType = getScaleDomainDataTypeFromJSONObj(obj, ctx);
  QueryDataType rangeType = getScaleRangeDataTypeFromJSONObj(obj, ctx);

  // TODO(croot): put this in the quantize scale class? It's easier to do here, so keeping for
  // now.
  if (scaleType == ScaleType::QUANTIZE) {
    RUNTIME_EX_ASSERT(
        domainType == QueryDataType::UINT || domainType == QueryDataType::INT || domainType == QueryDataType::FLOAT ||
            domainType == QueryDataType::DOUBLE,
        RapidJSONUtils::getJsonParseErrorStr(
            obj, "Domain type " + to_string(domainType) + " is not a supported domain for a quantize scale."));
  }

  switch (domainType) {
    case QueryDataType::UINT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<unsigned int, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<unsigned int, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<unsigned int, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<unsigned int, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<unsigned int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::INT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<int, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<int, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<int, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<int, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::FLOAT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<float, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<float, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<float, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<float, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<float, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::DOUBLE:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<double, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<double, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<double, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<double, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<double, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::COLOR:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<ColorRGBA, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<ColorRGBA, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<ColorRGBA, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<ColorRGBA, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<ColorRGBA, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::STRING:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<std::string, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<std::string, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<std::string, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<std::string, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createScalePtr<std::string, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "domain type is unsupported."));
  }
}

QueryDataType getHigherPriorityDataType(const QueryDataType baseDataType, const QueryDataType checkDataType) {
  if (baseDataType == checkDataType) {
    return baseDataType;
  }

  switch (baseDataType) {
    case QueryDataType::FLOAT:
      if (checkDataType == QueryDataType::UINT || checkDataType == QueryDataType::INT) {
        return baseDataType;
      } else if (checkDataType == QueryDataType::DOUBLE) {
        return checkDataType;
      }
      break;
    case QueryDataType::DOUBLE:
      if (checkDataType == QueryDataType::UINT || checkDataType == QueryDataType::INT ||
          checkDataType == QueryDataType::FLOAT) {
        return baseDataType;
      }
      break;
    case QueryDataType::INT:
      if (checkDataType == QueryDataType::UINT) {
        return baseDataType;
      } else if (checkDataType == QueryDataType::FLOAT || checkDataType == QueryDataType::DOUBLE) {
        return checkDataType;
      }
      break;
    case QueryDataType::UINT:
      if (checkDataType == QueryDataType::INT || checkDataType == QueryDataType::FLOAT ||
          checkDataType == QueryDataType::DOUBLE) {
        return checkDataType;
      }
      break;
    default:
      break;
  }

  THROW_RUNTIME_EX(to_string(baseDataType) + " is incompatible with " + to_string(checkDataType));
  return baseDataType;
}

}  // namespace QueryRenderer
