#include "Utils.h"
#include "Scale.h"
#include <Rendering/Colors/ColorRGBA.h>
#include <Rendering/Colors/ColorHSL.h>
#include <Rendering/Colors/ColorLAB.h>
#include <Rendering/Colors/ColorHCL.h>
#include <type_traits>
#include <boost/any.hpp>

namespace QueryRenderer {

using ::Rendering::Colors::ColorRGBA;
using ::Rendering::Colors::ColorHSL;
using ::Rendering::Colors::ColorLAB;
using ::Rendering::Colors::ColorHCL;

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
    } else if (strScaleType == "log") {
      rtn = ScaleType::LOG;
    } else if (strScaleType == "pow") {
      rtn = ScaleType::POW;
    } else if (strScaleType == "sqrt") {
      rtn = ScaleType::SQRT;
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

QueryDataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj,
                                                const QueryRendererContextShPtr& ctx,
                                                const ScaleType scaleType) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;

  // TODO(croot): expose "domain" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (itr->value.IsArray() && itr->value.Size())),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "\"domain\" property for scales must exist and must be an object or an array."));

  QueryDataType domainType{QueryDataType::UINT};

  if (isObject) {
    domainType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else {
    // TODO(croot): Probably need to have specific classes to correspond
    // to the different scales. For example, ordinal/categorical scales
    // can support strings for domain values. Others shouldn't.
    // Will allow all domains to accept all strings for now.

    AnyDataType domainAnyType, itemAnyType;
    size_t startIdx = 0;
    switch (scaleType) {
      case ScaleType::LOG:
      case ScaleType::SQRT:
      case ScaleType::POW:
        // TODO(croot): support float?
        domainAnyType.set(QueryDataType::DOUBLE, double(0));
        startIdx = 0;
        break;
      default:
        domainAnyType = RapidJSONUtils::getAnyDataFromJSONObj(itr->value[0], true);
        startIdx = 1;
        break;
    }

    for (size_t i = startIdx; i < itr->value.Size(); ++i) {
      itemAnyType = RapidJSONUtils::getAnyDataFromJSONObj(itr->value[i], true);

      try {
        domainType = RapidJSONUtils::getHigherPriorityDataType(domainAnyType, itemAnyType);
      } catch (const std::runtime_error& e) {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
            obj,
            "scale domain item at index " + std::to_string(i) + " has an incompatible type. " +
                to_string(domainAnyType.getType()) + " is not compatible with " + to_string(itemAnyType.getType())));
      }
    }
  }

  return domainType;
}

QueryDataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj,
                                               const QueryRendererContextShPtr& ctx,
                                               const ScaleType scaleType) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;
  bool isString;

  // TODO(croot): expose "range" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("range")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (isString = itr->value.IsString()) ||
                         (itr->value.IsArray() && itr->value.Size())),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "\"range\" property for scales must exist and must be an object or a string."));

  QueryDataType rangeType{QueryDataType::UINT};
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
    AnyDataType rangeAnyType = RapidJSONUtils::getAnyDataFromJSONObj(itr->value[0]), itemAnyType;

    for (size_t i = 1; i < itr->value.Size(); ++i) {
      itemAnyType = RapidJSONUtils::getAnyDataFromJSONObj(itr->value[i]);

      try {
        rangeType = RapidJSONUtils::getHigherPriorityDataType(rangeAnyType, itemAnyType);
      } catch (const std::runtime_error& e) {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
            obj,
            "scale range item at index " + std::to_string(i) + " has an incompatible type. " +
                to_string(rangeAnyType.getType()) + " is not compatible with " + to_string(itemAnyType.getType())));
      }
    }
  }

  return rangeType;
}

template <typename DomainType, typename RangeType>
static ScaleShPtr createScalePtr(const rapidjson::Value& obj,
                                 const rapidjson::Pointer& objPath,
                                 const QueryRendererContextShPtr& ctx,
                                 const std::string& scaleName,
                                 const ScaleType scaleType,
                                 const ScaleInterpType interpType = ScaleInterpType::UNDEFINED) {
  switch (scaleType) {
    case ScaleType::LINEAR:
      return ScaleShPtr(new LinearScale<DomainType, RangeType>(obj, objPath, ctx, scaleName, interpType));
    case ScaleType::LOG: {
      // TODO(croot): support float?
      // NOTE: DomainType should have been validated before hand
      bool isValid = std::is_same<DomainType, double>::value;
      CHECK(isValid);
      return ScaleShPtr(new LogScale<double, RangeType>(obj, objPath, ctx, scaleName, interpType));
    }
    case ScaleType::POW: {
      // TODO(croot): support float?
      // NOTE: DomainType should have been validated before hand
      bool isValid = std::is_same<DomainType, double>::value;
      CHECK(isValid);
      return ScaleShPtr(new PowScale<double, RangeType>(obj, objPath, ctx, scaleName, interpType));
    }
    case ScaleType::SQRT: {
      // TODO(croot): support float?
      // NOTE: DomainType should have been validated before hand
      bool isValid = std::is_same<DomainType, double>::value;
      CHECK(isValid);
      return ScaleShPtr(new SqrtScale<double, RangeType>(obj, objPath, ctx, scaleName, interpType));
    }
    case ScaleType::ORDINAL:
      return ScaleShPtr(new OrdinalScale<DomainType, RangeType>(obj, objPath, ctx, scaleName));
    case ScaleType::QUANTIZE:
      return ScaleShPtr(new QuantizeScale<DomainType, RangeType>(obj, objPath, ctx, scaleName));
    default:
      THROW_RUNTIME_EX(
          RapidJSONUtils::getJsonParseErrorStr(obj, "Scale type " + to_string(scaleType) + " is unsupported."));
  }
}

static ::Rendering::Colors::ColorType getColorTypeFromScaleInterpType(const ScaleInterpType interpType) {
  switch (interpType) {
    case ScaleInterpType::InterpolateRgb:
      return ::Rendering::Colors::ColorType::RGBA;
    case ScaleInterpType::InterpolateHsl:
    case ScaleInterpType::InterpolateHslLong:
      return ::Rendering::Colors::ColorType::HSL;
    case ScaleInterpType::InterpolateLab:
      return ::Rendering::Colors::ColorType::LAB;
    case ScaleInterpType::InterpolateHcl:
    case ScaleInterpType::InterpolateHclLong:
      return ::Rendering::Colors::ColorType::HCL;
    default:
      throw ::Rendering::RenderError("ScaleInterpType " + std::to_string(static_cast<int>(interpType)) +
                                     " is not a valid color interpolator");
  }

  return ::Rendering::Colors::ColorType::RGBA;
}

std::pair<::Rendering::Colors::ColorType, ScaleInterpType> getScaleRangeColorTypeFromJSONObj(
    const rapidjson::Value& obj) {
  ::Rendering::Colors::ColorType colorType = ::Rendering::Colors::ColorType::RGBA;

  CHECK(obj.IsObject());
  auto itr = obj.FindMember("range");
  CHECK(itr != obj.MemberEnd() && itr->value.IsArray() && itr->value.Size());
  const rapidjson::Value& firstRangeItem = itr->value[0];

  CHECK(firstRangeItem.IsString());

  auto interpType = getScaleInterpTypeFromJSONObj(obj);
  try {
    colorType = getColorTypeFromScaleInterpType(interpType);
  } catch (::Rendering::RenderError& err) {
    colorType = ::Rendering::Colors::getColorTypeFromColorString(firstRangeItem.GetString());
  }

  return std::make_pair(colorType, interpType);
}

template <typename DomainType>
static ScaleShPtr createRangeColorScalePtr(const rapidjson::Value& obj,
                                           const rapidjson::Pointer& objPath,
                                           const QueryRendererContextShPtr& ctx,
                                           const std::string& scaleName,
                                           ScaleType scaleType) {
  ::Rendering::Colors::ColorType colorType;
  ScaleInterpType interpType;
  std::tie(colorType, interpType) = getScaleRangeColorTypeFromJSONObj(obj);

  switch (colorType) {
    case ::Rendering::Colors::ColorType::RGBA:
      return createScalePtr<DomainType, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType, interpType);
    case ::Rendering::Colors::ColorType::HSL:
      return createScalePtr<DomainType, ColorHSL>(obj, objPath, ctx, scaleName, scaleType, interpType);
    case ::Rendering::Colors::ColorType::LAB:
      return createScalePtr<DomainType, ColorLAB>(obj, objPath, ctx, scaleName, scaleType, interpType);
    case ::Rendering::Colors::ColorType::HCL:
      return createScalePtr<DomainType, ColorHCL>(obj, objPath, ctx, scaleName, scaleType, interpType);
    default:
      THROW_RUNTIME_EX("Unsupported color type: " + std::to_string(static_cast<int>(colorType)) +
                       ". Cannot create scale with color range.");
  }
}

template <typename RangeType>
static ScaleShPtr createDomainColorScalePtr(const rapidjson::Value& obj,
                                            const rapidjson::Pointer& objPath,
                                            const QueryRendererContextShPtr& ctx,
                                            const std::string& scaleName,
                                            ScaleType scaleType) {
  rapidjson::Value::ConstMemberIterator itr = obj.FindMember("domain");
  CHECK(itr != obj.MemberEnd() && itr->value.IsArray() && itr->value.Size());
  const rapidjson::Value& firstItem = itr->value[0];

  CHECK(firstItem.IsString());
  auto colorType = ::Rendering::Colors::getColorTypeFromColorString(firstItem.GetString());
  switch (colorType) {
    case ::Rendering::Colors::ColorType::RGBA:
      return createScalePtr<ColorRGBA, RangeType>(obj, objPath, ctx, scaleName, scaleType);
    case ::Rendering::Colors::ColorType::HSL:
      return createScalePtr<ColorHSL, RangeType>(obj, objPath, ctx, scaleName, scaleType);
    case ::Rendering::Colors::ColorType::LAB:
      return createScalePtr<ColorLAB, RangeType>(obj, objPath, ctx, scaleName, scaleType);
    case ::Rendering::Colors::ColorType::HCL:
      return createScalePtr<ColorHCL, RangeType>(obj, objPath, ctx, scaleName, scaleType);
    default:
      THROW_RUNTIME_EX("Unsupported color type: " + std::to_string(static_cast<int>(colorType)) +
                       ". Cannot create scale with color range.");
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

  QueryDataType domainType = getScaleDomainDataTypeFromJSONObj(obj, ctx, scaleType);
  QueryDataType rangeType = getScaleRangeDataTypeFromJSONObj(obj, ctx, scaleType);

  RUNTIME_EX_ASSERT(
      isScaleDomainCompatible(scaleType, domainType),
      RapidJSONUtils::getJsonParseErrorStr(
          obj, "Domain type " + to_string(domainType) + " is not supported for a " + to_string(scaleType) + " scale."));

  RUNTIME_EX_ASSERT(
      isScaleRangeCompatible(scaleType, rangeType),
      RapidJSONUtils::getJsonParseErrorStr(
          obj, "Range type " + to_string(domainType) + " is not supported for a " + to_string(scaleType) + " scale."));

  switch (domainType) {
    case QueryDataType::UINT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<unsigned int, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<unsigned int, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<unsigned int, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::UINT64:
          return createScalePtr<unsigned int, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<unsigned int, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<unsigned int, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<unsigned int>(obj, objPath, ctx, scaleName, scaleType);
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
        case QueryDataType::UINT64:
          return createScalePtr<int, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<int, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<int, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<int>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::UINT64:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<uint64_t, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<uint64_t, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<uint64_t, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::UINT64:
          return createScalePtr<uint64_t, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<uint64_t, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<uint64_t, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::INT64:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<int64_t, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<int64_t, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<int64_t, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::UINT64:
          return createScalePtr<int64_t, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<int64_t, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<int64_t, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<int64_t>(obj, objPath, ctx, scaleName, scaleType);
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
        case QueryDataType::UINT64:
          return createScalePtr<float, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<float, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<float, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<float>(obj, objPath, ctx, scaleName, scaleType);
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
        case QueryDataType::UINT64:
          return createScalePtr<double, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<double, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<double, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<double>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::COLOR:
      switch (rangeType) {
        case QueryDataType::UINT:
          return createDomainColorScalePtr<unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createDomainColorScalePtr<int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createDomainColorScalePtr<float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::UINT64:
          return createDomainColorScalePtr<uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createDomainColorScalePtr<int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createDomainColorScalePtr<double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<ColorRGBA>(obj, objPath, ctx, scaleName, scaleType);
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
        case QueryDataType::UINT64:
          return createScalePtr<std::string, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<std::string, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<std::string, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<std::string>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::BOOL:
      // NOTE: using unsigned ints for booleans. Doing this shouldn't be too wasteful
      // memory-wise as the domains/ranges of any boolean type scale should be no more
      // than two. Ultimately using unsigned ints for 2 reasons:
      // 1) the inability to get a bool pointer for std::vector<bool>
      //    (i.e. bool* boolptr = &boolvec[0]; // this results in errors)
      // 2) although glsl supports a bool type, setting a uniform boolean array
      //    requires using the glUniform{2|3|4}{f|i|ui}, which means the booleans need
      //    to be stored in 32-bit registers when passed as uniforms
      // TODO(croot): one way to possibly avert doing this is to use a ScaleDomainRangeData
      // class template specialization for booleans
      switch (rangeType) {
        case QueryDataType::UINT:
          return createScalePtr<unsigned int, unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT:
          return createScalePtr<unsigned int, int>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::FLOAT:
          return createScalePtr<unsigned int, float>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::UINT64:
          return createScalePtr<unsigned int, uint64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::INT64:
          return createScalePtr<unsigned int, int64_t>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::DOUBLE:
          return createScalePtr<unsigned int, double>(obj, objPath, ctx, scaleName, scaleType);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<unsigned int>(obj, objPath, ctx, scaleName, scaleType);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "domain type is unsupported."));
  }
}

bool isScaleDomainCompatible(const ScaleType scaleType, const QueryDataType domainType) {
  // TODO(croot): put this in the scale classes? It's easier to do here because
  // otherwise we'd have to do template specializations, which would require a lot
  // of extra code, so keeping here for now.
  switch (scaleType) {
    case ScaleType::LOG:
    case ScaleType::POW:
    case ScaleType::SQRT:
      return (domainType == QueryDataType::DOUBLE || domainType == QueryDataType::FLOAT);
    case ScaleType::QUANTIZE:
      return (domainType == QueryDataType::UINT || domainType == QueryDataType::INT ||
              domainType == QueryDataType::UINT64 || domainType == QueryDataType::INT64 ||
              domainType == QueryDataType::FLOAT || domainType == QueryDataType::DOUBLE);
    default:
      return true;
  }

  return true;
}

bool isScaleRangeCompatible(const ScaleType scaleType, const QueryDataType rangeType) {
  return true;
}

bool areTypesCompatible(const QueryDataType srcType, const QueryDataType inType) {
  if (srcType == inType) {
    return true;
  }

  switch (srcType) {
    case QueryDataType::UINT:
    case QueryDataType::INT:
    case QueryDataType::FLOAT:
      return (inType == QueryDataType::UINT || inType == QueryDataType::INT || inType == QueryDataType::FLOAT);
    case QueryDataType::UINT64:
    case QueryDataType::INT64:
    case QueryDataType::DOUBLE:
      return (inType == QueryDataType::UINT || inType == QueryDataType::INT || inType == QueryDataType::FLOAT ||
              inType == QueryDataType::UINT64 || inType == QueryDataType::INT64 || inType == QueryDataType::DOUBLE);
    default:
      return false;
  }
}

bool areTypesCompatible(const std::type_info& srcType, const std::type_info& inType) {
  return areTypesCompatible(convertTypeIdToDataType(srcType), convertTypeIdToDataType(inType));
}

QueryDataType convertTypeIdToDataType(const std::type_info& srcTypeId) {
  if (srcTypeId == typeid(int)) {
    return QueryDataType::INT;
  } else if (srcTypeId == typeid(unsigned int)) {
    return QueryDataType::UINT;
  } else if (srcTypeId == typeid(float)) {
    return QueryDataType::FLOAT;
  } else if (srcTypeId == typeid(int64_t)) {
    return QueryDataType::INT64;
  } else if (srcTypeId == typeid(uint64_t)) {
    return QueryDataType::UINT64;
  } else if (srcTypeId == typeid(double)) {
    return QueryDataType::DOUBLE;
  } else if (srcTypeId == typeid(::Rendering::Colors::ColorRGBA)) {
    return QueryDataType::COLOR;
  } else if (srcTypeId == typeid(::Rendering::Colors::ColorHSL)) {
    return QueryDataType::COLOR;
  } else if (srcTypeId == typeid(::Rendering::Colors::ColorLAB)) {
    return QueryDataType::COLOR;
  } else if (srcTypeId == typeid(::Rendering::Colors::ColorHCL)) {
    return QueryDataType::COLOR;
  } else {
    THROW_RUNTIME_EX("Type id: " + std::string(srcTypeId.name()) + " cannot be converted to a QueryDataType");
  }
}

template <typename C0, typename C1, typename C2, typename C3>
static C0 convertColorType(const QueryDataType type, const boost::any& value) {
  RUNTIME_EX_ASSERT(type == QueryDataType::COLOR,
                    "Converting " + to_string(type) + " to a " + std::string(typeid(C0).name()) + " is unsupported.");

  C0 rtn;
  try {
    rtn = boost::any_cast<C0>(value);
  } catch (const boost::bad_any_cast&) {
    try {
      auto c1 = boost::any_cast<C1>(value);
      convertColor(c1, rtn);
    } catch (const boost::bad_any_cast&) {
      try {
        auto c2 = boost::any_cast<C2>(value);
        convertColor(c2, rtn);
      } catch (const boost::bad_any_cast&) {
        auto c3 = boost::any_cast<C3>(value);
        convertColor(c3, rtn);
      }
    }
  }

  return rtn;
}

template <>
ColorRGBA convertType(const QueryDataType type, const boost::any& value) {
  return convertColorType<ColorRGBA, ColorHSL, ColorLAB, ColorHCL>(type, value);
}

template <>
ColorHSL convertType(const QueryDataType type, const boost::any& value) {
  return convertColorType<ColorHSL, ColorRGBA, ColorLAB, ColorHCL>(type, value);
}

template <>
ColorLAB convertType(const QueryDataType type, const boost::any& value) {
  return convertColorType<ColorLAB, ColorRGBA, ColorHSL, ColorHCL>(type, value);
}

template <>
ColorHCL convertType(const QueryDataType type, const boost::any& value) {
  return convertColorType<ColorHCL, ColorRGBA, ColorHSL, ColorLAB>(type, value);
}

}  // namespace QueryRenderer
