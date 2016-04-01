#include "QueryRendererObjects.h"
#include "../QueryRendererContext.h"
#include "shaders/pointTemplate_vert.h"
#include "shaders/pointTemplate_frag.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"

#include "../QueryEngine/Execute.h"

#include <Rendering/Renderer/GL/TypeGL.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

#include <boost/algorithm/string/replace.hpp>
// #include <boost/algorithm/string/find.hpp>
// #include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
// #include <boost/algorithm/string/find_iterator.hpp>
// #include <boost/algorithm/string/regex_find_format.hpp>

// #include <iostream>
// #include <sstream>
// #include <utility>
// #include <functional>

// // CROOT - remote the following includes -- only used for debugging
// // #include <fstream>

namespace QueryRenderer {

using ::Rendering::Window;
using ::Rendering::Objects::ColorRGBA;
using ::Rendering::GL::BaseTypeGL;
using ::Rendering::GL::TypeGLShPtr;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

using ::Rendering::GL::Resources::GLShader;
using ::Rendering::GL::Resources::GLVertexBufferShPtr;
using ::Rendering::GL::Resources::GLBufferAttrType;
using ::Rendering::GL::Resources::GLVertexArray;

using ::Rendering::GL::Resources::VboAttrToShaderAttrPair;
using ::Rendering::GL::Resources::VboAttrToShaderAttrList;
using ::Rendering::GL::Resources::VboAttrToShaderAttrMap;

typedef std::string::iterator str_itr;
typedef boost::iterator_range<str_itr> str_itr_range;

std::string to_string(BaseScale::ScaleType scaleType) {
  switch (scaleType) {
    case BaseScale::ScaleType::LINEAR:
      return "LINEAR";
    case BaseScale::ScaleType::ORDINAL:
      return "ORDINAL";
    case BaseScale::ScaleType::UNDEFINED:
      return "UNDEFINED";
    default:
      return "<scale type " + std::to_string(static_cast<int>(scaleType)) + ">";
  }

  return "";
}

str_itr_range getGLSLFunctionBounds(std::string& codeStr, const std::string& funcName) {
  std::string regexStr = "\\h*\\w+\\h+" + funcName + "\\h*\\([\\w\\h\\v,]*\\)\\h*\\v*\\{";
  boost::regex funcSignatureRegex(regexStr);

  str_itr_range signature_range = boost::find_regex(codeStr, funcSignatureRegex);

  if (signature_range.empty()) {
    return signature_range;
  }

  str_itr lastItr = signature_range.end();
  std::vector<str_itr> scopeStack = {lastItr - 1};

  size_t curr_pos = signature_range.end() - codeStr.begin();
  while ((curr_pos = codeStr.find_first_of("{}", curr_pos)) != std::string::npos) {
    if (codeStr[curr_pos] == '{') {
      scopeStack.push_back(codeStr.begin() + curr_pos);
    } else {
      // found a '}'
      scopeStack.pop_back();
      if (scopeStack.empty()) {
        lastItr = codeStr.begin() + curr_pos + 1;
        break;
      }
    }
  }

  if (!scopeStack.empty()) {
    // return an empty range
    return str_itr_range();
  }

  return str_itr_range(signature_range.begin(), lastItr);
}

QueryDataType getDataTypeFromJSONObj(const rapidjson::Value& obj, bool supportString = false) {
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

QueryDataType getDataTypeFromDataRefJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "data reference is not a JSON object."));

  rapidjson::Value::ConstMemberIterator mitr;
  RUNTIME_EX_ASSERT(
      (mitr = obj.FindMember("data")) != obj.MemberEnd() && mitr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "data reference object doesn't contain a \"data\" string property."));
  std::string dataTableName = mitr->value.GetString();
  QueryDataTableVBOShPtr tablePtr = ctx->getDataTable(dataTableName);

  RUNTIME_EX_ASSERT(tablePtr != nullptr,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "data table \"" + dataTableName + "\" doesn't exist."));

  RUNTIME_EX_ASSERT(
      (mitr = obj.FindMember("field")) != obj.MemberEnd() && mitr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "data reference object must contain a \"field\" string property."));
  return tablePtr->getColumnType(mitr->value.GetString());
}

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    LinearScaleTemplate_vert::source,  // LINEAR
    OrdinalScaleTemplate_vert::source  // ORDINAL
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     BaseScale::ScaleType type)
    : _name(name),
      _type(type),
      _useClamp(false),
      _domainDataType(domainDataType),
      _domainTypeGL(nullptr),
      _rangeDataType(rangeDataType),
      _rangeTypeGL(nullptr),
      _ctx(ctx) {
}

BaseScale::BaseScale(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     BaseScale::ScaleType type)
    : BaseScale(ctx, domainDataType, rangeDataType, name, type) {
  _initFromJSONObj(obj, objPath);
}

BaseScale::~BaseScale() {
}

bool BaseScale::_initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = false;
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    RUNTIME_EX_ASSERT(
        obj.IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "scale items must be objects."));

    if (!_name.length()) {
      _name = getScaleNameFromJSONObj(obj);
    }

    if (_type == ScaleType::UNDEFINED) {
      _type = getScaleTypeFromJSONObj(obj);
    }

    if (_type == ScaleType::LINEAR) {
      // TODO(croot): create a separate derived class per scale?
      // once we do this, there's no need to do any updates to the BaseScale class
      // since it would only handle _name & _type & if either of those ever
      // change, a new object is built. Therefore the _initFromJSONObj() call in
      // the derived class's updateFromJSONObj() can be removed and this _initFromJSONObj()
      // method can be made private.
      // Also, by doing this, the _jsonPath property can be fullty managed
      // by the derived class, so this base class can provide a "_setJsonPath()"
      // method or something.
      rapidjson::Value::ConstMemberIterator itr;

      // TODO(croot): move the "clamp" prop name into a const somewhere.
      std::string clampProp = "clamp";

      bool prevClamp = _useClamp;
      if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(
            itr->value.IsBool(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "the \"clamp\" property for linear scales must be a boolean."));

        _useClamp = itr->value.GetBool();
      } else {
        // TODO(croot): set a const default for _useClamp somewhere
        _useClamp = false;
      }

      if (prevClamp != _useClamp) {
        _clampChanged = true;
      } else {
        _clampChanged = false;
      }
    }

    rtn = true;
  }

  return rtn;
}

const ::Rendering::GL::TypeGLShPtr& BaseScale::getDomainTypeGL() {
  RUNTIME_EX_ASSERT(_domainTypeGL != nullptr,
                    std::string(*this) + " getDomainTypeGL(): the domain type is uninitialized.");
  return _domainTypeGL;
}

const ::Rendering::GL::TypeGLShPtr& BaseScale::getRangeTypeGL() {
  RUNTIME_EX_ASSERT(_rangeTypeGL != nullptr,
                    std::string(*this) + " getRangeTypeGL(): the range type is uninitialized.");

  return _rangeTypeGL;
}

std::string BaseScale::getScaleGLSLFuncName(const std::string& extraSuffix) {
  std::string scaleName;

  switch (_type) {
    case ScaleType::LINEAR:
      scaleName = "Linear";
      break;
    case ScaleType::ORDINAL:
      scaleName = "Ordinal";
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + " getScaleGLSLFuncName(): scale type is not supported.");
  }

  return "get" + scaleName + "Scale_" + _name + extraSuffix;
}

std::string BaseScale::_printInfo() const {
  return "(name: " + _name + ", type: " + to_string(_type) + ") " + to_string(_ctx->getUserWidgetIds());
}

template <typename T>
void ScaleDomainRangeData<T>::initializeFromJSONObj(const rapidjson::Value& obj,
                                                    const rapidjson::Pointer& objPath,
                                                    const QueryRendererContextShPtr& ctx,
                                                    BaseScale::ScaleType type,
                                                    bool& sizeChanged,
                                                    bool& valsChanged) {
  sizeChanged = false;
  valsChanged = false;

  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  bool isObject = false, isString = false;
  QueryDataTableVBOShPtr tablePtr;
  DataColumnShPtr columnPtr;

  mitr = obj.FindMember(_name.c_str());
  RUNTIME_EX_ASSERT(mitr != obj.MemberEnd(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        ctx->getUserWidgetIds(), obj, "scale objects must have a \"" + _name + "\" property."));

  if (!ctx->isJSONCacheUpToDate(_jsonPath, mitr->value)) {
    if (_useString) {
      RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
                         (mitr->value.IsArray() && mitr->value.Size())),
                        RapidJSONUtils::getJsonParseErrorStr(
                            ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
    } else {
      RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())),
                        RapidJSONUtils::getJsonParseErrorStr(
                            ctx->getUserWidgetIds(), mitr->value, "scale " + _name + " is invalid."));
    }

    const rapidjson::Value& jsonObj = mitr->value;

    int prevSz = 0;
    if (_vectorPtr) {
      prevSz = _vectorPtr->size();
    }
    if (isObject) {
      RUNTIME_EX_ASSERT(
          (mitr = jsonObj.FindMember("data")) != jsonObj.MemberEnd() && mitr->value.IsString(),
          RapidJSONUtils::getJsonParseErrorStr(
              ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"data\" string property."));
      tablePtr = ctx->getDataTable(mitr->value.GetString());

      RUNTIME_EX_ASSERT(
          (mitr = jsonObj.FindMember("field")) != jsonObj.MemberEnd() && mitr->value.IsString(),
          RapidJSONUtils::getJsonParseErrorStr(
              ctx->getUserWidgetIds(), jsonObj, "scale data reference must have a \"field\" string property."));

      // Only supports hand-written data right now.
      // TODO(croot): Support query result vbo -- this is somewhat
      // tricky, because in order to do so we'd have to use compute
      // shaders/cuda to do min/max/other math stuff, and uniform buffers or
      // storage buffers to send the values as uniforms
      RUNTIME_EX_ASSERT(tablePtr->getType() == QueryDataTableType::EMBEDDED,
                        RapidJSONUtils::getJsonParseErrorStr(
                            ctx->getUserWidgetIds(),
                            jsonObj,
                            "scale data references only support JSON-embedded data tables currently."));

      DataTable* dataTablePtr = dynamic_cast<DataTable*>(tablePtr.get());

      columnPtr = dataTablePtr->getColumn(mitr->value.GetString());

      TDataColumn<T>* dataColumnPtr = dynamic_cast<TDataColumn<T>*>(columnPtr.get());

      _vectorPtr = dataColumnPtr->getColumnData();

      _updateVectorDataByType(dataColumnPtr, type);
    } else if (isString) {
      _setFromStringValue(mitr->value.GetString(), type);
    } else {
      _vectorPtr.reset(new std::vector<T>(jsonObj.Capacity()));

      // gather all the items
      QueryDataType itemType;

      bool supportStr = (typeid(T) == typeid(std::string));
      size_t idx = 0;
      for (vitr = jsonObj.Begin(); vitr != jsonObj.End(); ++vitr, ++idx) {
        // only strings are currently allowed in the domains
        itemType = getDataTypeFromJSONObj(*vitr, supportStr);

        RUNTIME_EX_ASSERT(
            itemType == dataType,
            RapidJSONUtils::getJsonParseErrorStr(
                ctx->getUserWidgetIds(),
                jsonObj,
                "scale " + _name + " item " + std::to_string(vitr - jsonObj.Begin()) + " has an invalid type."));

        // _pushItem(*vitr);
        _setItem(idx, *vitr);
      }
    }

    int newSz = 0;
    if (_vectorPtr) {
      newSz = _vectorPtr->size();
    }

    // only need to regenerate the shader if the size of the
    // scale data changed, but we need to update any coerced
    // values if the scale's values changed, which means
    // we need to distinguish between when the size of the
    // data changed vs only the values changed.
    valsChanged = true;
    sizeChanged = (prevSz != newSz);
  }

  updateJSONPath(objPath);
}

template <typename T>
void ScaleDomainRangeData<T>::updateJSONPath(const rapidjson::Pointer& objPath) {
  _jsonPath = objPath.Append(_name.c_str(), _name.length());
}

template <typename T>
void ScaleDomainRangeData<T>::_pushItem(const rapidjson::Value& obj) {
  _vectorPtr->push_back(getDataValueFromJSONObj(obj));
}

template <typename T>
void ScaleDomainRangeData<T>::_setItem(size_t idx, const rapidjson::Value& obj) {
  (*_vectorPtr)[idx] = getDataValueFromJSONObj(obj);
}

template <typename T>
inline const TypeGLShPtr& ScaleDomainRangeData<T>::getTypeGL() {
  if (!_cachedTypeGL) {
    _cachedTypeGL.reset(new ::Rendering::GL::TypeGL<T>());
  }
  return _cachedTypeGL;
}

template <>
inline const TypeGLShPtr& ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
  // TODO(croot): combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions
  if (!_cachedTypeGL) {
    _cachedTypeGL.reset(new ::Rendering::GL::TypeGL<float, 4>());
  }
  return _cachedTypeGL;
}

template <>
inline const TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL() {
  if (_cachedTypeGL) {
    // NOTE: we should never get in here, but just in case :)
    _cachedTypeGL = nullptr;
  }
  return _cachedTypeGL;
}

template <>
const QueryDataType ScaleDomainRangeData<unsigned int>::dataType = QueryDataType::UINT;

template <>
unsigned int ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetUint();
}

template <>
const QueryDataType ScaleDomainRangeData<int>::dataType = QueryDataType::INT;

template <>
int ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetInt();
}

template <>
const QueryDataType ScaleDomainRangeData<float>::dataType = QueryDataType::FLOAT;

template <>
float ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return static_cast<float>(obj.GetDouble());
}

template <>
const QueryDataType ScaleDomainRangeData<double>::dataType = QueryDataType::DOUBLE;

template <>
double ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetDouble();
}

template <>
const QueryDataType ScaleDomainRangeData<ColorRGBA>::dataType = QueryDataType::COLOR;

template <>
ColorRGBA ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorRGBA(obj.GetString());
}

template <>
const QueryDataType ScaleDomainRangeData<std::string>::dataType = QueryDataType::STRING;

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetString();
}

template <typename T>
void ScaleDomainRangeData<T>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
  if (type == BaseScale::ScaleType::LINEAR && (strVal == "width" || strVal == "height")) {
    // TODO(croot): Handle actual width/height vals
    _vectorPtr.reset(new std::vector<T>({0, 1}));

    // need to put the width/height range from -1 to 1. Didn't do that
    // in the previous line because the compiler was complaining that
    // you cannot convert -1 to an unsigned int, tho if range is
    // an unsigned int, you should never get here. Doing a subtraction here
    // to get a -1 to 1 range without the compilation error.

    (*_vectorPtr)[0] -= 1;
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                          to_string(type) + " for domain/range is invalid."));
  }
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " +
                                                        to_string(type) + " color domain/range is invalid."));
}

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
  THROW_RUNTIME_EX(
      RapidJSONUtils::getJsonParseErrorStr("string value \"" + strVal + "\" for scale type " + to_string(type) +
                                           " for string domain/range is invalid. It must be an array of strings."));
}

template <typename T>
void ScaleDomainRangeData<T>::_updateVectorDataByType(TDataColumn<T>* dataColumnPtr, BaseScale::ScaleType type) {
  if (type == BaseScale::ScaleType::LINEAR) {
    std::pair<T, T> minmaxDomain = dataColumnPtr->getExtrema();

    _vectorPtr.reset(new std::vector<T>({minmaxDomain.first, minmaxDomain.second}));
  }
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA>* dataColumnPtr,
                                                              BaseScale::ScaleType type) {
  if (type == BaseScale::ScaleType::LINEAR) {
    // TODO(croot): update ColorRGBA to be supported by the <,>,etc. operators
    // so that the getExtrema() call will work to get the min and max colors
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        "getting the extrema of colors in the domain/range is unsupported for scale type" + to_string(type) + "."));
  }
}

template <typename T>
ScaleDomainRangeData<T>::operator std::string() const {
  return "ScaleDomainRangeData<T>(name: " + _name + ")";
}

template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::Scale(const rapidjson::Value& obj,
                                    const rapidjson::Pointer& objPath,
                                    const QueryRendererContextShPtr& ctx,
                                    const std::string& name,
                                    BaseScale::ScaleType type)
    : BaseScale(obj, objPath, ctx, getDataTypeForType<DomainType>(), getDataTypeForType<RangeType>(), name, type),
      _domainPtr("domain", false),
      _rangePtr("range", true),
      _defaultVal() {
  _initGLTypes();
  updateFromJSONObj(obj, objPath);
}

template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::~Scale() {
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_setDefaultFromJSONObj(const rapidjson::Value& obj,
                                                          const rapidjson::Pointer& objPath) {
  // TODO(croot): make different derived scale classes per scale type
  // and move this function only into the "OrdinalScale" derived class
  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot): expose "default" as a constant somewhere;
  std::string defaultStr = "default";

  if (!_ctx->isJSONCacheUpToDate(_defaultJsonPath, obj)) {
    if ((mitr = obj.FindMember(defaultStr.c_str())) != obj.MemberEnd()) {
      QueryDataType itemType = getDataTypeFromJSONObj(mitr->value);

      RUNTIME_EX_ASSERT(
          itemType == _rangePtr.dataType,
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                               mitr->value,
                                               "scale \"" + _name + "\" default is not the same type as its range."));

      _defaultVal = _rangePtr.getDataValueFromJSONObj(mitr->value);
    } else {
      // set an undefined default
      _defaultVal = RangeType();
    }
  }

  _updateDefaultJSONPath(objPath);
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_updateDefaultJSONPath(const rapidjson::Pointer& objPath) {
  // TODO(croot): expose "default" as a constant somewhere;
  std::string defaultStr = "default";
  _defaultJsonPath = objPath.Append(defaultStr.c_str(), defaultStr.length());
}

template <typename DomainType, typename RangeType>
bool Scale<DomainType, RangeType>::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = BaseScale::_initFromJSONObj(obj, objPath);

  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    _domainPtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _domainSizeChanged, _domainValsChanged);
    _rangePtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _rangeSizeChanged, _rangeValsChanged);

    if (_type == BaseScale::ScaleType::ORDINAL) {
      _setDefaultFromJSONObj(obj, objPath);
    }

    rtn = rtn || hasDomainDataChanged() || hasRangeDataChanged();
  } else if (_jsonPath != objPath) {
    _domainSizeChanged = _domainValsChanged = _rangeSizeChanged = _rangeValsChanged = true;

    _domainPtr.updateJSONPath(objPath);
    _rangePtr.updateJSONPath(objPath);
    _updateDefaultJSONPath(objPath);
  }

  _jsonPath = objPath;

  return rtn;
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_initGLTypes() {
  _domainTypeGL = _domainPtr.getTypeGL();
  _rangeTypeGL = _rangePtr.getTypeGL();
}

template <typename DomainType, typename RangeType>
std::string Scale<DomainType, RangeType>::getGLSLCode(const std::string& extraSuffix,
                                                      bool ignoreDomain,
                                                      bool ignoreRange) {
  RUNTIME_EX_ASSERT(_domainPtr.size() > 0 && _rangePtr.size() > 0,
                    std::string(*this) + " getGLSLCode(): domain/range of scale \"" + _name + "\" has no value.");

  std::string shaderCode = scaleVertexShaderSource[static_cast<int>(_type)];
  std::ostringstream ss;

  if (!ignoreDomain) {
    boost::replace_first(shaderCode, "<domainType>", _domainTypeGL->glslType());
  }

  if (!ignoreRange) {
    boost::replace_first(shaderCode, "<rangeType>", _rangeTypeGL->glslType());
  }

  ss << _domainPtr.size();
  boost::replace_first(shaderCode, "<numDomains>", ss.str());

  ss.str("");
  ss << _rangePtr.size();
  boost::replace_first(shaderCode, "<numRanges>", ss.str());

  boost::replace_all(shaderCode, "<name>", _name + extraSuffix);

  if (_type == ScaleType::LINEAR) {
    ss.str("");
    ss << _useClamp;
    boost::replace_all(shaderCode, "<useClamp>", ss.str());
  }

  shaderCode += '\n';

  return shaderCode;
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::bindUniformsToRenderer(GLShader* activeShader,
                                                          const std::string& extraSuffix,
                                                          bool ignoreDomain,
                                                          bool ignoreRange) {
  if (!ignoreDomain) {
    activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorData());
  }

  if (!ignoreRange) {
    activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorData());
  }

  if (_type == ScaleType::ORDINAL) {
    activeShader->setUniformAttribute(getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
  }
}

template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::operator std::string() const {
  return "Scale<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + ">" +
         _printInfo();
}

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "scale items must be JSON objects."));

  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT(
      (itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "scale objects must contain a \"name\" string property."));

  return itr->value.GetString();
}

BaseScale::ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj) {
  // TODO(croot): expose default as a static attr.
  BaseScale::ScaleType rtn = BaseScale::ScaleType::LINEAR;

  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "scale items must be JSON objects."));

  rapidjson::Value::ConstMemberIterator itr;
  if ((itr = obj.FindMember("type")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(
        itr->value.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(itr->value, "\"type\" property in scale objects must be a string."));
    std::string strScaleType(itr->value.GetString());

    if (strScaleType == "linear") {
      rtn = BaseScale::ScaleType::LINEAR;
    } else if (strScaleType == "ordinal") {
      rtn = BaseScale::ScaleType::ORDINAL;
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

  QueryDataType domainType;

  if (isObject) {
    domainType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else {
    // TODO(croot): Probably need to have specific classes to correspond
    // to the different scales. For example, ordinal/categorical scales
    // can support strings for domain values. Others shouldn't.
    // Will allow all domains to accept all strings for now.
    domainType = getDataTypeFromJSONObj(itr->value[0], true);
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

  QueryDataType rangeType;
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
  }

  return rangeType;
}

ScaleShPtr createScale(const rapidjson::Value& obj,
                       const rapidjson::Pointer& objPath,
                       const QueryRendererContextShPtr& ctx,
                       const std::string& name,
                       BaseScale::ScaleType type) {
  std::string scaleName(name);
  if (!scaleName.length()) {
    scaleName = getScaleNameFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleName.length() > 0,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "Scales must have a \"name\" property"));

  BaseScale::ScaleType scaleType(type);
  if (scaleType == BaseScale::ScaleType::UNDEFINED) {
    scaleType = getScaleTypeFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleType != BaseScale::ScaleType::UNDEFINED,
                    RapidJSONUtils::getJsonParseErrorStr(obj, "Scale type for \"" + scaleName + "\" is undefined."));

  rapidjson::Value::ConstMemberIterator itr;

  QueryDataType domainType = getScaleDomainDataTypeFromJSONObj(obj, ctx);
  QueryDataType rangeType = getScaleRangeDataTypeFromJSONObj(obj, ctx);

  switch (domainType) {
    case QueryDataType::UINT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<unsigned int, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<unsigned int, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<unsigned int, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<unsigned int, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<unsigned int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::INT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<int, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<int, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<int, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<int, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::FLOAT:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<float, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<float, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<float, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<float, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<float, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::DOUBLE:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<double, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<double, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<double, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<double, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<double, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::COLOR:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<ColorRGBA, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<ColorRGBA, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<ColorRGBA, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<ColorRGBA, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<ColorRGBA, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    case QueryDataType::STRING:
      switch (rangeType) {
        case QueryDataType::UINT:
          return ScaleShPtr(new Scale<std::string, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::INT:
          return ScaleShPtr(new Scale<std::string, int>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::FLOAT:
          return ScaleShPtr(new Scale<std::string, float>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::DOUBLE:
          return ScaleShPtr(new Scale<std::string, double>(obj, objPath, ctx, scaleName, scaleType));
        case QueryDataType::COLOR:
          return ScaleShPtr(new Scale<std::string, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
              obj, "range type is unsupported: " + std::to_string(static_cast<int>(rangeType))));
      }
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(obj, "domain type is unsupported."));
  }
}

std::string BaseScaleRef::_printInfo() const {
  std::string rtn = to_string(_ctx->getUserWidgetIds());
  if (_scalePtr) {
    rtn += ", scale reference: " + std::string(*_scalePtr);
  }
  if (_rndrPropPtr) {
    rtn += ", render property: " + std::string(*_rndrPropPtr);
  }

  return rtn;
}

template <class T, class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<T>>& destData, ScaleDomainRangeData<TT>* srcData) {
  std::vector<TT>& srcVec = srcData->getVectorData();

  destData.reset(new ScaleDomainRangeData<T>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<T>& destVec = destData->getVectorData();
  for (size_t i = 0; i < srcVec.size(); ++i) {
    destVec[i] = static_cast<T>(srcVec[i]);
  }
}

template <class T>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<T>>& destData,
                            ScaleDomainRangeData<ColorRGBA>* srcData) {
  THROW_RUNTIME_EX("Cannot convert a color (" + std::string(*srcData) + ") to a numeric value (" +
                   std::string(*destData) + ").");
}

template <class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<ColorRGBA>>& destData,
                            ScaleDomainRangeData<TT>* srcData) {
  THROW_RUNTIME_EX("Cannot convert a numeric value (" + std::string(*srcData) + ") to a color (" +
                   std::string(*destData) + ").");
}

void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<ColorRGBA>>& destData,
                            ScaleDomainRangeData<ColorRGBA>* srcData) {
  std::vector<ColorRGBA>& srcVec = srcData->getVectorData();

  destData.reset(new ScaleDomainRangeData<ColorRGBA>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<ColorRGBA>& destVec = destData->getVectorData();
  for (size_t i = 0; i < srcVec.size(); ++i) {
    destVec[i] = srcVec[i];
  }
}

template <typename DomainType, typename RangeType>
ScaleRef<DomainType, RangeType>::ScaleRef(const QueryRendererContextShPtr& ctx,
                                          const ScaleShPtr& scalePtr,
                                          BaseRenderProperty* rndrProp)
    : BaseScaleRef(ctx, scalePtr, rndrProp), _coercedDomainData(nullptr), _coercedRangeData(nullptr) {
  _updateDomainRange(true, true);
}

template <typename DomainType, typename RangeType>
void ScaleRef<DomainType, RangeType>::_updateDomainRange(bool updateDomain, bool updateRange, bool force) {
  CHECK(_scalePtr != nullptr);

  ScaleDomainRangeData<unsigned int>* uintDomain;
  ScaleDomainRangeData<int>* intDomain;
  ScaleDomainRangeData<float>* floatDomain;
  ScaleDomainRangeData<double>* doubleDomain;
  ScaleDomainRangeData<std::string>* stringDomain;
  ScaleDomainRangeData<ColorRGBA>* colorDomain;
  bool doSort = false;

  if (updateDomain) {
    BaseScaleDomainRangeData* domainDataPtr = _scalePtr->getDomainData();
    if (force || domainDataPtr->getTypeInfo() != typeid(DomainType)) {
      uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(domainDataPtr);
      intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(domainDataPtr);
      floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(domainDataPtr);
      doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(domainDataPtr);
      stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(domainDataPtr);

      if (uintDomain) {
        convertDomainRangeData(_coercedDomainData, uintDomain);
      } else if (intDomain) {
        convertDomainRangeData(_coercedDomainData, intDomain);
      } else if (floatDomain) {
        convertDomainRangeData(_coercedDomainData, floatDomain);
      } else if (doubleDomain) {
        convertDomainRangeData(_coercedDomainData, doubleDomain);
      } else if (stringDomain) {
        _doStringToDataConversion(stringDomain);
        doSort = true;
      } else {
        THROW_RUNTIME_EX(std::string(*this) + ": Cannot create scale reference - unsupported domain type.");
      }
    } else {
      _coercedDomainData = nullptr;
    }
  }

  if (updateRange) {
    BaseScaleDomainRangeData* rangeDataPtr = _scalePtr->getRangeData();
    if (force || rangeDataPtr->getTypeInfo() != typeid(RangeType)) {
      uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(rangeDataPtr);
      intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(rangeDataPtr);
      floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(rangeDataPtr);
      doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(rangeDataPtr);
      stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(rangeDataPtr);
      colorDomain = dynamic_cast<ScaleDomainRangeData<ColorRGBA>*>(rangeDataPtr);

      if (uintDomain) {
        convertDomainRangeData(_coercedRangeData, uintDomain);
      } else if (intDomain) {
        convertDomainRangeData(_coercedRangeData, intDomain);
      } else if (floatDomain) {
        convertDomainRangeData(_coercedRangeData, floatDomain);
      } else if (doubleDomain) {
        convertDomainRangeData(_coercedRangeData, doubleDomain);
      } else if (force && colorDomain) {
        convertDomainRangeData(_coercedRangeData, colorDomain);
      } else {
        THROW_RUNTIME_EX(std::string(*this) + ": Cannot create scale reference - unsupported range type.");
      }
    } else {
      _coercedRangeData = nullptr;
    }
  }

  if (doSort) {
    _sort();
  }
}

template <typename DomainType, typename RangeType>
void ScaleRef<DomainType, RangeType>::updateScaleRef(const ScaleShPtr& scalePtr) {
  if (scalePtr != _scalePtr) {
    _scalePtr = scalePtr;
  }

  _updateDomainRange(_scalePtr->hasDomainDataChanged(), _scalePtr->hasRangeDataChanged());
}

template <typename DomainType, typename RangeType>
const TypeGLShPtr& ScaleRef<DomainType, RangeType>::getDomainTypeGL() {
  _verifyScalePointer();
  if (_coercedDomainData) {
    return _coercedDomainData->getTypeGL();
  }

  return _scalePtr->getDomainTypeGL();
}

template <typename DomainType, typename RangeType>
const TypeGLShPtr& ScaleRef<DomainType, RangeType>::getRangeTypeGL() {
  _verifyScalePointer();
  if (_coercedRangeData) {
    return _coercedRangeData->getTypeGL();
  }

  return _scalePtr->getRangeTypeGL();
}

template <typename DomainType, typename RangeType>
std::string ScaleRef<DomainType, RangeType>::getGLSLCode(const std::string& extraSuffix) {
  _verifyScalePointer();

  bool coerceDomain = (_coercedDomainData != nullptr);
  bool coerceRange = (_coercedRangeData != nullptr);

  // force to ignore the domain of the original scale
  std::string shaderCode = _scalePtr->getGLSLCode(extraSuffix, coerceDomain, coerceRange);

  if (coerceDomain) {
    boost::replace_first(shaderCode, "<domainType>", _coercedDomainData->getTypeGL()->glslType());
  }

  if (coerceRange) {
    boost::replace_first(shaderCode, "<rangeType>", _coercedRangeData->getTypeGL()->glslType());
  }

  return shaderCode;
}

template <typename DomainType, typename RangeType>
void ScaleRef<DomainType, RangeType>::bindUniformsToRenderer(GLShader* activeShader, const std::string& extraSuffix) {
  _verifyScalePointer();

  bool coerceDomain = (_coercedDomainData != nullptr);
  bool coerceRange = (_coercedRangeData != nullptr);

  if (coerceDomain) {
    activeShader->setUniformAttribute(_scalePtr->getDomainGLSLUniformName() + extraSuffix,
                                      _coercedDomainData->getVectorData());
  }

  if (coerceRange) {
    activeShader->setUniformAttribute(_scalePtr->getRangeGLSLUniformName() + extraSuffix,
                                      _coercedRangeData->getVectorData());
  }

  _scalePtr->bindUniformsToRenderer(activeShader, extraSuffix, coerceDomain, coerceRange);
}

template <typename DomainType, typename RangeType>
void ScaleRef<DomainType, RangeType>::_doStringToDataConversion(ScaleDomainRangeData<std::string>* domainData) {
  const Executor* const executor = _ctx->getExecutor();

  RUNTIME_EX_ASSERT(executor != nullptr,
                    std::string(*this) + ": An executor is not defined. Cannot numerically convert a string column.");

  const QueryDataTableVBOShPtr& dataTable = _rndrPropPtr->getDataTablePtr();

  RUNTIME_EX_ASSERT(dataTable != nullptr,
                    std::string(*this) + ": A data table is not referenced by render property \"" +
                        _rndrPropPtr->getName() + "\". Cannot numerically convert a string column.");

  SqlQueryDataTable* sqlDataTable = dynamic_cast<SqlQueryDataTable*>(dataTable.get());

  RUNTIME_EX_ASSERT(sqlDataTable != nullptr,
                    std::string(*this) + ": The data table referenced by render property \"" + _rndrPropPtr->getName() +
                        "\" is not an sql data table. Cannot numerically convert a string column");

  std::string tableName = sqlDataTable->getTableName();
  RUNTIME_EX_ASSERT(tableName.length() != 0,
                    std::string(*this) + "The sql data table referenced by render property \"" +
                        _rndrPropPtr->getName() +
                        "\" is not properly initialized. It is missing a \"dbTableName\" property. Cannot numerically "
                        "convert a string column");

  std::string colName = _rndrPropPtr->getDataColumnName();
  RUNTIME_EX_ASSERT(
      colName.length() != 0,
      std::string(*this) + ": The render property \"" + _rndrPropPtr->getName() +
          "\" is missing a column name to reference in the data. Cannot numerically convert a string column.");

  const QueryDataLayoutShPtr& queryDataLayoutPtr = _ctx->getQueryDataLayout();
  CHECK(queryDataLayoutPtr != nullptr);

  std::vector<std::string>& vec = domainData->getVectorData();
  _coercedDomainData.reset(
      new ScaleDomainRangeData<DomainType>(domainData->getName(), vec.size(), domainData->useString()));

  std::vector<DomainType>& coercedVec = _coercedDomainData->getVectorData();
  for (size_t i = 0; i < vec.size(); ++i) {
    // get data from the executor
    coercedVec[i] =
        static_cast<DomainType>(executor->getStringId(tableName, colName, vec[i], queryDataLayoutPtr.get()));
  }
}

template <typename DomainType, typename RangeType>
void ScaleRef<DomainType, RangeType>::_sort() {
  _verifyScalePointer();

  bool hasDomain = (_coercedDomainData != nullptr);
  bool hasRange = (_coercedRangeData != nullptr);

  // force a copy of both the domain and range to sort
  _updateDomainRange(!hasDomain, !hasRange, true);

  // TODO(croot): somehow do a sort in place? Not sure how to do this without
  // creating an iterator class on the ScaleRef objects (which might be nice
  // to do in the future). So, for now, I'm just copying all the domain/range
  // data as pairs into a vector, sorting that vector based on the domain, and
  // placing the results back. Very hacky, but since domains/ranges should on the
  // whole be small, this shouldn't be a big bottle neck.

  // TODO(croot): Possible bug -- the size of the domains/ranges don't have to
  // be equal. You can have more domains than ranges and vice-versa. So we need
  // to sort by the smaller of the two and leave the hanging items alone.

  std::vector<DomainType>& domainVec = _coercedDomainData->getVectorData();
  std::vector<RangeType>& rangeVec = _coercedRangeData->getVectorData();

  int numItems = std::min(domainVec.size(), rangeVec.size());
  std::vector<std::pair<DomainType, RangeType>> sortVec(numItems);

  int i;
  for (i = 0; i < numItems; ++i) {
    sortVec[i] = std::make_pair(domainVec[i], rangeVec[i]);
  }

  std::sort(sortVec.begin(),
            sortVec.end(),
            [](const std::pair<DomainType, RangeType>& a,
               const std::pair<DomainType, RangeType>& b) { return a.first < b.first; });

  for (i = 0; i < numItems; ++i) {
    domainVec[i] = sortVec[i].first;
    rangeVec[i] = sortVec[i].second;
  }
}

template <typename DomainType, typename RangeType>
ScaleRef<DomainType, RangeType>::operator std::string() const {
  return "ScaleRef<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + "> " +
         _printInfo();
}

void setRenderPropertyTypeInShaderSrc(const BaseRenderProperty& prop, std::string& shaderSrc) {
  std::ostringstream in_ss, out_ss;

  std::string inname = prop.getInGLSLName();
  std::string intype = prop.getInGLSLType();

  in_ss << "<" << inname << "Type"
        << ">";
  boost::replace_first(shaderSrc, in_ss.str(), intype);

  std::string outname = prop.getOutGLSLName();
  std::string outtype = prop.getOutGLSLType();

  out_ss << "<" << outname << "Type"
         << ">";
  boost::replace_first(shaderSrc, out_ss.str(), outtype);
}

void setRenderPropertyAttrTypeInShaderSrc(const BaseRenderProperty& prop, std::string& shaderSrc, bool isUniform) {
  std::ostringstream ss;

  std::string name = prop.getName();

  ss << "<useU" << name << ">";
  boost::replace_first(shaderSrc, ss.str(), (isUniform ? "1" : "0"));
}

void BaseRenderProperty::initializeFromJSONObj(const rapidjson::Value& obj,
                                               const rapidjson::Pointer& objPath,
                                               const QueryDataTableVBOShPtr& dataPtr) {
  if (obj.IsObject()) {
    rapidjson::Value::ConstMemberIterator mitr;

    // TODO(croot): move the following prop strings to a const somewhere
    std::string fieldProp = "field";
    std::string valueProp = "value";
    std::string scaleProp = "scale";

    if ((mitr = obj.FindMember(fieldProp.c_str())) != obj.MemberEnd()) {
      // need to clear out the value path
      _valueJsonPath = rapidjson::Pointer();

      if (!_ctx->isJSONCacheUpToDate(_fieldJsonPath, mitr->value)) {
        RUNTIME_EX_ASSERT(dataPtr != nullptr,
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
                              obj,
                              "a data reference for the mark is not defined. Cannot access \"field\"."));
        RUNTIME_EX_ASSERT(mitr->value.IsString(),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(), obj, "\"field\" property for mark must be a string."));

        // TODO(croot): need to update references when a data
        // ptr has changed, but the scale reference hasn't
        // changed.
        initializeFromData(mitr->value.GetString(), dataPtr);
      }

      _fieldJsonPath = objPath.Append(fieldProp.c_str(), fieldProp.length());
    } else if ((mitr = obj.FindMember(valueProp.c_str())) != obj.MemberEnd()) {
      // need to clear out the field path
      _fieldJsonPath = rapidjson::Pointer();
      _dataPtr = nullptr;
      _vboAttrName = "";

      if (!_ctx->isJSONCacheUpToDate(_valueJsonPath, mitr->value)) {
        _initValueFromJSONObj(mitr->value);
      }
      _valueJsonPath = objPath.Append(valueProp.c_str(), valueProp.length());
    } else {
      // need some value source, either by "field" or by "value"
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(),
          obj,
          "invalid mark property object. Must contain a data reference via a \"field\" property "
          "or "
          "a \"value\" property."));
    }

    if ((mitr = obj.FindMember(scaleProp.c_str())) != obj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_scaleJsonPath, mitr->value)) {
        RUNTIME_EX_ASSERT(
            _useScale,
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "render property \"" + _name + "\" does not support scale references."));

        _initScaleFromJSONObj(mitr->value);
        _verifyScale();
      }

      _scaleJsonPath = objPath.Append(scaleProp.c_str(), scaleProp.length());
    } else {
      // TODO(croot): If a scale was used previously but now it's not, we need to
      // indicate that a shader rebuild is needed. Two possible approaches:
      // 1) have a local pointer that points back to the mark that encapsulates
      //    this render property and have a public function on the mark object
      //    to mark the shader as dirty.
      // 2) mark a local dirty flag and leave it up to the mark to traverse
      //    all its render properties looking for the dirty flag.

      // need to clear out the _scaleJsonPath
      _scaleJsonPath = rapidjson::Pointer();

      if (_scaleConfigPtr) {
        _prntMark->setShaderDirty();
        _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
      }

      _scaleConfigPtr = nullptr;
    }

    _initFromJSONObj(obj);

  } else {
    // need to clear out the object paths
    _fieldJsonPath = rapidjson::Pointer();
    _valueJsonPath = rapidjson::Pointer();
    _scaleJsonPath = rapidjson::Pointer();
    _dataPtr = nullptr;
    _vboAttrName = "";

    if (_scaleConfigPtr) {
      _prntMark->setShaderDirty();
      _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
    }
    _scaleConfigPtr = nullptr;

    _initValueFromJSONObj(obj);
  }
}

void BaseRenderProperty::initializeFromData(const std::string& columnName, const QueryDataTableVBOShPtr& dataPtr) {
  RUNTIME_EX_ASSERT(dataPtr != nullptr,
                    std::string(*this) + ": Cannot initialize mark property " + _name +
                        " from data. A valid data reference hasn't been initialized.");

  _dataPtr = dataPtr;
  _vboAttrName = columnName;

  _initVBOs(_dataPtr->getColumnDataVBOs(columnName));
  _vboInitType = VboInitType::FROM_DATAREF;

  _prntMark->setPropsDirty();

  _initTypeFromVbo();
}

int BaseRenderProperty::size(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end()) {
    return itr->second.vbo->numItems();
  }
  return 0;
}

bool BaseRenderProperty::hasVboPtr() {
  for (auto& itr : _perGpuData) {
    if (itr.second.vbo != nullptr) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasVboPtr(const GpuId& gpuId) {
  auto itr = _perGpuData.find(gpuId);

  return (itr != _perGpuData.end() && itr->second.vbo != nullptr);
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end()) {
    return itr->second.vbo;
  }

  return nullptr;
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr() const {
  auto itr = _perGpuData.begin();
  if (itr != _perGpuData.end()) {
    return itr->second.vbo;
  }

  return nullptr;
}

std::string BaseRenderProperty::getInGLSLType() const {
  RUNTIME_EX_ASSERT(_inType != nullptr,
                    std::string(*this) + " getInGLSLType(): input type for \"" + _name + "\" is uninitialized.");

  if (_scaleConfigPtr != nullptr) {
    std::string glslType = _scaleConfigPtr->getDomainTypeGL()->glslType();
    RUNTIME_EX_ASSERT(glslType == _inType->glslType(),
                      std::string(*this) + " getInGLSLType(): the domain type for scale \"" +
                          _scaleConfigPtr->getNameRef() + "\" does not match the type for mark property \"" + _name +
                          "\"");
    return glslType;
  }

  return _inType->glslType();
}

std::string BaseRenderProperty::getOutGLSLType() const {
  if (_scaleConfigPtr != nullptr) {
    return _scaleConfigPtr->getRangeTypeGL()->glslType();
  }

  RUNTIME_EX_ASSERT(
      _outType != nullptr,
      std::string(*this) + " getOutGLSLType(): output type for mark property \"" + _name + "\" is uninitialized.");

  return (_outType->glslType());
}

void BaseRenderProperty::addToVboAttrMap(const GpuId& gpuId,
                                         ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) const {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr->second.vbo != nullptr,
                    std::string(*this) +
                        " addToVboAttrMap(): A vertex buffer is not defined. Cannot add vbo attrs to "
                        "vbo->shader attr map.");

  ::Rendering::GL::Resources::GLVertexBufferShPtr glVbo = itr->second.vbo->getGLVertexBufferPtr();
  auto attrMapItr = attrMap.find(glVbo);
  if (attrMapItr == attrMap.end()) {
    attrMap.insert({glVbo, {{_vboAttrName, _name}}});
  } else {
    attrMapItr->second.push_back({_vboAttrName, _name});
  }
}

void BaseRenderProperty::initGpuResources(const QueryRendererContext* ctx,
                                          const std::unordered_set<GpuId> unusedGpus,
                                          bool initializing) {
  const QueryRendererContext::PerGpuDataMap& qrcPerGpuData = ctx->getGpuDataMap();
  for (auto& itr : qrcPerGpuData) {
    if (_perGpuData.find(itr.first) == _perGpuData.end()) {
      PerGpuData gpuData(itr.second);
      if (!initializing && _dataPtr) {
        gpuData.vbo = _dataPtr->getColumnDataVBO(itr.first, _vboAttrName);
      }
      _perGpuData.emplace(itr.first, std::move(gpuData));
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }
}

std::string BaseRenderProperty::_printInfo() const {
  return "(name: " + _name + ", vbo attr name: " + _vboAttrName + ") " + to_string(_ctx->getUserWidgetIds());
}

template <typename T, int numComponents>
RenderProperty<T, numComponents>::RenderProperty(BaseMark* prntMark,
                                                 const std::string& name,
                                                 const QueryRendererContextShPtr& ctx,
                                                 bool useScale,
                                                 bool flexibleType)
    : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType), _mult(), _offset() {
  _inType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
  _outType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
}

template <typename T, int numComponents>
RenderProperty<T, numComponents>::~RenderProperty() {
  if (_scaleConfigPtr) {
    _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
  }
}

template <>
RenderProperty<ColorRGBA, 1>::RenderProperty(BaseMark* prntMark,
                                             const std::string& name,
                                             const QueryRendererContextShPtr& ctx,
                                             bool useScale,
                                             bool flexibleType)
    : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType), _mult(), _offset(), _uniformVal() {
  _inType.reset(new ::Rendering::GL::TypeGL<float, 4>());
  _outType.reset(new ::Rendering::GL::TypeGL<float, 4>());
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initScaleFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(
      obj.IsString(),
      RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(), obj, "scale reference for mark property \"" + _name + "\" must be a string."));

  RUNTIME_EX_ASSERT(_ctx != nullptr && _scaleConfigPtr == nullptr,
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(),
                        obj,
                        "cannot initialize mark property \"" + _name +
                            "\" from a scale. The context is uninitialized or a scale is already being referenced."));

  // TODO(croot): We probably need a better way to deal with types. We've got an _inType that is either defined
  // by an incoming data reference or an explicit value (or set of values). The latter is easy. We already
  // have the type in T/numComponents of the template. But the data reference is trickier.

  ScaleShPtr scalePtr = _ctx->getScale(obj.GetString());
  RUNTIME_EX_ASSERT(scalePtr != nullptr,
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(),
                        obj,
                        "the scale \"" + std::string(obj.GetString()) + "\" does not exist in the json."));

  _updateScalePtr(scalePtr);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_updateScalePtr(const ScaleShPtr& scalePtr) {
  if (!_scaleConfigPtr) {
    _prntMark->setShaderDirty();
  } else {
    ScaleShPtr prevPtr = _scaleConfigPtr->getScalePtr();
    if (!prevPtr || scalePtr.get() != prevPtr.get()) {
      _prntMark->setShaderDirty();
      _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
    }
  }

  if (dynamic_cast<::Rendering::GL::TypeGL<unsigned int, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<unsigned int, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<::Rendering::GL::TypeGL<int, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<int, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<::Rendering::GL::TypeGL<float, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<float, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<::Rendering::GL::TypeGL<double, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
  } else {
    THROW_RUNTIME_EX(std::string(*this) + ": Scale domain with shader type \"" +
                     scalePtr->getDomainTypeGL()->glslType() + "\" and data with shader type \"" + _inType->glslType() +
                     "\" are not supported to work together");
  }

  // setup callbacks for scale updates
  _scaleRefSubscriptionCB = std::bind(
      &RenderProperty<T, numComponents>::_scaleRefUpdateCB, this, std::placeholders::_1, std::placeholders::_2);
  _ctx->subscribeToRefEvent(RefEventType::ALL, scalePtr, _scaleRefSubscriptionCB);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initFromJSONObj(const rapidjson::Value& obj) {
  // this is internally called at the appropriate time from
  // the base class's initialization function, so there's
  // no need to check that obj is valid since that should've
  // already been done.
  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot) - fill this in with mult/offset json path caches.
  // These values will ultimately act as uniforms, so modifications
  // to them should not force a full shader build/compile.
  // if ((mitr = obj.FindMember("mult")) != obj.MemberEnd()) {
  //   _mult = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
  // }

  // if ((mitr = obj.FindMember("offset")) != obj.MemberEnd()) {
  //   _offset = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
  // }
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::initializeValue(const T& val) {
  // TODO: this is a public function.. should I protect from already existing data?

  if (_vboInitType != VboInitType::FROM_VALUE) {
    _inType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
    _outType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());

    for (auto& itr : _perGpuData) {
      itr.second.vbo = nullptr;
    }

    _vboInitType = VboInitType::FROM_VALUE;
    _prntMark->setPropsDirty();
  } else {
    // TODO(croot): do we need to set props dirty on the parent?
  }

  _uniformVal = val;
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::bindUniformToRenderer(GLShader* activeShader,
                                                             const std::string& uniformAttrName) const {
  // TODO(croot): deal with numComponents here by using a vector instead?
  activeShader->setUniformAttribute<T>(uniformAttrName, _uniformVal);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initValueFromJSONObj(const rapidjson::Value& obj) {
  T val = RapidJSONUtils::getNumValFromJSONObj<T>(obj);

  initializeValue(val);
}

template <>
void RenderProperty<ColorRGBA, 1>::initializeValue(const ColorRGBA& val) {
  // TODO: combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions

  // TODO(croot): make thread safe

  if (_vboInitType != VboInitType::FROM_VALUE) {
    _inType.reset(new ::Rendering::GL::TypeGL<float, 4>());
    _outType.reset(new ::Rendering::GL::TypeGL<float, 4>());

    for (auto& itr : _perGpuData) {
      itr.second.vbo = nullptr;
    }

    _vboInitType = VboInitType::FROM_VALUE;
    _prntMark->setPropsDirty();
  } else {
    // TODO(croot): do we need to set props dirty on the parent?
  }

  _uniformVal = val;
}

template <>
void RenderProperty<ColorRGBA, 1>::bindUniformToRenderer(GLShader* activeShader,
                                                         const std::string& uniformAttrName) const {
  // TODO(croot): deal with numComponents here by using a vector instead?
  activeShader->setUniformAttribute<std::array<float, 4>>(uniformAttrName, _uniformVal.getColorArray());
}

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj) {
  // TODO: what about offsets / mults for colors?
}

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsString(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "value for color property \"" + _name + "\" must be a string."));

  ColorRGBA color(obj.GetString());

  initializeValue(color);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initTypeFromVbo() {
  auto itr = _perGpuData.begin();

  RUNTIME_EX_ASSERT((itr != _perGpuData.end() && itr->second.vbo != nullptr),
                    std::string(*this) +
                        ": Vertex buffer is uninitialized. Cannot initialize type for mark property \"" + _name +
                        "\".");

  TypeGLShPtr vboType = itr->second.vbo->getAttributeTypeGL(_vboAttrName);

  if (_flexibleType) {
    _inType = vboType;
    _outType = vboType;
  } else {
    // the render property is rigid, meaning it cannot accept
    // different types. So verify that the type of the attribute
    // in the vbo is appropriate.
    RUNTIME_EX_ASSERT((*_outType) == (*vboType),
                      std::string(*this) +
                          ": The vertex buffer type does not match the output type for mark property \"" + _name +
                          "\".");
  }
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_verifyScale() {
}

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale() {
  RUNTIME_EX_ASSERT(_scaleConfigPtr != nullptr,
                    std::string(*this) + ": Cannot verify scale for mark property \"" + _name +
                        "\". Scale reference is uninitialized.");

  TypeGLShPtr vboType = _scaleConfigPtr->getRangeTypeGL();

  // colors need to be a specific type
  RUNTIME_EX_ASSERT(ColorRGBA::isValidTypeGL(vboType),
                    std::string(*this) + ": Vertex buffer to use for mark property \"" + _name +
                        "\" is not an appropriate type for colors.");
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_scaleRefUpdateCB(RefEventType refEventType, const ScaleShPtr& scalePtr) {
  switch (refEventType) {
    case RefEventType::UPDATE:
      _scaleConfigPtr->updateScaleRef(scalePtr);

      if (scalePtr->hasClampChanged() || scalePtr->hasDomainChangedInSize() || scalePtr->hasRangeChangedInSize()) {
        _prntMark->setShaderDirty();
      }
      break;
    case RefEventType::REPLACE:
      _updateScalePtr(scalePtr);
      break;
    case RefEventType::REMOVE:
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " + std::to_string(static_cast<int>(refEventType)) +
                       " isn't currently supported for scale reference updates.");
      break;
  }
}

template <typename T, int numComponents>
RenderProperty<T, numComponents>::operator std::string() const {
  return "RenderProperty<" + std::string(typeid(T).name()) + ", " + std::to_string(numComponents) + "> " + _printInfo();
}

BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx)
    : _type(geomType),
      key(this, "key", ctx, false, false),
      _invalidKey(ctx->getInvalidKey()),
      _dataPtr(nullptr),
      _perGpuData(),
      _ctx(ctx),
      _shaderDirty(true),
      _propsDirty(true),
      _vboProps(),
      _uniformProps() {
  _initGpuResources(_ctx.get());
}

BaseMark::BaseMark(GeomType geomType,
                   const QueryRendererContextShPtr& ctx,
                   const rapidjson::Value& obj,
                   const rapidjson::Pointer& objPath)
    : BaseMark(geomType, ctx) {
  _initFromJSONObj(obj, objPath);
}

BaseMark::~BaseMark() {
}

void BaseMark::_initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  RUNTIME_EX_ASSERT(
      obj.IsObject(),
      RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "definition for marks must be an object."));

  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot): move the "from" to a const somewhere
  std::string fromProp = "from";
  if ((mitr = obj.FindMember(fromProp.c_str())) != obj.MemberEnd()) {
    const rapidjson::Value& fromObj = mitr->value;

    if (!_ctx->isJSONCacheUpToDate(_dataPtrJsonPath, fromObj)) {
      RUNTIME_EX_ASSERT(fromObj.IsObject(),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(), fromObj, "mark data reference must be an object."));

      RUNTIME_EX_ASSERT(
          (mitr = fromObj.FindMember("data")) != fromObj.MemberEnd() && mitr->value.IsString(),
          RapidJSONUtils::getJsonParseErrorStr(
              _ctx->getUserWidgetIds(), fromObj, "mark data reference must contain a \"data\" string property."));

      _dataPtr = _ctx->getDataTable(mitr->value.GetString());
    }

    _dataPtrJsonPath = objPath.Append(fromProp.c_str(), fromProp.length());
  } else {
    // TODO(croot): what about references???
    _dataPtr = nullptr;
  }
}

void BaseMark::_buildVertexArrayObjectFromProperties() {
  if (!_propsDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  // TODO(croot): make thread safe?
  // GLRenderer* prevRenderer = GLRenderer::getCurrentThreadRenderer();
  // Window* prevWindow = GLRenderer::getCurrentThreadWindow();
  GLRenderer* currRenderer = nullptr;

  QueryRenderManager::PerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    qrmGpuData = itr.second.getQRMGpuData();
    CHECK(qrmGpuData);

    qrmGpuData->makeActiveOnCurrentThread();

    currRenderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());

    CHECK(currRenderer != nullptr);
    CHECK(itr.second.shaderPtr != nullptr);

    currRenderer->bindShader(itr.second.shaderPtr);

    // build property map for how vertex buffer attributes will
    // be bound to shader attributes
    VboAttrToShaderAttrMap attrMap;
    _addPropertiesToAttrMap(itr.first, attrMap);

    GLResourceManagerShPtr rsrcMgr = currRenderer->getResourceManager();

    itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap);
  }

  // if (currRenderer && prevRenderer != currRenderer) {
  //   if (prevRenderer) {
  //     prevRenderer->makeActiveOnCurrentThread(prevWindow);
  //   } else {
  //     currRenderer->makeInactive();
  //   }
  // }

  _propsDirty = false;
}

void BaseMark::_initGpuResources(const QueryRendererContext* ctx,
                                 const std::unordered_set<GpuId> unusedGpus,
                                 bool initializing) {
  const QueryRendererContext::PerGpuDataMap& qrcPerGpuData = ctx->getGpuDataMap();

  ::Rendering::GL::GLRenderer* renderer;
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr;
  int numGpus = _perGpuData.size();
  bool update = (numGpus > 0 && _perGpuData.begin()->second.shaderPtr);
  bool createdNewGpuRsrc = false;
  for (auto& itr : qrcPerGpuData) {
    auto perGpuItr = _perGpuData.find(itr.first);
    if (perGpuItr == _perGpuData.end()) {
      PerGpuData gpuData(itr.second);
      if (update) {
        auto beginItr = _perGpuData.begin();
        CHECK(beginItr != _perGpuData.end() && beginItr->second.shaderPtr);

        beginItr->second.makeActiveOnCurrentThread();
        std::string vertSrc = beginItr->second.shaderPtr->getVertexSource();
        std::string fragSrc = beginItr->second.shaderPtr->getFragmentSource();

        itr.second.makeActiveOnCurrentThread();
        renderer = dynamic_cast<::Rendering::GL::GLRenderer*>(itr.second.getQRMGpuData()->rendererPtr.get());
        CHECK(renderer);

        rsrcMgr = renderer->getResourceManager();

        // TODO(croot): make resource copy constructors which appropriately
        // deal with different contexts, including contexts on different gpus
        gpuData.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);

        // NOTE: we need to create the VAO after the properties have
        // been updated.
        createdNewGpuRsrc = true;
      }
      _perGpuData.emplace(itr.first, std::move(gpuData));
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }

  if (numGpus && _perGpuData.size() == 0) {
    // TODO(croot): make a makeAllDirty() function
    setShaderDirty();
    setPropsDirty();
  }

  key.initGpuResources(ctx, unusedGpus, initializing);

  if (!initializing) {
    _updateRenderPropertyGpuResources(ctx, unusedGpus);
  }

  if (createdNewGpuRsrc) {
    for (auto& itr : _perGpuData) {
      if (!itr.second.vaoPtr) {
        itr.second.makeActiveOnCurrentThread();
        renderer = dynamic_cast<::Rendering::GL::GLRenderer*>(itr.second.getQRMGpuData()->rendererPtr.get());
        CHECK(renderer);

        rsrcMgr = renderer->getResourceManager();

        CHECK(itr.second.shaderPtr);
        renderer->bindShader(itr.second.shaderPtr);

        ::Rendering::GL::Resources::VboAttrToShaderAttrMap attrMap;
        _addPropertiesToAttrMap(itr.first, attrMap);

        itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap);
      }
    }
  }
}

PointMark::PointMark(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx)
    : BaseMark(POINTS, ctx, obj, objPath),
      x(this, "x", ctx),
      y(this, "y", ctx),
      z(this, "z", ctx),
      size(this, "size", ctx),

      // TODO(croot): let's log a warning and continue onwards if
      // hit testing is asked for, but the input sql data doesn't
      // have an id.
      id(this, "id", ctx, false),
      fillColor(this, "fillColor", ctx) {
  _initPropertiesFromJSONObj(obj, objPath);
  _updateShader();
}

PointMark::~PointMark() {
}

void PointMark::_initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  // TODO(croot): move "properties" to a const somewhere
  std::string propertiesProp = "properties";

  rapidjson::Value::ConstMemberIterator mitr;

  RUNTIME_EX_ASSERT((mitr = obj.FindMember(propertiesProp.c_str())) != obj.MemberEnd(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "mark objects must have a \"properties\" property."));

  const rapidjson::Value& propObj = mitr->value;

  if (!_ctx->isJSONCacheUpToDate(_propertiesJsonPath, propObj)) {
    std::vector<BaseRenderProperty*> usedProps{&x, &y, &size, &fillColor};  // TODO(croot) add z

    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());

    RUNTIME_EX_ASSERT(
        propObj.IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "The \"properties\" property of marks must be a json object."));

    // TODO(croot): move "x" to a const somewhere
    std::string xProp = "x";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(xProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + xProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_xJsonPath, mitr->value)) {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + xProp + "\" mark property must be a scale/data reference or a number."));
      x.initializeFromJSONObj(mitr->value, _xJsonPath, _dataPtr);
    } else {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
    }

    // TODO(croot): move "y" to a const somewhere
    std::string yProp = "y";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(yProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + yProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_yJsonPath, mitr->value)) {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + yProp + "\" mark property must be a scale/data reference or a number."));
      y.initializeFromJSONObj(mitr->value, _yJsonPath, _dataPtr);
    } else {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
    }

    // TODO(croot): move "z" to a const somewhere
    std::string zProp = "z";
    if ((mitr = propObj.FindMember(zProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_zJsonPath, mitr->value)) {
        _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
        RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
                              mitr->value,
                              "\"" + zProp + "\" mark property must be a scale/data reference or a number."));
        z.initializeFromJSONObj(mitr->value, _zJsonPath, _dataPtr);
      } else {
        _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
      }

      usedProps.push_back(&z);
    } else {
      // empty the json path for z
      _zJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): move "size" to a const somewhere
    std::string sizeProp = "size";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(sizeProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + sizeProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_sizeJsonPath, mitr->value)) {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + sizeProp + "\" mark property must be a scale/data reference or a number."));
      size.initializeFromJSONObj(mitr->value, _sizeJsonPath, _dataPtr);
    } else {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
    }

    // TODO(croot): move "fillColor" to a const somewhere
    std::string fillColorProp = "fillColor";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(fillColorProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + fillColorProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_fillColorJsonPath, mitr->value)) {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsString()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + fillColorProp + "\" mark property must be a scale/data reference or a string."));
      fillColor.initializeFromJSONObj(mitr->value, _fillColorJsonPath, _dataPtr);
    } else {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
    }

    if (_ctx->doHitTest()) {
      // TODO(croot): move "id" to a const somewhere
      std::string idProp = "id";

      if ((mitr = propObj.FindMember(idProp.c_str())) != propObj.MemberEnd()) {
        if (!_ctx->isJSONCacheUpToDate(_idJsonPath, mitr->value)) {
          _idJsonPath = _propertiesJsonPath.Append(idProp.c_str(), idProp.length());
          RUNTIME_EX_ASSERT(mitr->value.IsObject(),
                            RapidJSONUtils::getJsonParseErrorStr(
                                _ctx->getUserWidgetIds(),
                                mitr->value,
                                "\"id\" is a special mark property that must be defined by a data reference."));
          id.initializeFromJSONObj(mitr->value, _idJsonPath, _dataPtr);
        } else {
          // update the json path, if it's been changed
          _idJsonPath = _propertiesJsonPath.Append(idProp.c_str(), idProp.length());
        }
      } else {
        // clear out the json path
        _idJsonPath = rapidjson::Pointer();
        if (_dataPtr != nullptr) {
          id.initializeFromData(DataTable::defaultIdColumnName, _dataPtr);
        } else {
          id.initializeValue(0);  // reaching here "should" guarantee that there's only
                                  // 1 row of data
        }
      }

      usedProps.push_back(&id);
    } else {
      // clear out id path
      _idJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): put the following in the BaseMark class somewhere so that all
    // future marks (lines, polys) will pick up this code.
    QueryVertexBufferShPtr vboPtr;
    static const QueryVertexBuffer::VboType resultVBO = QueryVertexBuffer::VboType::QUERY_RESULT_VBO;
    if ((((vboPtr = x.getVboPtr()) && vboPtr->getType() == resultVBO) ||
         ((vboPtr = y.getVboPtr()) && vboPtr->getType() == resultVBO) ||
         ((vboPtr = size.getVboPtr()) && vboPtr->getType() == resultVBO) ||
         // TODO(croot): what if we have multiple sqls? How do we handle the "key" value then?
         ((vboPtr = fillColor.getVboPtr()) && vboPtr->getType() == resultVBO)) &&
        vboPtr->hasAttribute(key.getName())) {
      key.initializeFromData(key.getName(), _dataPtr);
      usedProps.push_back(&key);
    }

    // Now update which props are vbo-defined, and which will be uniforms
    _vboProps.clear();
    _uniformProps.clear();

    for (const auto& prop : usedProps) {
      if (prop->hasVboPtr()) {
        _vboProps.push_back(prop);
      } else {
        _uniformProps.push_back(prop);
      }
    }
  } else {
    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());
  }
}

void PointMark::_updateShader() {
  // TODO(croot): need to determine a build-appropriate way to access
  // shaders. The best way probably is to create a cmake build
  // script that converts all shaders into their own header
  // files with static strings of the shader's source to access.

  if (!_shaderDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  std::string vertSrc(PointTemplate_Vert::source);

  std::vector<BaseRenderProperty*> props = {&x, &y, &size, &fillColor};  // TODO: add z & fillColor

  bool useKey = key.hasVboPtr();
  boost::replace_first(vertSrc, "<useKey>", std::to_string(useKey));

  // update all the types first
  for (auto& prop : props) {
    setRenderPropertyTypeInShaderSrc(*prop, vertSrc);
  }

  // now set props as uniform or vertex attrs
  for (auto& prop : _vboProps) {
    setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, false);
  }

  for (auto& prop : _uniformProps) {
    setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, true);
  }

  if (_ctx->doHitTest()) {
    props.push_back(&id);
  } else {
    // define the id as uniform to get the shader to compile
    setRenderPropertyAttrTypeInShaderSrc(id, vertSrc, true);
  }

  // now insert any additional functionality
  // std::unordered_map<std::string, BaseScale*> visitedScales;
  std::unordered_map<std::string, BaseScale*>::iterator itr;

  std::string funcName;
  std::string propFuncName;
  str_itr_range funcRange;

  for (auto prop : props) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      // NOTE: Because the domains of scales can be coerced into
      // the render property's type, we need to provide a new
      // set of GLSL code for each scale reference, even tho
      // it is possible to reference the same scale multiple times.

      // TODO(croot): there are ways we can reduce the amount of
      // shader code here. Domains of certain scales can be coerced,
      // but not all scales, so we can find some optimizations there.
      // Also, ranges can not be coerced, so optimizations can be
      // do there as well, but it is likely rare that the same
      // scale be referenced many times at this point (11/9/15), so
      // it's probably not worth the effort to optimize at this point.
      propFuncName = prop->getGLSLFunc();

      funcRange = getGLSLFunctionBounds(vertSrc, propFuncName);

      RUNTIME_EX_ASSERT(!funcRange.empty(),
                        std::string(*this) + ": Cannot find a properly defined \"" + propFuncName +
                            "\" function in the vertex shader.");

      std::string scaleCode = scalePtr->getGLSLCode("_" + prop->getName());

      boost::replace_range(vertSrc, funcRange, scaleCode);

      funcName = scalePtr->getScaleGLSLFuncName("_" + prop->getName());

      boost::replace_all(
          vertSrc, prop->getGLSLFunc() + "(" + prop->getName() + ")", funcName + "(" + prop->getName() + ")");
    }
  }

  // std::string fragSrc = getShaderCodeFromFile(pointFragmentShaderFilename);
  std::string fragSrc(PointTemplate_Frag::source);

  // static int CROOTcnt = 0;
  // CROOTcnt++;
  // if (CROOTcnt == 1) {
  //   std::ofstream shadersrcstream;
  //   shadersrcstream.open("shadersource.vert");
  //   shadersrcstream << vertSrc;
  //   shadersrcstream.close();
  // }

  // now build the shader object
  // TODO(croot): How would we share shaders across different
  // query renderers?

  // TODO(croot): Make thread safe?

  GLRenderer* prevRenderer = GLRenderer::getCurrentThreadRenderer();
  GLRenderer* currRenderer = nullptr;
  QueryRenderManager::PerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    qrmGpuData = itr.second.getQRMGpuData();
    qrmGpuData->makeActiveOnCurrentThread();
    currRenderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());

    GLResourceManagerShPtr rsrcMgr = currRenderer->getResourceManager();
    itr.second.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);

    // TODO(croot): should I make make the current thread
    // have an inactive renderer?
  }

  if (currRenderer && prevRenderer != currRenderer) {
    if (prevRenderer) {
      prevRenderer->makeActiveOnCurrentThread();
    } else {
      currRenderer->makeInactive();
    }
  }

  _shaderDirty = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void PointMark::_addPropertiesToAttrMap(const GpuId& gpuId, VboAttrToShaderAttrMap& attrMap) {
  int cnt = 0;
  int vboSize = 0;
  int itrSize = 0;
  for (auto& itr : _vboProps) {
    cnt++;
    itrSize = itr->size(gpuId);
    if (cnt == 1) {
      vboSize = itrSize;
    } else {
      RUNTIME_EX_ASSERT(itrSize == vboSize,
                        std::string(*this) +
                            ": Invalid point mark. The sizes of the vertex buffer attributes do not match for gpuId " +
                            std::to_string(gpuId) + ". " + std::to_string(vboSize) + "!=" + std::to_string(itrSize));
    }

    itr->addToVboAttrMap(gpuId, attrMap);
    // itr->bindToRenderer(activeShader);
  }
}

void PointMark::_bindUniformProperties(GLShader* activeShader) {
  // TODO(croot): create a static invalidKeyAttrName string on the class
  static const std::string invalidKeyAttrName = "invalidKey";
  if (key.hasVboPtr()) {
    if (activeShader->hasUniformAttribute(invalidKeyAttrName)) {
      GLint type = activeShader->getUniformAttributeGLType(invalidKeyAttrName);
      if (type == GL_INT) {
        activeShader->setUniformAttribute<int>(invalidKeyAttrName, static_cast<int>(_invalidKey));
      }  // else if (GLEW_NV_vertex_attrib_integer_64bit && type == GL_INT64_NV) {
         // TODO(croot) - do we need to do the glew extension check above or
         // would there be an error at shader compilation if the extension
         // didn't exist?

      // TODO(croot) fill this out
      // }
    }
  }

  for (auto prop : _vboProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, "_" + prop->getName());
    }
  }

  for (auto prop : _uniformProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, "_" + prop->getName());
    }

    prop->bindUniformToRenderer(activeShader, prop->getName());
  }
}

void PointMark::draw(GLRenderer* renderer, const GpuId& gpuId) {
  // NOTE: shader should have been updated before calling this
  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  QueryRenderManager::PerGpuDataShPtr qrmGpuData = itr->second.getQRMGpuData();
  CHECK(qrmGpuData);

  ::Rendering::Renderer* rndr = qrmGpuData->rendererPtr.get();
  CHECK(itr->second.shaderPtr && rndr == renderer);

  // now bind the shader
  renderer->bindShader(itr->second.shaderPtr);
  renderer->bindVertexArray(itr->second.vaoPtr);

  _bindUniformProperties(itr->second.shaderPtr.get());

  // TODO: render state stack -- push/pop
  renderer->enable(GL_PROGRAM_POINT_SIZE);

  // now draw points
  // TODO: What about the possibility of index buffers?
  // Probably don't have to worry about them here since
  // we're specifically looking at a point config that
  // is deemed not to have any index buffers, but we
  // need to have a way to generically draw bound buffers
  // which would be best with a renderer class
  // MAPD_CHECK_GL_ERROR(glDrawArrays(GL_POINTS, 0, x.size()));
  renderer->drawVertexBuffers(GL_POINTS, 0, itr->second.vaoPtr->numItems());

  // unset state
  renderer->disable(GL_PROGRAM_POINT_SIZE);
}

bool PointMark::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = false;
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    BaseMark::_initFromJSONObj(obj, objPath);
    _initPropertiesFromJSONObj(obj, objPath);
    rtn = true;
  } else if (_jsonPath != objPath) {
    // TODO(croot) - Bug! What if the cache is up-to-date, but the path has changed -- we need to update
    // the paths for this and all sub-objects that manage paths.
    // NOTE: We should never get in here since marks
    // are stored as an array but if we ever change the storage container
    // in the future (i.e. an unordered_map or the like), we'd reach this
    THROW_RUNTIME_EX(std::string(*this) + ": The cache for mark \"" + RapidJSONUtils::getPointerPath(objPath) +
                     "\" is up-to-date, but the path in the JSON has changed from " +
                     RapidJSONUtils::getPointerPath(_jsonPath) + " to " + RapidJSONUtils::getPointerPath(objPath) +
                     ", so the path caches need updating. This "
                     "has yet to be implemented.");
  }

  // TODO(croot): if the obj hasn't changed, but the path has, we need
  // to trickle that path change to all subobjects who cache their
  // json data. How should we handle this?
  _jsonPath = objPath;

  return rtn;
}

PointMark::operator std::string() const {
  return "PointMark " + to_string(_ctx->getUserWidgetIds());
}

BaseMark::GeomType getMarkTypeFromJSONObj(const rapidjson::Value& obj) {
  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("type")) != obj.MemberEnd() && itr->value.IsString(),
                    RapidJSONUtils::getJsonParseErrorStr(obj, "a mark object must have a \"type\" string property."));

  std::string strGeomType(itr->value.GetString());

  if (strGeomType == "points") {
    return BaseMark::GeomType::POINTS;
  } else {
    THROW_RUNTIME_EX(
        RapidJSONUtils::getJsonParseErrorStr(obj, "a mark of type \"" + strGeomType + "\" is unsupported."));
  }
}

GeomConfigShPtr createMark(const rapidjson::Value& obj,
                           const rapidjson::Pointer& objPath,
                           const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), RapidJSONUtils::getJsonParseErrorStr(obj, "marks must be objects."));

  switch (getMarkTypeFromJSONObj(obj)) {
    case BaseMark::GeomType::POINTS:
      return GeomConfigShPtr(new PointMark(obj, objPath, ctx));
      break;
  }

  return GeomConfigShPtr(nullptr);
}

}  // namespace QueryRenderer
