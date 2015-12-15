#include "QueryRendererObjects.h"
#include "ShaderUtils.h"
#include "shaders/pointTemplate_vert.h"
#include "shaders/pointTemplate_frag.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"
#include "../QueryEngine/Execute.h"

#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/find.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/regex_find_format.hpp>

#include <iostream>
#include <sstream>
#include <utility>

// CROOT - remote the following includes -- only used for debugging
// #include <fstream>

using namespace MapD_Renderer;

typedef std::string::iterator str_itr;
typedef boost::iterator_range<str_itr> str_itr_range;

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

DataType getDataTypeFromJSONObj(const rapidjson::Value& obj, bool supportString = false) {
  DataType rtn(DataType::INT);
  rapidjson::Type type = obj.GetType();

  switch (type) {
    case rapidjson::kNumberType:
      if (obj.IsInt()) {
        rtn = DataType::INT;
      } else if (obj.IsUint()) {
        rtn = DataType::UINT;
      } else if (obj.IsDouble()) {
        rtn = DataType::DOUBLE;

        // double val = obj.GetDouble();
        // TODO(croot): how do we handle floats?
        // if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
        //   rtn = DataType::FLOAT;
        // } else {
        //   rtn = DataType::DOUBLE;
        // }
      } else {
        THROW_RUNTIME_EX("getDataTypeFromJSONObj(): RapidJSON number type is not supported.");
      }
      break;
    case rapidjson::kStringType: {
      std::string val = obj.GetString();
      if (supportString) {
        rtn = DataType::STRING;
      } else if (ColorRGBA::isColorString(val)) {
        rtn = DataType::COLOR;
      } else {
        THROW_RUNTIME_EX("getDataTypeFromJSONObj(): non-color strings are not a supported type.");
      }
    } break;
    default:
      THROW_RUNTIME_EX("getDataTypeFromJSONObj(): type from JSON is unsupported.");
  }

  return rtn;
}

DataType getDataTypeFromDataRefJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - data reference is not a JSON object.");

  rapidjson::Value::ConstMemberIterator mitr;
  RUNTIME_EX_ASSERT((mitr = obj.FindMember("data")) != obj.MemberEnd() && mitr->value.IsString(),
                    "JSON parse error - data reference object doesn't contain a \"data\" string property.");
  std::string dataTableName = mitr->value.GetString();
  DataVBOShPtr tablePtr = ctx->getDataTable(dataTableName);

  RUNTIME_EX_ASSERT(tablePtr != nullptr, "JSON parse error - data table \"" + dataTableName + "\" doesn't exist.");

  RUNTIME_EX_ASSERT((mitr = obj.FindMember("field")) != obj.MemberEnd() && mitr->value.IsString(),
                    "JSON parse error - data reference object must contain a \"field\" string property.");
  return tablePtr->getColumnType(mitr->value.GetString());
}

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    ::MapD_Renderer::LinearScaleTemplate_vert::source,  // LINEAR
    ::MapD_Renderer::OrdinalScaleTemplate_vert::source  // ORDINAL
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx,
                     DataType domainDataType,
                     DataType rangeDataType,
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
                     DataType domainDataType,
                     DataType rangeDataType,
                     const std::string& name,
                     BaseScale::ScaleType type)
    : BaseScale(ctx, domainDataType, rangeDataType, name, type) {
  _initFromJSONObj(obj, objPath);
}

BaseScale::~BaseScale() {
}

void BaseScale::_initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - scale items must be objects.");

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

      if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(itr->value.IsBool(),
                          "JSON parse error - the \"clamp\" property for linear scales must be a boolean.");

        _useClamp = itr->value.GetBool();
      } else {
        // TODO(croot): set a const default for _useClamp somewhere
        _useClamp = false;
      }
    }
  }
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
      THROW_RUNTIME_EX("BaseScale::getScaleGLSLFuncName(): scale type is not supported.");
  }

  return "get" + scaleName + "Scale_" + _name + extraSuffix;
}

template <typename T>
void ScaleDomainRangeData<T>::initializeFromJSONObj(const rapidjson::Value& obj,
                                                    const rapidjson::Pointer& objPath,
                                                    const QueryRendererContextShPtr& ctx,
                                                    BaseScale::ScaleType type) {
  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  bool isObject = false, isString = false;
  DataVBOShPtr tablePtr;
  DataColumnShPtr columnPtr;

  mitr = obj.FindMember(_name.c_str());
  RUNTIME_EX_ASSERT(mitr != obj.MemberEnd(),
                    "JSON parse error - scale objects must have a \"" + _name + "\" property.");

  if (!ctx->isJSONCacheUpToDate(_jsonPath, mitr->value)) {
    if (_useString) {
      RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
                         (mitr->value.IsArray() && mitr->value.Size())),
                        "JSON parse error - scale " + _name + " is invalid.");
    } else {
      RUNTIME_EX_ASSERT(((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())),
                        "JSON parse error - scale " + _name + " is invalid.");
    }

    const rapidjson::Value& jsonObj = mitr->value;

    if (isObject) {
      RUNTIME_EX_ASSERT((mitr = jsonObj.FindMember("data")) != jsonObj.MemberEnd() && mitr->value.IsString(),
                        "JSON parse error - scale data reference must have a \"data\" string property.");
      tablePtr = ctx->getDataTable(mitr->value.GetString());

      RUNTIME_EX_ASSERT((mitr = jsonObj.FindMember("field")) != jsonObj.MemberEnd() && mitr->value.IsString(),
                        "JSON parse error - scale data reference must have a \"field\" string property.");

      // Only supports hand-written data right now.
      // TODO(croot): Support query result vbo -- this is somewhat
      // tricky, because in order to do so we'd have to use compute
      // shaders/cuda to do min/max/other math stuff, and uniform buffers or
      // storage buffers to send the values as uniforms
      RUNTIME_EX_ASSERT(tablePtr->getType() == BaseDataTableVBO::DataTableType::EMBEDDED,
                        "JSON parse error - scale data references only support JSON-embedded data tables currently.");

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
      DataType itemType;

      bool supportStr = (typeid(T) == typeid(std::string));
      size_t idx = 0;
      for (vitr = jsonObj.Begin(); vitr != jsonObj.End(); ++vitr, ++idx) {
        // only strings are currently allowed in the domains
        itemType = getDataTypeFromJSONObj(*vitr, supportStr);

        RUNTIME_EX_ASSERT(itemType == dataType,
                          "JSON parse error - scale " + _name + " item " + std::to_string(vitr - jsonObj.Begin()) +
                              " has an invalid type.");

        // _pushItem(*vitr);
        _setItem(idx, *vitr);
      }
    }
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
    _cachedTypeGL.reset(new TypeGL<T>());
  }
  return _cachedTypeGL;
}

template <>
inline const TypeGLShPtr& ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
  // TODO(croot): combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions
  if (!_cachedTypeGL) {
    _cachedTypeGL.reset(new TypeGL<float, 4>());
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
const DataType ScaleDomainRangeData<unsigned int>::dataType = DataType::UINT;

template <>
unsigned int MapD_Renderer::ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetUint();
}

template <>
const DataType ScaleDomainRangeData<int>::dataType = DataType::INT;

template <>
int MapD_Renderer::ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetInt();
}

template <>
const DataType ScaleDomainRangeData<float>::dataType = DataType::FLOAT;

template <>
float MapD_Renderer::ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return static_cast<float>(obj.GetDouble());
}

template <>
const DataType ScaleDomainRangeData<double>::dataType = DataType::DOUBLE;

template <>
double MapD_Renderer::ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetDouble();
}

template <>
const DataType ScaleDomainRangeData<ColorRGBA>::dataType = DataType::COLOR;

template <>
ColorRGBA MapD_Renderer::ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorRGBA(obj.GetString());
}

template <>
const DataType ScaleDomainRangeData<std::string>::dataType = DataType::STRING;

template <>
std::string MapD_Renderer::ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
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
    THROW_RUNTIME_EX("JSON parse rror - string value for domain/range is invalid.");
  }
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
  THROW_RUNTIME_EX("JSON parse error - string value for color domain/range is invalid.");
}

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
  THROW_RUNTIME_EX(
      "JSON parse error - string value for string domain/range is invalid. It must be an array of strings.");
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
    THROW_RUNTIME_EX("JSON parse error - getting the extrema of colors in the domain/range is unsupported.");
  }
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
      DataType itemType = getDataTypeFromJSONObj(mitr->value);

      RUNTIME_EX_ASSERT(itemType == _rangePtr.dataType,
                        "JSON parse error - scale \"" + _name + "\" default is not the same type as its range.");

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
void Scale<DomainType, RangeType>::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  BaseScale::_initFromJSONObj(obj, objPath);

  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    _domainPtr.initializeFromJSONObj(obj, objPath, _ctx, _type);
    _rangePtr.initializeFromJSONObj(obj, objPath, _ctx, _type);

    if (_type == BaseScale::ScaleType::ORDINAL) {
      _setDefaultFromJSONObj(obj, objPath);
    }
  } else if (_jsonPath != objPath) {
    _domainPtr.updateJSONPath(objPath);
    _rangePtr.updateJSONPath(objPath);
    _updateDefaultJSONPath(objPath);
  }

  _jsonPath = objPath;
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
                    "Scale::getGLSLCode(): domain/range of scale \"" + _name + "\" has no value.");

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
void Scale<DomainType, RangeType>::bindUniformsToRenderer(Shader* activeShader,
                                                          const std::string& extraSuffix,
                                                          bool ignoreDomain,
                                                          bool ignoreRange) {
  if (!ignoreDomain) {
    activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorData());
  }

  if (!ignoreRange) {
    activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorData());

    if (_type == ScaleType::ORDINAL) {
      activeShader->setUniformAttribute(getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
    }
  }
}

std::string MapD_Renderer::getScaleNameFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - scale items must be JSON objects.");

  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
                    "JSON parse error - scale objects must contain a \"name\" string property.");

  return itr->value.GetString();
}

BaseScale::ScaleType MapD_Renderer::getScaleTypeFromJSONObj(const rapidjson::Value& obj) {
  // TODO(croot): expose default as a static attr.
  BaseScale::ScaleType rtn = BaseScale::ScaleType::LINEAR;

  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - scale items must be JSON objects.");

  rapidjson::Value::ConstMemberIterator itr;
  if ((itr = obj.FindMember("type")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(), "JSON parse error - \"type\" property in scale objects must be a string.");
    std::string strScaleType(itr->value.GetString());

    if (strScaleType == "linear") {
      rtn = BaseScale::ScaleType::LINEAR;
    } else if (strScaleType == "ordinal") {
      rtn = BaseScale::ScaleType::ORDINAL;
    } else {
      THROW_RUNTIME_EX("JSON parse error - scale type \"" + strScaleType + "\" is not a supported type.");
    }
  }

  return rtn;
}

DataType MapD_Renderer::getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj,
                                                          const QueryRendererContextShPtr& ctx) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;

  // TODO(croot): expose "domain" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (itr->value.IsArray() && itr->value.Size())),
                    "JSON parse error - \"domain\" property for scales must exist and must be an object or an array.");

  DataType domainType;

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

DataType MapD_Renderer::getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj,
                                                         const QueryRendererContextShPtr& ctx) {
  rapidjson::Value::ConstMemberIterator itr;
  bool isObject = false;
  bool isString;

  // TODO(croot): expose "range" as a const somewhere.
  RUNTIME_EX_ASSERT((itr = obj.FindMember("range")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (isString = itr->value.IsString()) ||
                         (itr->value.IsArray() && itr->value.Size())),
                    "JSON parse error - \"range\" property for scales must exist and must be an object or a string.");

  DataType rangeType;
  if (isObject) {
    rangeType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else if (isString) {
    std::string strVal = itr->value.GetString();
    RUNTIME_EX_ASSERT(
        strVal == "width" || strVal == "height",
        "JSON parse error - invalid \"range\" string property for scales. Only string values supported are "
        "\"width\" and \"height\"");

    // TODO(croot): should we actually use width/height values in the scale?
    // The easy way is just to use the -1 to 1 NDC range, but what
    // if someone uses "height" as the range, but uses that scale
    // for an x coordinate or some other property? Should use the
    // true 0-height in that case.
    rangeType = DataType::FLOAT;
  } else {
    rangeType = getDataTypeFromJSONObj(itr->value[0]);
  }

  return rangeType;
}

ScaleShPtr MapD_Renderer::createScale(const rapidjson::Value& obj,
                                      const rapidjson::Pointer& objPath,
                                      const QueryRendererContextShPtr& ctx,
                                      const std::string& name,
                                      BaseScale::ScaleType type) {
  std::string scaleName(name);
  if (!scaleName.length()) {
    scaleName = getScaleNameFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleName.length() > 0, "JSON parse error - Scales must have a \"name\" property");

  BaseScale::ScaleType scaleType(type);
  if (scaleType == BaseScale::ScaleType::UNDEFINED) {
    scaleType = getScaleTypeFromJSONObj(obj);
  }

  RUNTIME_EX_ASSERT(scaleType != BaseScale::ScaleType::UNDEFINED,
                    "JSON parse error - Scale type for \"" + scaleName + "\" is undefined.");

  rapidjson::Value::ConstMemberIterator itr;

  DataType domainType = getScaleDomainDataTypeFromJSONObj(obj, ctx);
  DataType rangeType = getScaleRangeDataTypeFromJSONObj(obj, ctx);

  switch (domainType) {
    case DataType::UINT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<unsigned int, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<unsigned int, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<unsigned int, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<unsigned int, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<unsigned int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    case DataType::INT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<int, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<int, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<int, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<int, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<int, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    case DataType::FLOAT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<float, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<float, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<float, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<float, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<float, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    case DataType::DOUBLE:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<double, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<double, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<double, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<double, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<double, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    case DataType::COLOR:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<ColorRGBA, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<ColorRGBA, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<ColorRGBA, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<ColorRGBA, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<ColorRGBA, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    case DataType::STRING:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<std::string, unsigned int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::INT:
          return ScaleShPtr(new Scale<std::string, int>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<std::string, float>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<std::string, double>(obj, objPath, ctx, scaleName, scaleType));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<std::string, ColorRGBA>(obj, objPath, ctx, scaleName, scaleType));
        default:
          THROW_RUNTIME_EX("JSON parse error - range type is unsupported: " +
                           std::to_string(static_cast<int>(rangeType)));
      }
    default:
      THROW_RUNTIME_EX("JSON parse error - domain type is unsupported.");
  }
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

template <class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<ColorRGBA>>& destData,
                            ScaleDomainRangeData<TT>* srcData) {
  THROW_RUNTIME_EX("Cannot convert a numeric value to a color.");
}

template <typename DomainType, typename RangeType>
ScaleRef<DomainType, RangeType>::ScaleRef(const QueryRendererContextShPtr& ctx,
                                          const ScaleShPtr& scalePtr,
                                          BaseRenderProperty* rndrProp)
    : BaseScaleRef(ctx, scalePtr, rndrProp), _coercedDomainData(nullptr), _coercedRangeData(nullptr) {
  ScaleDomainRangeData<unsigned int>* uintDomain;
  ScaleDomainRangeData<int>* intDomain;
  ScaleDomainRangeData<float>* floatDomain;
  ScaleDomainRangeData<double>* doubleDomain;
  ScaleDomainRangeData<std::string>* stringDomain;

  BaseScaleDomainRangeData* domainData = scalePtr->getDomainData();
  BaseScaleDomainRangeData* rangeData = scalePtr->getRangeData();

  if (domainData->getTypeInfo() != typeid(DomainType)) {
    uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(domainData);
    intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(domainData);
    floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(domainData);
    doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(domainData);
    stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(domainData);

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
    } else {
      THROW_RUNTIME_EX("Cannot create scale reference - unsupported domain type.");
    }
  } else {
    _coercedDomainData = nullptr;
  }

  if (rangeData->getTypeInfo() != typeid(RangeType)) {
    uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(rangeData);
    intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(rangeData);
    floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(rangeData);
    doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(rangeData);
    stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(rangeData);

    if (uintDomain) {
      convertDomainRangeData(_coercedRangeData, uintDomain);
    } else if (intDomain) {
      convertDomainRangeData(_coercedRangeData, intDomain);
    } else if (floatDomain) {
      convertDomainRangeData(_coercedRangeData, floatDomain);
    } else if (doubleDomain) {
      convertDomainRangeData(_coercedRangeData, doubleDomain);
    } else {
      THROW_RUNTIME_EX("Cannot create scale reference - unsupported range type.");
    }
  } else {
    _coercedRangeData = nullptr;
  }
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
void ScaleRef<DomainType, RangeType>::bindUniformsToRenderer(Shader* activeShader, const std::string& extraSuffix) {
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

  RUNTIME_EX_ASSERT(executor != nullptr, "An executor is not defined. Cannot numerically convert a string column.");

  const DataVBOShPtr& dataTable = _rndrPropPtr->getDataTablePtr();

  RUNTIME_EX_ASSERT(dataTable != nullptr,
                    "A data table is not referenced by render property \"" + _rndrPropPtr->getName() +
                        "\". Cannot numerically convert a string column.");

  SqlQueryDataTable* sqlDataTable = dynamic_cast<SqlQueryDataTable*>(dataTable.get());

  RUNTIME_EX_ASSERT(sqlDataTable != nullptr,
                    "The data table referenced by render property \"" + _rndrPropPtr->getName() +
                        "\" is not an sql data table. Cannot numerically convert a string column");

  std::string tableName = sqlDataTable->getTableName();
  RUNTIME_EX_ASSERT(tableName.length() != 0,
                    "The sql data table referenced by render property \"" + _rndrPropPtr->getName() +
                        "\" is not properly initialized. It is missing a \"dbTableName\" property. Cannot numerically "
                        "convert a string column");

  std::string colName = _rndrPropPtr->getDataColumnName();
  RUNTIME_EX_ASSERT(
      colName.length() != 0,
      "The render property \"" + _rndrPropPtr->getName() +
          "\" is missing a column name to reference in the data. Cannot numerically convert a string column.");

  // std::vector<std::string>& vec = domainData->getVectorData();
  // _coercedDomainData.reset(
  //     new ScaleDomainRangeData<DomainType>(domainData->getName(), vec.size(), domainData->useString()));
  // std::vector<DomainType>& coercedVec = _coercedDomainData->getVectorData();
  // for (size_t i = 0; i < vec.size(); ++i) {
  //   // get data from the executor
  //   coercedVec[i] = static_cast<DomainType>(executor->getStringId(tableName, colName, vec[i]));
  // }
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
                                               const DataVBOShPtr& dataPtr) {
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
                          "JSON parse error - a data reference for the mark is not defined. Cannot access \"field\".");
        RUNTIME_EX_ASSERT(mitr->value.IsString(), "JSON parse error - \"field\" property for mark must be a string.");

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
      THROW_RUNTIME_EX(
          "JSON parse error - invalid mark property object. Must contain a data reference via a \"field\" property or "
          "a \"value\" property.");
    }

    if ((mitr = obj.FindMember(scaleProp.c_str())) != obj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_scaleJsonPath, mitr->value)) {
        RUNTIME_EX_ASSERT(_useScale,
                          "JSON parse error - render property \"" + _name + "\" does not support scale references.");

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
    _scaleConfigPtr = nullptr;
    _initValueFromJSONObj(obj);
  }
}

void BaseRenderProperty::initializeFromData(const std::string& columnName, const DataVBOShPtr& dataPtr) {
  RUNTIME_EX_ASSERT(
      dataPtr != nullptr,
      "Cannot initialize mark property " + _name + " from data. A valid data reference hasn't been initialized.");

  _dataPtr = dataPtr;
  _vboAttrName = columnName;
  _vboPtr = dataPtr->getColumnDataVBO(columnName);
  _vboInitType = VboInitType::FROM_DATAREF;

  _prntMark->setPropsDirty();

  _initTypeFromVbo();
}

std::string BaseRenderProperty::getInGLSLType() const {
  RUNTIME_EX_ASSERT(_inType != nullptr,
                    "BaseRenderProperty::getInGLSLType(): input type for \"" + _name + "\" is uninitialized.");

  if (_scaleConfigPtr != nullptr) {
    std::string glslType = _scaleConfigPtr->getDomainTypeGL()->glslType();
    RUNTIME_EX_ASSERT(glslType == _inType->glslType(),
                      "BaseRenderProperty::getInGLSLType(): the domain type for scale \"" +
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
      "BaseRenderProperty::getOutGLSLType(): output type for mark property \"" + _name + "\" is uninitialized.");

  return (_outType->glslType());
}

template <>
RenderProperty<ColorRGBA, 1>::RenderProperty(BaseMark* prntMark,
                                             const std::string& name,
                                             const QueryRendererContextShPtr& ctx,
                                             bool useScale,
                                             bool flexibleType)
    : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType), _mult(), _offset(), _uniformVal() {
  _inType.reset(new TypeGL<float, 4>());
  _outType.reset(new TypeGL<float, 4>());
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initScaleFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsString(),
                    "JSON parse error - scale reference for mark property \"" + _name + "\" must be a string.");

  RUNTIME_EX_ASSERT(_ctx != nullptr && _scaleConfigPtr == nullptr,
                    "JSON parse error - cannot initialize mark property \"" + _name +
                        "\" from a scale. The context is uninitialized or a scale is already being referenced.");

  // TODO(croot): We probably need a better way to deal with types. We've got an _inType that is either defined
  // by an incoming data reference or an explicit value (or set of values). The latter is easy. We already
  // have the type in T/numComponents of the template. But the data reference is trickier.

  ScaleShPtr scalePtr = _ctx->getScale(obj.GetString());
  RUNTIME_EX_ASSERT(
      scalePtr != nullptr,
      "JSON parse error - the scale \"" + std::string(obj.GetString()) + "\" does not exist in the json.");

  if (!_scaleConfigPtr) {
    _prntMark->setShaderDirty();
  } else {
    ScaleShPtr prevPtr = _scaleConfigPtr->getScalePtr();
    if (!prevPtr || scalePtr.get() != prevPtr.get()) {
      _prntMark->setShaderDirty();
    }
  }

  if (dynamic_cast<TypeGL<unsigned int, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<unsigned int, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<TypeGL<int, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<int, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<TypeGL<float, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<float, T>(_ctx, scalePtr, this));
  } else if (dynamic_cast<TypeGL<double, 1>*>(_inType.get())) {
    _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
  } else {
    THROW_RUNTIME_EX("Scale domain with shader type \"" + scalePtr->getDomainTypeGL()->glslType() +
                     "\" and data with shader type \"" + _inType->glslType() + "\" are not supported to work together");
  }
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
    _inType.reset(new TypeGL<T, numComponents>());
    _outType.reset(new TypeGL<T, numComponents>());

    _vboPtr = nullptr;
    _vboInitType = VboInitType::FROM_VALUE;
    _prntMark->setPropsDirty();
  } else {
    // TODO(croot): do we need to set props dirty on the parent?
  }

  _uniformVal = val;
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::bindUniformToRenderer(Shader* activeShader,
                                                             const std::string& uniformAttrName) const {
  RUNTIME_EX_ASSERT(_vboPtr == nullptr,
                    "BaseRenderProperty::bindUniformToRenderer(): A vertex buffer is defined. There should be no "
                    "vertex buffers defined for uniform properties.");

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

  if (_vboInitType != VboInitType::FROM_VALUE) {
    _inType.reset(new TypeGL<float, 4>());
    _outType.reset(new TypeGL<float, 4>());

    _vboPtr = nullptr;
    _vboInitType = VboInitType::FROM_VALUE;
    _prntMark->setPropsDirty();
  } else {
    // TODO(croot): do we need to set props dirty on the parent?
  }

  _uniformVal = val;
}

template <>
void RenderProperty<ColorRGBA, 1>::bindUniformToRenderer(Shader* activeShader,
                                                         const std::string& uniformAttrName) const {
  RUNTIME_EX_ASSERT(_vboPtr == nullptr,
                    "BaseRenderProperty::bindUniformToRenderer(): A vertex buffer is defined. There should be no "
                    "vertex buffers defined for uniform properties.");

  // TODO(croot): deal with numComponents here by using a vector instead?
  activeShader->setUniformAttribute<std::array<float, 4>>(uniformAttrName, _uniformVal.getColorArray());
}

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj) {
  // TODO: what about offsets / mults for colors?
}

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsString(), "JSON parse error - value for color property \"" + _name + "\" must be a string.");

  ColorRGBA color(obj.GetString());

  initializeValue(color);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initTypeFromVbo() {
  RUNTIME_EX_ASSERT(_vboPtr != nullptr,
                    "Vertex buffer is uninitialized. Cannot initialize type for mark property \"" + _name + "\".");

  TypeGLShPtr vboType = _vboPtr->getAttributeTypeGL(_vboAttrName);

  if (_flexibleType) {
    _inType = vboType;
    _outType = vboType;
  } else {
    // the render property is rigid, meaning it cannot accept
    // different types. So verify that the type of the attribute
    // in the vbo is appropriate.
    RUNTIME_EX_ASSERT((*_outType) == (*vboType),
                      "The vertex buffer type does not match the output type for mark property \"" + _name + "\".");
  }
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_verifyScale() {
}

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale() {
  RUNTIME_EX_ASSERT(_scaleConfigPtr != nullptr,
                    "Cannot verify scale for mark property \"" + _name + "\". Scale reference is uninitialized.");

  TypeGLShPtr vboType = _scaleConfigPtr->getRangeTypeGL();

  // colors need to be a specific type
  RUNTIME_EX_ASSERT(ColorRGBA::isValidTypeGL(vboType),
                    "Vertex buffer to use for mark property \"" + _name + "\" is not an appropriate type for colors.");
}

BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx)
    : _type(geomType),
      key(this, "key", ctx, false, false),
      _invalidKey(ctx->getInvalidKey()),
      _dataPtr(nullptr),
      _shaderPtr(nullptr),
      _ctx(ctx),
      _shaderDirty(true),
      _propsDirty(true),
      _vboProps(),
      _uniformProps(),
      _vao(0) {
}

BaseMark::BaseMark(GeomType geomType,
                   const QueryRendererContextShPtr& ctx,
                   const rapidjson::Value& obj,
                   const rapidjson::Pointer& objPath)
    : BaseMark(geomType, ctx) {
  _initFromJSONObj(obj, objPath);
}

BaseMark::~BaseMark() {
  if (_vao) {
    MAPD_CHECK_GL_ERROR(glDeleteVertexArrays(1, &_vao));
  }
}

void BaseMark::_initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - definition for marks must be an object.");

  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot): move the "from" to a const somewhere
  std::string fromProp = "from";
  if ((mitr = obj.FindMember(fromProp.c_str())) != obj.MemberEnd()) {
    const rapidjson::Value& fromObj = mitr->value;

    if (!_ctx->isJSONCacheUpToDate(_dataPtrJsonPath, fromObj)) {
      RUNTIME_EX_ASSERT(fromObj.IsObject(), "JSON parse error - mark data reference must be an object.");

      RUNTIME_EX_ASSERT((mitr = fromObj.FindMember("data")) != fromObj.MemberEnd() && mitr->value.IsString(),
                        "JSON parse error - mark data reference must contain a \"data\" string property.");

      _dataPtr = _ctx->getDataTable(mitr->value.GetString());
    }

    _dataPtrJsonPath = objPath.Append(fromProp.c_str(), fromProp.length());
  } else {
    // TODO(croot): what about references???
    _dataPtr = nullptr;
  }
}

void BaseMark::_buildVertexArrayObjectFromProperties(Shader* activeShader) {
  if (!_propsDirty) {
    // early out
    return;
  }

  if (_vao) {
    MAPD_CHECK_GL_ERROR(glDeleteVertexArrays(1, &_vao));
    _vao = 0;
  }

  MAPD_CHECK_GL_ERROR(glGenVertexArrays(1, &_vao));
  MAPD_CHECK_GL_ERROR(glBindVertexArray(_vao));

  _initPropertiesForRendering(activeShader);

  // reset so no VAO is bound
  // TODO: Actually look for a currently bound vao at start of function and reset to that?
  // NOTE: No need to unbind at this point since we're going directly into rendering
  // after this function?
  // MAPD_CHECK_GL_ERROR(glBindVertexArray(0));

  _propsDirty = false;
}

void BaseMark::_bindToRenderer(Shader* activeShader) {
  _buildVertexArrayObjectFromProperties(activeShader);

  MAPD_CHECK_GL_ERROR(glBindVertexArray(_vao));

  _bindPropertiesToRenderer(activeShader);
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
                    "JSON parse error - mark objects must have a \"properties\" property.");

  const rapidjson::Value& propObj = mitr->value;

  if (!_ctx->isJSONCacheUpToDate(_propertiesJsonPath, propObj)) {
    std::vector<BaseRenderProperty*> usedProps{&x, &y, &size, &fillColor};  // TODO(croot) add z

    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());

    RUNTIME_EX_ASSERT(propObj.IsObject(),
                      "JSON parse error - The \"properties\" property of marks must be a json object.");

    // TODO(croot): move "x" to a const somewhere
    std::string xProp = "x";
    RUNTIME_EX_ASSERT((mitr = propObj.FindMember(xProp.c_str())) != propObj.MemberEnd(),
                      "JSON parse error - \"" + xProp + "\" mark property must exist for point marks.");

    if (!_ctx->isJSONCacheUpToDate(_xJsonPath, mitr->value)) {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
      RUNTIME_EX_ASSERT(
          (mitr->value.IsObject() || mitr->value.IsNumber()),
          "JSON parse error - \"" + xProp + "\" mark property must be a scale/data reference or a number.");
      x.initializeFromJSONObj(mitr->value, _xJsonPath, _dataPtr);
    } else {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
    }

    // TODO(croot): move "y" to a const somewhere
    std::string yProp = "y";
    RUNTIME_EX_ASSERT((mitr = propObj.FindMember(yProp.c_str())) != propObj.MemberEnd(),
                      "JSON parse error - \"" + yProp + "\" mark property must exist for point marks.");

    if (!_ctx->isJSONCacheUpToDate(_yJsonPath, mitr->value)) {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
      RUNTIME_EX_ASSERT(
          (mitr->value.IsObject() || mitr->value.IsNumber()),
          "JSON parse error - \"" + yProp + "\" mark property must be a scale/data reference or a number.");
      y.initializeFromJSONObj(mitr->value, _yJsonPath, _dataPtr);
    } else {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
    }

    // TODO(croot): move "z" to a const somewhere
    std::string zProp = "z";
    if ((mitr = propObj.FindMember(zProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_zJsonPath, mitr->value)) {
        _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
        RUNTIME_EX_ASSERT(
            (mitr->value.IsObject() || mitr->value.IsNumber()),
            "JSON parse error - \"" + zProp + "\" mark property must be a scale/data reference or a number.");
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
    RUNTIME_EX_ASSERT((mitr = propObj.FindMember(sizeProp.c_str())) != propObj.MemberEnd(),
                      "JSON parse error - \"" + sizeProp + "\" mark property must exist for point marks.");

    if (!_ctx->isJSONCacheUpToDate(_sizeJsonPath, mitr->value)) {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
      RUNTIME_EX_ASSERT(
          (mitr->value.IsObject() || mitr->value.IsNumber()),
          "JSON parse error - \"" + sizeProp + "\" mark property must be a scale/data reference or a number.");
      size.initializeFromJSONObj(mitr->value, _sizeJsonPath, _dataPtr);
    } else {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
    }

    // TODO(croot): move "fillColor" to a const somewhere
    std::string fillColorProp = "fillColor";
    RUNTIME_EX_ASSERT((mitr = propObj.FindMember(fillColorProp.c_str())) != propObj.MemberEnd(),
                      "JSON parse error - \"" + fillColorProp + "\" mark property must exist for point marks.");

    if (!_ctx->isJSONCacheUpToDate(_fillColorJsonPath, mitr->value)) {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
      RUNTIME_EX_ASSERT(
          (mitr->value.IsObject() || mitr->value.IsString()),
          "JSON parse error - \"" + fillColorProp + "\" mark property must be a scale/data reference or a string.");
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
          RUNTIME_EX_ASSERT(
              mitr->value.IsObject(),
              "JSON parse error - \"id\" is a special mark property that must be defined by a data reference.");
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
    BaseVertexBufferShPtr vboPtr;
    static const BaseVertexBuffer::VertexBufferType resultVBO =
        BaseVertexBuffer::VertexBufferType::QUERY_RESULT_VERTEX_BUFFER;
    if ((((vboPtr = x.getVboPtr()) && vboPtr->type() == resultVBO) ||
         ((vboPtr = y.getVboPtr()) && vboPtr->type() == resultVBO) ||
         ((vboPtr = size.getVboPtr()) && vboPtr->type() == resultVBO) ||
         // TODO(croot): what if we have multiple sqls? How do we handle the "key" value then?
         ((vboPtr = fillColor.getVboPtr()) && vboPtr->type() == resultVBO)) &&
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
  // TODO: need to determine a build-appropriate way to access
  // shaders. The best way probably is to create a cmake build
  // script that converts all shaders into their own header
  // files with static strings of the shader's source to access.

  if (!_shaderDirty) {
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
                        "Cannot find a properly defined \"" + propFuncName + "\" function in the vertex shader.");

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
  // TODO: How would we share shaders across different
  // query renderers?
  _shaderPtr.reset(new Shader(vertSrc, fragSrc));

  _shaderDirty = false;
}

void PointMark::_initPropertiesForRendering(Shader* activeShader) {
  int cnt = 0;
  int vboSize = 0;
  for (auto& itr : _vboProps) {
    cnt++;
    if (cnt == 1) {
      vboSize = itr->size();
    } else {
      RUNTIME_EX_ASSERT(itr->size() == vboSize,
                        "Invalid point mark. The sizes of the vertex buffer attributes do not match.");
    }
    itr->bindToRenderer(activeShader);
  }
}

void PointMark::_bindPropertiesToRenderer(Shader* activeShader) {
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

void PointMark::draw() {
  _shaderPtr->bindToRenderer();

  _bindToRenderer(_shaderPtr.get());

  // TODO: render state stack -- push/pop
  MAPD_CHECK_GL_ERROR(glEnable(GL_PROGRAM_POINT_SIZE));

  // now draw points
  // TODO: What about the possibility of index buffers?
  // Probably don't have to worry about them here since
  // we're specifically looking at a point config that
  // is deemed not to have any index buffers, but we
  // need to have a way to generically draw bound buffers
  // which would be best with a renderer class
  MAPD_CHECK_GL_ERROR(glDrawArrays(GL_POINTS, 0, x.size()));

  // unset state
  MAPD_CHECK_GL_ERROR(glDisable(GL_PROGRAM_POINT_SIZE));
}

void PointMark::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    BaseMark::_initFromJSONObj(obj, objPath);
    _initPropertiesFromJSONObj(obj, objPath);
    _updateShader();
  } else if (_jsonPath != objPath) {
    // TODO(croot) - Bug! What if the cache is up-to-date, but the path has changed -- we need to update
    // the paths for this and all sub-objects that manage paths.
    // NOTE: We should never get in here since marks
    // are stored as an array but if we ever change the storage container
    // in the future (i.e. an unordered_map or the like), we'd reach this
    THROW_RUNTIME_EX("The cache for mark \"" + RapidJSONUtils::getPointerPath(objPath) +
                     "\" is up-to-date, but the path in the JSON has changed from " +
                     RapidJSONUtils::getPointerPath(_jsonPath) + " to " + RapidJSONUtils::getPointerPath(objPath) +
                     ", so the path caches need updating. This "
                     "has yet to be implemented.");
  }

  // TODO(croot): if the obj hasn't changed, but the path has, we need
  // to trickle that path change to all subobjects who cache their
  // json data. How should we handle this?
  _jsonPath = objPath;
}

BaseMark::GeomType MapD_Renderer::getMarkTypeFromJSONObj(const rapidjson::Value& obj) {
  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("type")) != obj.MemberEnd() && itr->value.IsString(),
                    "JSON parse error - a mark object must have a \"type\" string property.");

  std::string strGeomType(itr->value.GetString());

  if (strGeomType == "points") {
    return BaseMark::GeomType::POINTS;
  } else {
    THROW_RUNTIME_EX("JSON parse error - a mark of type \"" + strGeomType + "\" is unsupported.");
  }
}

GeomConfigShPtr MapD_Renderer::createMark(const rapidjson::Value& obj,
                                          const rapidjson::Pointer& objPath,
                                          const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - marks must be objects.");

  switch (getMarkTypeFromJSONObj(obj)) {
    case BaseMark::GeomType::POINTS:
      return GeomConfigShPtr(new PointMark(obj, objPath, ctx));
      break;
  }

  return GeomConfigShPtr(nullptr);
}
