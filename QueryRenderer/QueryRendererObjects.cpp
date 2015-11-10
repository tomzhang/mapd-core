#include "QueryRendererObjects.h"
#include "ShaderUtils.h"
#include "shaders/pointTemplate_vert.h"
#include "shaders/pointTemplate_frag.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"

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
  // std::vector<str_itr> scopeStack;
  // scopeStack.push_back(signature_range.end());

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

DataType getDataTypeFromJSONObj(const rapidjson::Value& obj) {
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
      if (ColorRGBA::isColorString(val)) {
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
    // "shaders/linearScaleTemplate.vert",  // LINEAR
    // "shaders/ordinalScaleTemplate.vert"  // ORDINAL

    ::MapD_Renderer::LinearScaleTemplate_vert::source,  // LINEAR
    ::MapD_Renderer::OrdinalScaleTemplate_vert::source  // ORDINAL
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx)
    : name(""), type(ScaleType::LINEAR), _domainType(nullptr), _rangeType(nullptr), _ctx(ctx), _useClamp(false) {
}

BaseScale::BaseScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(ctx) {
  _initFromJSONObj(obj);
}

BaseScale::~BaseScale() {
}

void BaseScale::_initFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - scale items must be objects.");

  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
                    "JSON parse error - scale objects must contain a \"name\" string property.");

  name = itr->value.GetString();

  if ((itr = obj.FindMember("type")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(), "JSON parse error - \"type\" property in scale objects must be a string.");
    std::string strScaleType(itr->value.GetString());

    if (strScaleType == "linear") {
      type = ScaleType::LINEAR;

      if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(itr->value.IsBool(),
                          "JSON parse error - the \"clamp\" property for linear scales must be a boolean.");

        _useClamp = itr->value.GetBool();
      }
    } else if (strScaleType == "ordinal") {
      type = ScaleType::ORDINAL;
    } else {
      THROW_RUNTIME_EX("JSON parse error - scale type \"" + strScaleType + "\" is not a supported type.");
    }
  }
}

std::string BaseScale::getScaleGLSLFuncName() {
  std::string scaleName;

  switch (type) {
    case ScaleType::LINEAR:
      scaleName = "Linear";
      break;
    case ScaleType::ORDINAL:
      scaleName = "Ordinal";
      break;
    default:
      THROW_RUNTIME_EX("BaseScale::getScaleGLSLFuncName(): scale type is not supported.");
  }

  std::ostringstream ss;

  ss << "get" << scaleName << "Scale_" << name;

  return ss.str();
}

template <typename T>
void ScaleDomainRangeData<T>::initializeFromJSONObj(const rapidjson::Value& obj,
                                                    const QueryRendererContextShPtr& ctx,
                                                    BaseScale::ScaleType type) {
  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  bool isObject = false, isString = false;
  DataVBOShPtr tablePtr;
  DataColumnShPtr columnPtr;

  if (_useString) {
    RUNTIME_EX_ASSERT((mitr = obj.FindMember(_name.c_str())) != obj.MemberEnd() &&
                          ((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
                           (mitr->value.IsArray() && mitr->value.Size())),
                      "JSON parse error - scale domain/range is invalid.");
  } else {
    RUNTIME_EX_ASSERT((mitr = obj.FindMember(_name.c_str())) != obj.MemberEnd() &&
                          ((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())),
                      "JSON parse error - scale domain/range is invalid.");
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
    RUNTIME_EX_ASSERT(tablePtr->getType() == BaseDataTableVBO::DataTableType::OTHER,
                      "JSON parse error - scale data references only support JSON-embedded data tables currently.");

    DataTable* dataTablePtr = dynamic_cast<DataTable*>(tablePtr.get());

    columnPtr = dataTablePtr->getColumn(mitr->value.GetString());

    TDataColumn<T>* dataColumnPtr = dynamic_cast<TDataColumn<T>*>(columnPtr.get());

    _vectorPtr = dataColumnPtr->getColumnData();

    _updateVectorDataByType(dataColumnPtr, type);
  } else if (isString) {
    _setFromStringValue(mitr->value.GetString(), type);
  } else {
    _vectorPtr.reset(new std::vector<T>());

    // gather all the items
    // TODO(croot): can improve this by allocating the full vector
    // size upfront, but cost should be negligable since large domains/ranges
    // would be really rare. Normally we're just talking about a couple
    // handfulls of values max.
    DataType itemType;
    for (vitr = jsonObj.Begin(); vitr != jsonObj.End(); ++vitr) {
      itemType = getDataTypeFromJSONObj(*vitr);

      RUNTIME_EX_ASSERT(itemType == dataType,
                        "JSON parse error - scale domain/range item " + std::to_string(vitr - jsonObj.Begin()) +
                            " has an invalid type.");

      _pushItem(*vitr);
    }
  }
}

template <typename T>
inline TypeGLShPtr ScaleDomainRangeData<T>::getTypeGL() {
  return TypeGLShPtr(new TypeGL<T>());
}

template <>
inline TypeGLShPtr ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
  // TODO(croot): combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions
  return TypeGLShPtr(new TypeGL<float, 4>());
}

template <>
const DataType ScaleDomainRangeData<unsigned int>::dataType = DataType::UINT;

template <>
unsigned int ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetUint();
}

template <>
void MapD_Renderer::ScaleDomainRangeData<unsigned int>::_pushItem(const rapidjson::Value& obj) {
  _vectorPtr->push_back(obj.GetUint());
}

template <>
const DataType ScaleDomainRangeData<int>::dataType = DataType::INT;

template <>
int ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetInt();
}

template <>
void MapD_Renderer::ScaleDomainRangeData<int>::_pushItem(const rapidjson::Value& obj) {
  _vectorPtr->push_back(obj.GetInt());
}

template <>
const DataType ScaleDomainRangeData<float>::dataType = DataType::FLOAT;

template <>
float ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return static_cast<float>(obj.GetDouble());
}

template <>
void MapD_Renderer::ScaleDomainRangeData<float>::_pushItem(const rapidjson::Value& obj) {
  _vectorPtr->push_back(static_cast<float>(obj.GetDouble()));
}

template <>
const DataType ScaleDomainRangeData<double>::dataType = DataType::DOUBLE;

template <>
double ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return obj.GetDouble();
}

template <>
void MapD_Renderer::ScaleDomainRangeData<double>::_pushItem(const rapidjson::Value& obj) {
  _vectorPtr->push_back(obj.GetDouble());
}

template <>
const DataType ScaleDomainRangeData<ColorRGBA>::dataType = DataType::COLOR;

template <>
ColorRGBA ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj) {
  return ColorRGBA(obj.GetString());
}

template <>
void MapD_Renderer::ScaleDomainRangeData<ColorRGBA>::_pushItem(const rapidjson::Value& obj) {
  ColorRGBA color(obj.GetString());
  _vectorPtr->push_back(color);
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
Scale<DomainType, RangeType>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx)
    : BaseScale(obj, ctx), _domainPtr("domain", false), _rangePtr("range", true), _defaultVal() {
  _initGLTypes();
  _initFromJSONObj(obj);
}

template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::~Scale() {
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_initFromJSONObj(const rapidjson::Value& obj) {
  _domainPtr.initializeFromJSONObj(obj, _ctx, type);
  _rangePtr.initializeFromJSONObj(obj, _ctx, type);

  rapidjson::Value::ConstMemberIterator mitr;
  if ((mitr = obj.FindMember("default")) != obj.MemberEnd()) {
    DataType itemType = getDataTypeFromJSONObj(mitr->value);

    RUNTIME_EX_ASSERT(itemType == _rangePtr.dataType,
                      "JSON parse error - scale \"" + name + "\" default is not the same type as its range.");

    _defaultVal = _rangePtr.getDataValueFromJSONObj(mitr->value);
  }
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_initGLTypes() {
  _domainType = _domainPtr.getTypeGL();
  _rangeType = _rangePtr.getTypeGL();
}

template <typename DomainType, typename RangeType>
std::string Scale<DomainType, RangeType>::getGLSLCode() {
  RUNTIME_EX_ASSERT(_domainPtr.size() > 0 && _rangePtr.size() > 0,
                    "Scale::getGLSLCode(): domain/range of scale \"" + name + "\" has no value.");

  // std::string shaderCode = getShaderCodeFromFile(scaleVertexShaderSource[static_cast<int>(type)]);
  std::string shaderCode = scaleVertexShaderSource[static_cast<int>(type)];
  std::ostringstream ss;

  boost::replace_first(shaderCode, "<domainType>", _domainType->glslType());

  boost::replace_first(shaderCode, "<rangeType>", _rangeType->glslType());

  // ss << _domainPtr->size();
  ss << _domainPtr.size();
  boost::replace_first(shaderCode, "<numDomains>", ss.str());

  ss.str("");
  ss << _rangePtr.size();
  boost::replace_first(shaderCode, "<numRanges>", ss.str());

  boost::replace_all(shaderCode, "<name>", name);

  if (type == ScaleType::LINEAR) {
    ss.str("");
    ss << _useClamp;
    boost::replace_all(shaderCode, "<useClamp>", ss.str());
  }

  shaderCode += '\n';

  return shaderCode;
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::bindUniformsToRenderer(Shader* activeShader) {
  // activeShader->setUniformAttribute(getDomainGLSLUniformName(), *_domainPtr);
  activeShader->setUniformAttribute(getDomainGLSLUniformName(), _domainPtr.getVectorData());
  // activeShader->setUniformAttribute(getRangeGLSLUniformName(), *_rangePtr);
  activeShader->setUniformAttribute(getRangeGLSLUniformName(), _rangePtr.getVectorData());

  if (type == ScaleType::ORDINAL) {
    activeShader->setUniformAttribute(getRangeDefaultGLSLUniformName(), _defaultVal);
  }
}

ScaleShPtr MapD_Renderer::createScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  rapidjson::Value::ConstMemberIterator itr;

  bool isObject = false;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
                        ((isObject = itr->value.IsObject()) || (itr->value.IsArray() && itr->value.Size())),
                    "JSON parse error - \"domain\" property for scales must exist and must be an object or an array.");

  DataType domainType;

  if (isObject) {
    domainType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
  } else {
    domainType = getDataTypeFromJSONObj(itr->value[0]);
  }

  bool isString;
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

  switch (domainType) {
    case DataType::UINT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<unsigned int, unsigned int>(obj, ctx));
        case DataType::INT:
          return ScaleShPtr(new Scale<unsigned int, int>(obj, ctx));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<unsigned int, float>(obj, ctx));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<unsigned int, double>(obj, ctx));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<unsigned int, ColorRGBA>(obj, ctx));
      }
    case DataType::INT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<int, unsigned int>(obj, ctx));
        case DataType::INT:
          return ScaleShPtr(new Scale<int, int>(obj, ctx));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<int, float>(obj, ctx));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<int, double>(obj, ctx));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<int, ColorRGBA>(obj, ctx));
      }
    case DataType::FLOAT:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<float, unsigned int>(obj, ctx));
        case DataType::INT:
          return ScaleShPtr(new Scale<float, int>(obj, ctx));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<float, float>(obj, ctx));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<float, double>(obj, ctx));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<float, ColorRGBA>(obj, ctx));
      }
    case DataType::DOUBLE:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<double, unsigned int>(obj, ctx));
        case DataType::INT:
          return ScaleShPtr(new Scale<double, int>(obj, ctx));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<double, float>(obj, ctx));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<double, double>(obj, ctx));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<double, ColorRGBA>(obj, ctx));
      }
    case DataType::COLOR:
      switch (rangeType) {
        case DataType::UINT:
          return ScaleShPtr(new Scale<ColorRGBA, unsigned int>(obj, ctx));
        case DataType::INT:
          return ScaleShPtr(new Scale<ColorRGBA, int>(obj, ctx));
        case DataType::FLOAT:
          return ScaleShPtr(new Scale<ColorRGBA, float>(obj, ctx));
        case DataType::DOUBLE:
          return ScaleShPtr(new Scale<ColorRGBA, double>(obj, ctx));
        case DataType::COLOR:
          return ScaleShPtr(new Scale<ColorRGBA, ColorRGBA>(obj, ctx));
      }
    default:
      THROW_RUNTIME_EX("JSON parse error - domain type is unsupported.");
  }
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

void BaseRenderProperty::initializeFromJSONObj(const rapidjson::Value& obj, const DataVBOShPtr& dataPtr) {
  if (obj.IsObject()) {
    rapidjson::Value::ConstMemberIterator mitr;

    if ((mitr = obj.FindMember("scale")) != obj.MemberEnd()) {
      RUNTIME_EX_ASSERT(_useScale,
                        "JSON parse error - render property \"" + _name + "\" does not support scale references.");

      _initScaleFromJSONObj(mitr->value);
      _verifyScale();
    }

    if ((mitr = obj.FindMember("field")) != obj.MemberEnd()) {
      // TODO: check for an object here
      RUNTIME_EX_ASSERT(dataPtr != nullptr,
                        "JSON parse error - a data reference for the mark is not defined. Cannot access \"field\".");
      RUNTIME_EX_ASSERT(mitr->value.IsString(), "JSON parse error - \"field\" property for mark must be a string.");

      initializeFromData(mitr->value.GetString(), dataPtr);
    } else if ((mitr = obj.FindMember("value")) != obj.MemberEnd()) {
      _initValueFromJSONObj(mitr->value, (dataPtr != nullptr ? dataPtr->numRows() : 1));
    } else {
      // need some value source, either by "field" or by "value"
      THROW_RUNTIME_EX(
          "JSON parse error - invalid mark property object. Must contain a data reference via a \"field\" property or "
          "a \"value\" property.");
    }

    _initFromJSONObj(obj);

  } else {
    _initValueFromJSONObj(obj, (dataPtr != nullptr ? dataPtr->numRows() : 1));
  }
}

void BaseRenderProperty::initializeFromData(const std::string& columnName, const DataVBOShPtr& dataPtr) {
  RUNTIME_EX_ASSERT(
      dataPtr != nullptr,
      "Cannot initialize mark property " + _name + " from data. A valid data reference hasn't been initialized.");

  _vboAttrName = columnName;
  _vboPtr = dataPtr->getColumnDataVBO(columnName);

  _initTypeFromVbo();

  // if (_scaleConfigPtr == nullptr) {
  //   _initTypeFromVbo();
  // }
}

void BaseRenderProperty::_initScaleFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsString(),
                    "JSON parse error - scale reference for mark property \"" + _name + "\" must be a string.");

  RUNTIME_EX_ASSERT(_ctx != nullptr && _scaleConfigPtr == nullptr,
                    "JSON parse error - cannot initialize mark property \"" + _name +
                        "\" from a scale. The context is uninitialized or a scale is already being referenced.");

  _scaleConfigPtr = _ctx->getScale(obj.GetString());
}

std::string BaseRenderProperty::getInGLSLType() const {
  RUNTIME_EX_ASSERT(_inType != nullptr,
                    "BaseRenderProperty::getInGLSLType(): input type for \"" + _name + "\" is uninitialized.");

  if (_scaleConfigPtr != nullptr) {
    std::string glslType = _scaleConfigPtr->getDomainType()->glslType();
    RUNTIME_EX_ASSERT(glslType == _inType->glslType(),
                      "BaseRenderProperty::getInGLSLType(): the domain type for scale \"" + _scaleConfigPtr->name +
                          "\" does not match the type for mark property \"" + _name + "\"");
    return glslType;
  }

  return _inType->glslType();
}

std::string BaseRenderProperty::getOutGLSLType() const {
  if (_scaleConfigPtr != nullptr) {
    return _scaleConfigPtr->getRangeType()->glslType();
  }

  RUNTIME_EX_ASSERT(
      _outType != nullptr,
      "BaseRenderProperty::getOutGLSLType(): output type for mark property \"" + _name + "\" is uninitialized.");

  return (_outType->glslType());
}

template <>
RenderProperty<ColorRGBA, 1>::RenderProperty(const std::string& name,
                                             const QueryRendererContextShPtr& ctx,
                                             bool useScale,
                                             bool flexibleType)
    : BaseRenderProperty(name, ctx, useScale, flexibleType), _mult(), _offset() {
  _inType.reset(new TypeGL<float, 4>());
  _outType.reset(new TypeGL<float, 4>());
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initFromJSONObj(const rapidjson::Value& obj) {
  // this is internally called at the appropriate time from
  // the base class's initialization function, so there's
  // no need to check that obj is valid since that should've
  // already been done.
  rapidjson::Value::ConstMemberIterator mitr;

  if ((mitr = obj.FindMember("mult")) != obj.MemberEnd()) {
    _mult = getNumValFromJSONObj<T>(mitr->value);
  }

  if ((mitr = obj.FindMember("offset")) != obj.MemberEnd()) {
    _offset = getNumValFromJSONObj<T>(mitr->value);
  }
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::initializeValue(const T& val, int numItems) {
  // TODO: this is a public function.. should I protect from already existing data?

  // TODO(croot): should I initialize the _inType/_outType regardless?
  _inType.reset(new TypeGL<T, numComponents>());
  _outType.reset(new TypeGL<T, numComponents>());

  std::vector<T> data(numItems, val);

  SequentialBufferLayout* vboLayout = new SequentialBufferLayout();
  vboLayout->addAttribute<T>(_name);

  VertexBuffer::BufferLayoutShPtr vboLayoutPtr;
  vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout*>(vboLayout));

  VertexBuffer* vbo = new VertexBuffer(data, vboLayoutPtr);
  _vboPtr.reset(vbo);
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems) {
  T val = getNumValFromJSONObj<T>(obj);

  initializeValue(val, numItems);
}

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj) {
  // TODO: what about offsets / mults for colors?
}

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems) {
  RUNTIME_EX_ASSERT(obj.IsString(), "JSON parse error - value for color property \"" + _name + "\" must be a string.");

  ColorRGBA color(obj.GetString());

  // TODO: combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions

  _inType.reset(new TypeGL<float, 4>());
  _outType.reset(new TypeGL<float, 4>());

  std::vector<ColorRGBA> data(numItems, color);

  SequentialBufferLayout* vboLayout = new SequentialBufferLayout();
  vboLayout->addAttribute(_name, BufferAttrType::VEC4F);

  VertexBuffer::BufferLayoutShPtr vboLayoutPtr;
  vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout*>(vboLayout));

  VertexBuffer* vbo = new VertexBuffer(data, vboLayoutPtr);
  // vbo->bufferData();
  _vboPtr.reset(vbo);
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

// TODO(croot): probably don't need a specialization for
// color in this case anymore with the advent of the "flexibleType"
template <>
void RenderProperty<ColorRGBA, 1>::_initTypeFromVbo() {
  RUNTIME_EX_ASSERT(_vboPtr != nullptr,
                    "Vertex buffer is uninitialized. Cannot initialize type for mark property \"" + _name + "\".");

  TypeGLShPtr vboType = _vboPtr->getAttributeTypeGL(_vboAttrName);

  // colors need to be a specific type
  if (!_scaleConfigPtr) {
    RUNTIME_EX_ASSERT(ColorRGBA::isValidTypeGL(vboType),
                      "Vertex buffer to use for mark property \"" + _name + "\" is not the appropriate type.");
  }

  if (_flexibleType) {
    _inType = vboType;
    _outType = vboType;
  } else {
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

  TypeGLShPtr vboType = _scaleConfigPtr->getRangeType();

  // colors need to be a specific type
  RUNTIME_EX_ASSERT(ColorRGBA::isValidTypeGL(vboType),
                    "Vertex buffer to use for mark property \"" + _name + "\" is not an appropriate type for colors.");
}

BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx)
    : type(geomType),
      key("key", ctx, false, false),
      _invalidKey(ctx->getInvalidKey()),
      _dataPtr(nullptr),
      _shaderPtr(nullptr),
      _ctx(ctx),
      _vao(0) {
}
BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx, const rapidjson::Value& obj)
    : BaseMark(geomType, ctx) {
  _initFromJSONObj(obj);
}

BaseMark::~BaseMark() {
  // std::cerr << "IN BaseMark DESTRUCTOR" << std::endl;

  if (_vao) {
    glDeleteVertexArrays(1, &_vao);
  }
}

void BaseMark::_initFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - definition for marks must be an object.");

  rapidjson::Value::ConstMemberIterator mitr;
  RUNTIME_EX_ASSERT((mitr = obj.FindMember("properties")) != obj.MemberEnd() && mitr->value.IsObject(),
                    "JSON parse error - marks must have a \"properties\" object property.");

  if ((mitr = obj.FindMember("from")) != obj.MemberEnd()) {
    const rapidjson::Value& fromObj = mitr->value;

    RUNTIME_EX_ASSERT(fromObj.IsObject(), "JSON parse error - mark data reference must be an object.");

    RUNTIME_EX_ASSERT((mitr = fromObj.FindMember("data")) != fromObj.MemberEnd() && mitr->value.IsString(),
                      "JSON parse error - mark data reference must contain a \"data\" string property.");

    _dataPtr = _ctx->getDataTable(mitr->value.GetString());
  }
}

void BaseMark::_buildVertexArrayObjectFromProperties(Shader* activeShader) {
  if (_vao) {
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
  }

  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);

  _initPropertiesForRendering(activeShader);

  // reset so no VAO is bound
  // TODO: Actually look for a currently bound vao at start of function and reset to that?
  // NOTE: No need to unbind at this point since we're going directly into rendering
  // after this function?
  // glBindVertexArray(0);
}

void BaseMark::_bindToRenderer(Shader* activeShader) {
  if (!_vao) {
    _buildVertexArrayObjectFromProperties(activeShader);
  }

  glBindVertexArray(_vao);

  _bindPropertiesToRenderer(activeShader);
}

PointMark::PointMark(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx)
    : BaseMark(POINTS, ctx, obj),
      x("x", ctx),
      y("y", ctx),
      z("z", ctx),
      size("size", ctx),

      // TODO(croot): let's log a warning and continue onwards if
      // hit testing is asked for, but the input sql data doesn't
      // have an id.
      id("id", ctx, false),
      fillColor("fillColor", ctx) {
  _initPropertiesFromJSONObj(obj);
  _initShader();
}

PointMark::~PointMark() {
  // std::cerr << "IN PointGeomConig DESTRUCTOR" << std::endl;
}

void PointMark::_initPropertiesFromJSONObj(const rapidjson::Value& obj) {
  // no need to check for the existence of the 'properties' member as
  // that should've been done by the base class constructor.
  const rapidjson::Value& propObj = obj["properties"];
  rapidjson::Value::ConstMemberIterator mitr;

  RUNTIME_EX_ASSERT(
      (mitr = propObj.FindMember("x")) != propObj.MemberEnd() && (mitr->value.IsObject() || mitr->value.IsNumber()),
      "JSON parse error - \"x\" mark property must exist and must be a scale/data reference or a number.");
  x.initializeFromJSONObj(mitr->value, _dataPtr);

  RUNTIME_EX_ASSERT(
      (mitr = propObj.FindMember("y")) != propObj.MemberEnd() && (mitr->value.IsObject() || mitr->value.IsNumber()),
      "JSON parse error - \"y\" mark property must exist and must be a scale/data reference or a number.");
  y.initializeFromJSONObj(mitr->value, _dataPtr);

  if ((mitr = propObj.FindMember("z")) != propObj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr->value.IsObject() || mitr->value.IsNumber(),
                      "JSON parse error - invalid type for \"z\" mark property.");
    z.initializeFromJSONObj(mitr->value, _dataPtr);
  }

  RUNTIME_EX_ASSERT(
      (mitr = propObj.FindMember("size")) != propObj.MemberEnd() && (mitr->value.IsObject() || mitr->value.IsNumber()),
      "JSON parse error - \"size\" mark property must exist and must be a scale/data reference or a number.");
  size.initializeFromJSONObj(mitr->value, _dataPtr);

  RUNTIME_EX_ASSERT(
      (mitr = propObj.FindMember("fillColor")) != propObj.MemberEnd() &&
          (mitr->value.IsObject() || mitr->value.IsString()),
      "JSON parse error - \"fillColor\" mark property must exist and must be a scale/data reference or a number.");
  fillColor.initializeFromJSONObj(mitr->value, _dataPtr);

  if (_ctx->doHitTest()) {
    if ((mitr = propObj.FindMember("id")) != propObj.MemberEnd()) {
      RUNTIME_EX_ASSERT(
          mitr->value.IsObject(),
          "JSON parse error - \"id\" is a special mark property that must be defined by a data reference.");
      id.initializeFromJSONObj(mitr->value, _dataPtr);
    } else if (_dataPtr != nullptr) {
      id.initializeFromData(DataTable::defaultIdColumnName, _dataPtr);
    } else {
      id.initializeValue(1);  // reaching here "should" guarantee that there's only
                              // 1 row of data
    }
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
  }
}

void PointMark::_initShader() {
  // TODO: need to determine a build-appropriate way to access
  // shaders
  // static const std::string pointVertexShaderFilename = "shaders/pointTemplate.vert";
  // static const std::string pointFragmentShaderFilename = "shaders/pointTemplate.frag";

  // std::string vertSrc = getShaderCodeFromFile(pointVertexShaderFilename);
  std::string vertSrc(PointTemplate_Vert::source);

  std::vector<BaseRenderProperty*> props = {&x, &y, &size, &fillColor};  // TODO: add z & fillColor

  bool useKey = key.hasVboPtr();
  boost::replace_first(vertSrc, "<useKey>", std::to_string(useKey));

  if (_ctx->doHitTest()) {
    props.push_back(&id);
  }

  // update all the types first
  for (auto prop : props) {
    setRenderPropertyTypeInShaderSrc(*prop, vertSrc);
  }

  // now insert any additional functionality
  std::unordered_map<std::string, BaseScale*> visitedScales;
  std::unordered_map<std::string, BaseScale*>::iterator itr;

  std::string funcName;
  std::string propFuncName;
  str_itr_range funcRange;

  for (auto prop : props) {
    ScaleShPtr& scalePtr = prop->getScaleConfig();
    if (scalePtr != nullptr) {
      propFuncName = prop->getGLSLFunc();

      if ((itr = visitedScales.find(scalePtr->name)) == visitedScales.end()) {
        visitedScales.insert(std::make_pair(scalePtr->name, scalePtr.get()));

        std::string scaleCode = scalePtr->getGLSLCode();

        funcRange = getGLSLFunctionBounds(vertSrc, propFuncName);

        RUNTIME_EX_ASSERT(!funcRange.empty(),
                          "Cannot find a properly defined \"" + propFuncName + "\" function in the vertex shader.");

        boost::replace_range(vertSrc, funcRange, scaleCode);

        // vertSrc.insert(funcInsertPt, scaleCode);
        // funcInsertPt += scaleCode.length();

        funcName = scalePtr->getScaleGLSLFuncName();
      } else {
        funcName = itr->second->getScaleGLSLFuncName();
      }

      // using replace_last to speed up the replace as there should only be 1 occurance
      // and it'll be toward the end of the src string.
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
}

void PointMark::_initPropertiesForRendering(Shader* activeShader) {
  RUNTIME_EX_ASSERT(!key.hasVboPtr() || key.size() == x.size(),
                    "Invalid point mark. The size of the vertex buffer's \"key\" and \"x\" sizes do not match.");
  RUNTIME_EX_ASSERT(x.size() == y.size() && x.size() == size.size() && x.size() == fillColor.size() &&
                        (!_ctx->doHitTest() || x.size() == id.size()),
                    "Invalid point mark. Not all of the sizes for the vertex attributes match.");

  // TODO(croot): only bind key if found? also, move this into base class?
  if (key.hasVboPtr()) {
    key.bindToRenderer(activeShader);
  }

  x.bindToRenderer(activeShader);
  y.bindToRenderer(activeShader);
  // z.bindToRenderer(activeShader);
  size.bindToRenderer(activeShader);
  fillColor.bindToRenderer(activeShader);

  if (_ctx->doHitTest()) {
    id.bindToRenderer(activeShader);
  }
}

void PointMark::_bindPropertiesToRenderer(Shader* activeShader) {
  std::unordered_map<std::string, BaseScale*> visitedScales;
  std::vector<BaseRenderProperty*> props = {&x, &y, &size, &fillColor};  // TODO: add z & fillColor

  // TODO(croot): create a static invalidKeyAttrName string on the class
  static const std::string invalidKeyAttrName = "invalidKey";
  if (key.hasVboPtr()) {
    if (activeShader->hasUniformAttribute(invalidKeyAttrName)) {
      GLint type = activeShader->getUniformAttributeGLType(invalidKeyAttrName);
      if (type == GL_INT) {
        activeShader->setUniformAttribute<int>(invalidKeyAttrName, static_cast<int>(_invalidKey));
      } else if (GLEW_NV_vertex_attrib_integer_64bit && type == GL_INT64_NV) {
        // TODO(croot) - do we need to do the glew extension check above or
        // would there be an error at shader compilation if the extension
        // didn't exist?

        // TODO(croot) fill this out
      }
    }
  }

  for (auto prop : props) {
    ScaleShPtr& scalePtr = prop->getScaleConfig();
    if (scalePtr != nullptr) {
      if (visitedScales.find(scalePtr->name) == visitedScales.end()) {
        visitedScales.insert(std::make_pair(scalePtr->name, scalePtr.get()));
        scalePtr->bindUniformsToRenderer(activeShader);
      }
    }
  }
}

void PointMark::draw() {
  _shaderPtr->bindToRenderer();

  _bindToRenderer(_shaderPtr.get());

  // TODO: render state stack -- push/pop
  glEnable(GL_PROGRAM_POINT_SIZE);

  // now draw points
  // TODO: What about the possibility of index buffers?
  // Probably don't have to worry about them here since
  // we're specifically looking at a point config that
  // is deemed not to have any index buffers, but we
  // need to have a way to generically draw bound buffers
  // which would be best with a renderer class
  glDrawArrays(GL_POINTS, 0, x.size());

  // unset state
  glDisable(GL_PROGRAM_POINT_SIZE);
}

GeomConfigShPtr MapD_Renderer::createMark(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - marks must be objects.");

  rapidjson::Value::ConstMemberIterator itr;
  RUNTIME_EX_ASSERT((itr = obj.FindMember("type")) != obj.MemberEnd() && itr->value.IsString(),
                    "JSON parse error - a mark object must have a \"type\" string property.");

  std::string strGeomType(itr->value.GetString());

  if (strGeomType == "points") {
    return GeomConfigShPtr(new PointMark(obj, ctx));
  } else {
    THROW_RUNTIME_EX("JSON parse error - a mark of type \"" + strGeomType + "\" is unsupported.");
  }

  return GeomConfigShPtr(nullptr);
}
