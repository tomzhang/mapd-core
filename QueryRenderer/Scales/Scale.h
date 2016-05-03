#ifndef QUERYRENDERER_SCALES_SCALE_H_
#define QUERYRENDERER_SCALES_SCALE_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "ScaleDomainRangeData.h"

#include <Rendering/Renderer/GL/TypeGL.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>

#include <string>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/regex.hpp>

namespace QueryRenderer {

class BaseScale {
 public:
  const static std::vector<std::string> scaleVertexShaderSource;

  // QueryDataType domainType;
  // QueryDataType rangeType;

  BaseScale(const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);
  BaseScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);

  virtual ~BaseScale();

  std::string getName() { return _name; }
  const std::string& getNameRef() const { return _name; }
  ScaleType getType() { return _type; }

  QueryDataType getDomainDataType() { return _domainDataType; }
  QueryDataType getRangeDataType() { return _rangeDataType; }

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();

  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");

  virtual std::string getGLSLCode(const std::string& extraSuffix = "",
                                  bool ignoreDomain = false,
                                  bool ignoreRange = false) = 0;

  std::string getDomainGLSLUniformName() { return "uDomains_" + _name; }

  std::string getRangeGLSLUniformName() { return "uRanges_" + _name; }

  std::string getRangeDefaultGLSLUniformName() { return "uDefault_" + _name; }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false) = 0;

  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  virtual bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

  bool hasClampChanged() { return _clampChanged; }
  bool hasDomainChangedInSize() { return _domainSizeChanged; }
  bool hasDomainValsChanged() { return _domainValsChanged; }
  bool hasDomainDataChanged() { return _domainSizeChanged || _domainValsChanged; }
  bool hasRangeChangedInSize() { return _rangeSizeChanged; }
  bool hasRangeValsChanged() { return _rangeValsChanged; }
  bool hasRangeDataChanged() { return _rangeSizeChanged || _rangeValsChanged; }

  virtual operator std::string() const = 0;

 protected:
  std::string _name;
  ScaleType _type;

  bool _useClamp;
  bool _clampChanged;

  // TODO(croot): somehow consolidate all the types and use typeid() or the like
  // to handle type-ness.
  QueryDataType _domainDataType;
  ::Rendering::GL::TypeGLShPtr _domainTypeGL;
  bool _domainSizeChanged;
  bool _domainValsChanged;

  QueryDataType _rangeDataType;
  ::Rendering::GL::TypeGLShPtr _rangeTypeGL;
  bool _rangeSizeChanged;
  bool _rangeValsChanged;

  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _jsonPath;

  bool _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  std::string _printInfo() const;

 private:
};

template <typename DomainType, typename RangeType>
class Scale : public BaseScale {
 public:
  Scale(const rapidjson::Value& obj,
        const rapidjson::Pointer& objPath,
        const QueryRendererContextShPtr& ctx,
        const std::string& name = "",
        ScaleType type = ScaleType::UNDEFINED)
      : BaseScale(obj, objPath, ctx, getDataTypeForType<DomainType>(), getDataTypeForType<RangeType>(), name, type),
        _domainPtr("domain", false),
        _rangePtr("range", true),
        _defaultVal() {
    _initGLTypes();
    updateFromJSONObj(obj, objPath);
  }

  ~Scale() {}

  std::string getGLSLCode(const std::string& extraSuffix = "", bool ignoreDomain = false, bool ignoreRange = false) {
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

    // TODO(croot): create a derived class per scale type?
    if (_type == ScaleType::LINEAR) {
      ss.str("");
      ss << _useClamp;
      boost::replace_all(shaderCode, "<useClamp>", ss.str());
    }

    shaderCode += '\n';

    return shaderCode;
  }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false) {
    if (!ignoreDomain) {
      activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorData());
    }

    if (!ignoreRange) {
      activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorData());
    }

    // TODO(croot): create a derived class per scale type?
    if (_type == ScaleType::ORDINAL) {
      activeShader->setUniformAttribute(getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
    } else if (_type == ScaleType::QUANTIZE) {
      double diff = _domainPtr.getDifference(_rangePtr.size());
      activeShader->setUniformAttribute<double>("quantizeDiff", diff);
    }
  }

  BaseScaleDomainRangeData* getDomainData() { return &_domainPtr; };
  BaseScaleDomainRangeData* getRangeData() { return &_rangePtr; };

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
    bool rtn = BaseScale::_initFromJSONObj(obj, objPath);

    if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
      _domainPtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _domainSizeChanged, _domainValsChanged);
      _rangePtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _rangeSizeChanged, _rangeValsChanged);

      if (_type == ScaleType::ORDINAL) {
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

  operator std::string() const final {
    return "Scale<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + ">" +
           _printInfo();
  }

 private:
  // std::vector<DomainType> _domain;
  // std::vector<RangeType> _range;

  // std::shared_ptr<std::vector<DomainType>> _domainPtr;
  // std::shared_ptr<std::vector<RangeType>> _rangePtr;

  ScaleDomainRangeData<DomainType> _domainPtr;
  ScaleDomainRangeData<RangeType> _rangePtr;

  RangeType _defaultVal;
  rapidjson::Pointer _defaultJsonPath;

  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initGLTypes() {
    _domainTypeGL = _domainPtr.getTypeGL();
    _rangeTypeGL = _rangePtr.getTypeGL();
  }

  void _setDefaultFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
    // TODO(croot): make different derived scale classes per scale type
    // and move this function only into the "OrdinalScale" derived class
    rapidjson::Value::ConstMemberIterator mitr;

    // TODO(croot): expose "default" as a constant somewhere;
    std::string defaultStr = "default";

    if (!_ctx->isJSONCacheUpToDate(_defaultJsonPath, obj)) {
      if ((mitr = obj.FindMember(defaultStr.c_str())) != obj.MemberEnd()) {
        QueryDataType itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value);

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

  void _updateDefaultJSONPath(const rapidjson::Pointer& objPath) {
    // TODO(croot): expose "default" as a constant somewhere;
    std::string defaultStr = "default";
    _defaultJsonPath = objPath.Append(defaultStr.c_str(), defaultStr.length());
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALE_H_
