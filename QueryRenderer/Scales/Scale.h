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

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false) = 0;

  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  virtual bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

  virtual bool hasPropertiesChanged() { return false; }
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
        _rangePtr("range", true) {
    _initGLTypes();
  }

  virtual ~Scale() {}

  BaseScaleDomainRangeData* getDomainData() { return &_domainPtr; };
  BaseScaleDomainRangeData* getRangeData() { return &_rangePtr; };

 protected:
  std::string _getGLSLCode(const std::string& extraSuffix, bool ignoreDomain, bool ignoreRange) {
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

    shaderCode += '\n';

    return shaderCode;
  }

  bool _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
    bool rtn = false;

    if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
      RUNTIME_EX_ASSERT(
          obj.IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "scale items must be objects."));

      _domainPtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _domainSizeChanged, _domainValsChanged);
      _rangePtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _rangeSizeChanged, _rangeValsChanged);

      rtn = hasDomainDataChanged() || hasRangeDataChanged();
    } else if (_jsonPath != objPath) {
      _domainSizeChanged = _domainValsChanged = _rangeSizeChanged = _rangeValsChanged = true;

      _domainPtr.updateJSONPath(objPath);
      _rangePtr.updateJSONPath(objPath);
    }

    return rtn;
  }

  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                               const std::string& extraSuffix,
                               bool ignoreDomain,
                               bool ignoreRange) {
    if (!ignoreDomain) {
      activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorData());
    }

    if (!ignoreRange) {
      activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorData());
    }
  }

  std::string _printInfo() const {
    return "<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + ">" +
           BaseScale::_printInfo();
  }

  ScaleDomainRangeData<DomainType> _domainPtr;
  ScaleDomainRangeData<RangeType> _rangePtr;

 private:
  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initGLTypes() {
    _domainTypeGL = _domainPtr.getTypeGL();
    _rangeTypeGL = _rangePtr.getTypeGL();
  }
};

template <typename DomainType, typename RangeType>
class LinearScale : public Scale<DomainType, RangeType> {
 public:
  LinearScale(const rapidjson::Value& obj,
              const rapidjson::Pointer& objPath,
              const QueryRendererContextShPtr& ctx,
              const std::string& name = "")
      : Scale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::LINEAR), _useClamp(false) {
    updateFromJSONObj(obj, objPath);
  }

  ~LinearScale() {}

  bool hasPropertiesChanged() final { return hasClampChanged(); }
  bool hasClampChanged() { return _clampChanged; }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false) final {
    this->_bindUniformsToRenderer(activeShader, extraSuffix, ignoreDomain, ignoreRange);
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false) final {
    std::string shaderCode = this->_getGLSLCode(extraSuffix, ignoreDomain, ignoreRange);

    std::ostringstream ss;
    ss << _useClamp;
    boost::replace_all(shaderCode, "<useClamp>", ss.str());

    return shaderCode;
  }

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool updated = this->_updateFromJSONObj(obj, objPath);

    if (updated) {
      rapidjson::Value::ConstMemberIterator itr;

      // TODO(croot): move the "clamp" prop name into a const somewhere.
      std::string clampProp = "clamp";

      bool prevClamp = _useClamp;
      if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(
            itr->value.IsBool(),
            RapidJSONUtils::getJsonParseErrorStr(
                this->_ctx->getUserWidgetIds(), obj, "the \"clamp\" property for linear scales must be a boolean."));

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

    this->_jsonPath = objPath;

    return updated;
  }

  operator std::string() const final { return "LinearScale" + this->_printInfo(); }

 private:
  bool _useClamp;
  bool _clampChanged;
};

template <typename DomainType, typename RangeType>
class OrdinalScale : public Scale<DomainType, RangeType> {
 public:
  OrdinalScale(const rapidjson::Value& obj,
               const rapidjson::Pointer& objPath,
               const QueryRendererContextShPtr& ctx,
               const std::string& name = "")
      : Scale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::ORDINAL), _defaultVal() {
    updateFromJSONObj(obj, objPath);
  }

  ~OrdinalScale() {}

  std::string getRangeDefaultGLSLUniformName() { return "uDefault_" + this->_name; }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false) final {
    this->_bindUniformsToRenderer(activeShader, extraSuffix, ignoreDomain, ignoreRange);
    activeShader->setUniformAttribute(this->getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false) final {
    std::string shaderCode = this->_getGLSLCode(extraSuffix, ignoreDomain, ignoreRange);

    return shaderCode;
  }

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool rtn = this->_updateFromJSONObj(obj, objPath);

    if (rtn) {
      if (!this->_ctx->isJSONCacheUpToDate(_defaultJsonPath, obj)) {
        // TODO(croot): expose "default" as a constant somewhere;
        std::string defaultStr = "default";
        rapidjson::Value::ConstMemberIterator mitr;

        if ((mitr = obj.FindMember(defaultStr.c_str())) != obj.MemberEnd()) {
          QueryDataType itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value);

          RUNTIME_EX_ASSERT(itemType == this->_rangePtr.dataType,
                            RapidJSONUtils::getJsonParseErrorStr(
                                this->_ctx->getUserWidgetIds(),
                                mitr->value,
                                "scale \"" + this->_name + "\" default is not the same type as its range."));

          _defaultVal = this->_rangePtr.getDataValueFromJSONObj(mitr->value);
        } else {
          // set an undefined default
          _defaultVal = RangeType();
        }
      }
    }

    this->_jsonPath = objPath;
    _updateDefaultJSONPath(objPath);

    return rtn;
  }

  operator std::string() const final { return "OrdinalScale" + this->_printInfo(); }

 private:
  RangeType _defaultVal;
  rapidjson::Pointer _defaultJsonPath;

  void _updateDefaultJSONPath(const rapidjson::Pointer& objPath) {
    // TODO(croot): expose "default" as a constant somewhere;
    std::string defaultStr = "default";
    _defaultJsonPath = objPath.Append(defaultStr.c_str(), defaultStr.length());
  }
};

template <typename DomainType, typename RangeType>
class QuantizeScale : public Scale<DomainType, RangeType> {
 public:
  QuantizeScale(const rapidjson::Value& obj,
                const rapidjson::Pointer& objPath,
                const QueryRendererContextShPtr& ctx,
                const std::string& name = "")
      : Scale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::QUANTIZE) {
    updateFromJSONObj(obj, objPath);
  }

  ~QuantizeScale() {}

  // bool hasPropertiesChanged() final { return hasClampChanged(); }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false) final {
    this->_bindUniformsToRenderer(activeShader, extraSuffix, ignoreDomain, ignoreRange);

    double diff = this->_domainPtr.getDifference(this->_rangePtr.size());
    activeShader->setUniformAttribute<double>("quantizeDiff", diff);
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false) final {
    std::string shaderCode = this->_getGLSLCode(extraSuffix, ignoreDomain, ignoreRange);
    return shaderCode;
  }

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool rtn = this->_updateFromJSONObj(obj, objPath);
    this->_jsonPath = objPath;

    return rtn;
  }

  operator std::string() const final { return "QuantizeScale" + this->_printInfo(); }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALE_H_
