#ifndef QUERYRENDERER_SCALES_SCALE_H_
#define QUERYRENDERER_SCALES_SCALE_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "ScaleDomainRangeData.h"

#include "../PerGpuData.h"
#include "../Utils/Utils.h"

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
  static const size_t maxAccumTextures;
  static size_t convertNumAccumValsToNumAccumTextures(size_t numAccumVals);
  static size_t convertNumAccumTexturesToNumAccumVals(size_t numAccumTxts);

  enum class AccumulatorType { MIN = 0, MAX, BLEND, UNDEFINED };

  const static std::vector<std::string> scaleVertexShaderSource;

  BaseScale(const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED,
            bool allowsAccumulator = false);
  BaseScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED,
            bool allowsAccumulator = false);

  virtual ~BaseScale();

  std::string getName() { return _name; }
  const std::string& getNameRef() const { return _name; }
  ScaleType getType() { return _type; }
  AccumulatorType getAccumulatorType() const { return _accumType; }
  bool hasAccumulator() const { return _accumType != AccumulatorType::UNDEFINED; }
  int getNumAccumulatorValues() const { return _numAccumulatorVals; }
  size_t getNumAccumulatorTextures() const { return convertNumAccumValsToNumAccumTextures(_numAccumulatorVals); }

  QueryDataType getDomainDataType() { return _domainDataType; }
  QueryDataType getRangeDataType() { return _rangeDataType; }

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();

  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");
  std::string getAccumulator1stPassFragSrc() const;

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

  virtual bool hasPropertiesChanged() const { return false; }
  bool hasDomainChangedInSize() const { return _domainSizeChanged; }
  bool hasDomainValsChanged() const { return _domainValsChanged; }
  bool hasDomainDataChanged() const { return _domainSizeChanged || _domainValsChanged; }
  bool hasRangeChangedInSize() const { return _rangeSizeChanged; }
  bool hasRangeValsChanged() const { return _rangeValsChanged; }
  bool hasRangeDataChanged() const { return _rangeSizeChanged || _rangeValsChanged; }
  bool hasAccumulatorTypeChanged() const { return _accumTypeChanged; }
  bool hasNumAccumulatorTexturesChanged() const { return _numAccumulatorTxtsChanged; }

  virtual operator std::string() const = 0;

  void accumulationPreRender(const GpuId& gpuId);
  void accumulationPostRender(const GpuId& gpuId);
  void renderAccumulation(::Rendering::GL::GLRenderer* glRenderer,
                          const GpuId& gpuId,
                          const ::Rendering::GL::Resources::GLTexture2dShPtr& idTex);

  const std::vector<::Rendering::GL::Resources::GLTexture2dShPtr>& getAccumulatorTextureArrayRef(const GpuId& gpuId);

  virtual void bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                                     const std::string& attrName,
                                     bool checkFullSize = true);

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

  bool _allowsAccumulator;

  std::string _printInfo() const;

  bool _updateAccumulatorFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _setNumAccumulatorVals(int numAccumulatorVals);
  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader);

  bool hasNumAccumulatorValsChanged() const { return _numAccumulatorValsChanged; }

 private:
  struct PerGpuData : BasePerGpuData {
    std::vector<::Rendering::GL::Resources::GLTexture2dShPtr> accumulatorTexPtrArray;

    // TODO(croot): can these all be cached somehow?
    ::Rendering::GL::Resources::GLShaderShPtr accumulator2ndPassShaderPtr;
    ::Rendering::GL::Resources::GLVertexBufferShPtr rectvbo;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vao;

    PerGpuData() : BasePerGpuData() {}
    explicit PerGpuData(const BasePerGpuData& data,
                        // const ::Rendering::GL::Resources::GLPixelBuffer2dShPtr& clearPboPtr = nullptr,
                        const ::Rendering::GL::Resources::GLShaderShPtr& accumulator2ndPassShaderPtr = nullptr,
                        const ::Rendering::GL::Resources::GLVertexBufferShPtr& rectvbo = nullptr,
                        const ::Rendering::GL::Resources::GLVertexArrayShPtr& vao = nullptr)
        : BasePerGpuData(data),
          // clearPboPtr(clearPboPtr),
          accumulator2ndPassShaderPtr(accumulator2ndPassShaderPtr),
          rectvbo(rectvbo),
          vao(vao) {}

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;
  AccumulatorType _accumType;
  bool _accumTypeChanged;

  int _numAccumulatorVals;
  bool _numAccumulatorValsChanged;
  bool _numAccumulatorTxtsChanged;

  // using this bool to determine whether the gpu resources for an accumulator
  // scale were just built in the current render configuration update.
  // _initGpuResources() can be called from 2 locations, in BaseScaleRef and
  // QueryRendererContext classes. The former builds the gpu resources because
  // we know at that point whether a scale is used by a mark or not - we don't
  // want to build the resources if the scale isn't used by a mark anywhere.
  // QueryRendererContext updates the gpu resources (rebuilds shaders or
  // resizes). _justBuilt is a flag used solely for saying whether the
  // resources were build in the same update loop to prevent possible
  // duplication of effort.
  bool _justBuilt;

  void _initGpuResources(QueryRendererContext* ctx, bool initializing);
  ::Rendering::GL::Resources::GLShaderShPtr _buildAccumulatorShader(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
                                                                    size_t numTextures);

  void _clearResources();

  friend class QueryRendererContext;
  friend class BaseScaleRef;
};

template <typename DomainType, typename RangeType>
class Scale : public BaseScale {
 public:
  Scale(const rapidjson::Value& obj,
        const rapidjson::Pointer& objPath,
        const QueryRendererContextShPtr& ctx,
        const std::string& name = "",
        ScaleType type = ScaleType::UNDEFINED,
        bool allowsAccumulator = false)
      : BaseScale(obj,
                  objPath,
                  ctx,
                  getDataTypeForType<DomainType>(),
                  getDataTypeForType<RangeType>(),
                  name,
                  type,
                  allowsAccumulator),
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

    boost::replace_all(shaderCode, "<doAccum>", std::to_string(hasAccumulator()));

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

      if (_allowsAccumulator) {
        bool accumUpdated = _updateAccumulatorFromJSONObj(obj, objPath);
        rtn = rtn || accumUpdated;
      }
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
      activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorDataRef());
    }

    if (!ignoreRange) {
      activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorDataRef());
    }

    BaseScale::_bindUniformsToRenderer(activeShader);
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

  bool hasPropertiesChanged() const final { return hasClampChanged(); }
  bool hasClampChanged() const { return _clampChanged; }

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
      : Scale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::ORDINAL, true), _defaultVal() {
    updateFromJSONObj(obj, objPath);
  }

  ~OrdinalScale() {}

  std::string getRangeDefaultGLSLUniformName() { return "uDefault_" + this->_name; }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false) final {
    if (this->hasAccumulator()) {
      this->_bindUniformsToRenderer(activeShader, extraSuffix, ignoreDomain, true);
    } else {
      this->_bindUniformsToRenderer(activeShader, extraSuffix, ignoreDomain, ignoreRange);
      activeShader->setUniformAttribute(this->getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
    }
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

    // add an additional accumulator for "defaults"
    if (this->hasAccumulator()) {
      this->_setNumAccumulatorVals(this->_rangePtr.size() + 1);
    }

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

  void bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                             const std::string& attrName,
                             bool checkFullSize = true) final {
    RUNTIME_EX_ASSERT(this->_rangePtr.getType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation types.");

    auto data = this->_rangePtr.getVectorData();
    data.push_back(_defaultVal);
    CHECK(static_cast<int>(data.size()) == this->getNumAccumulatorValues());
    shaderPtr->setUniformAttribute(attrName, data, checkFullSize);
  }
};

template <typename DomainType, typename RangeType>
class QuantizeScale : public Scale<DomainType, RangeType> {
 public:
  QuantizeScale(const rapidjson::Value& obj,
                const rapidjson::Pointer& objPath,
                const QueryRendererContextShPtr& ctx,
                const std::string& name = "")
      : Scale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::QUANTIZE, true) {
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

    if (this->hasAccumulator()) {
      this->_setNumAccumulatorVals(this->_rangePtr.size());
    }

    return rtn;
  }

  operator std::string() const final { return "QuantizeScale" + this->_printInfo(); }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALE_H_
