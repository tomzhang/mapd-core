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
#include <boost/any.hpp>

namespace QueryRenderer {

class BaseScale {
 public:
  static const size_t maxAccumTextures;
  static size_t convertNumAccumValsToNumAccumTextures(size_t numAccumVals, AccumulatorType accumType);
  static size_t convertNumAccumTexturesToNumAccumVals(size_t numAccumTxts, AccumulatorType accumType);

  enum class ScaleShaderType { QUANTITATIVE = 0, ORDINAL, QUANTIZE };
  const static std::vector<std::string> scaleVertexShaderSource;

  BaseScale(const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED,
            bool allowsAccumulator = false,
            uint8_t accumTypeMask = static_cast<uint8_t>(AccumulatorType::ALL));
  BaseScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED,
            bool allowsAccumulator = false,
            uint8_t accumTypeMask = static_cast<uint8_t>(AccumulatorType::ALL));

  virtual ~BaseScale();

  std::string getName() { return _name; }
  const std::string& getNameRef() const { return _name; }
  ScaleType getType() { return _type; }
  AccumulatorType getAccumulatorType() const { return _accumType; }
  bool hasAccumulator() const { return _accumType != AccumulatorType::UNDEFINED; }
  int getNumAccumulatorValues() const { return _numAccumulatorVals; }
  size_t getNumAccumulatorTextures() const {
    return convertNumAccumValsToNumAccumTextures(_numAccumulatorVals, _accumType);
  }

  QueryDataType getDomainDataType() { return _domainDataType; }
  QueryDataType getRangeDataType() { return _rangeDataType; }

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();

  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");
  std::string getAccumulator1stPassFragSrc() const;

  virtual std::string getGLSLCode(const std::string& extraSuffix = "",
                                  bool ignoreDomain = false,
                                  bool ignoreRange = false,
                                  bool ignoreAccum = false) = 0;

  std::string getDomainGLSLUniformName() { return "uDomains_" + _name; }

  std::string getRangeGLSLUniformName() { return "uRanges_" + _name; }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      std::unordered_map<std::string, std::string>& subroutineMap,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false,
                                      bool ignoreAccum = false) = 0;

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

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix) const = 0;
  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getRangeTypeUniforms(
      const std::string& extraSuffix) const = 0;

  virtual operator std::string() const = 0;

  void accumulationPreRender(const GpuId& gpuId);
  void accumulationPostRender(const GpuId& gpuId);
  void renderAccumulation(::Rendering::GL::GLRenderer* glRenderer,
                          const GpuId& gpuId,
                          ::Rendering::GL::Resources::GLTexture2dArray* compTxArrayPtr = nullptr);

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
  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                               std::unordered_map<std::string, std::string>& subroutineMap,
                               bool ignoreAccum);

  bool hasNumAccumulatorValsChanged() const { return _numAccumulatorValsChanged; }

 private:
  struct PerGpuData : BasePerGpuData {
    std::vector<::Rendering::GL::Resources::GLTexture2dShPtr> accumulatorTexPtrArray;

    ::Rendering::GL::Resources::GLShaderShPtr accumulator2ndPassShaderPtr;
    ::Rendering::GL::Resources::GLVertexBufferShPtr rectvbo;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vao;

    PerGpuData() : BasePerGpuData() {}
    explicit PerGpuData(const BasePerGpuData& data,
                        // const ::Rendering::GL::Resources::GLPixelBuffer2dShPtr& clearPboPtr = nullptr,
                        const ::Rendering::GL::Resources::GLShaderShPtr& accumulator2ndPassShaderPtr = nullptr,
                        const ::Rendering::GL::Resources::GLVertexBufferShPtr& rectvbo = nullptr,
                        const ::Rendering::GL::Resources::GLVertexArrayShPtr& vao = nullptr)
        : BasePerGpuData(data), accumulator2ndPassShaderPtr(accumulator2ndPassShaderPtr), rectvbo(rectvbo), vao(vao) {}

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;
  AccumulatorType _accumType;
  bool _accumTypeChanged;
  int16_t _accumTypeMask;

  int _numAccumulatorVals;
  bool _numAccumulatorValsChanged;
  bool _numAccumulatorTxtsChanged;

  uint8_t _numMinStdDev;
  uint32_t _minDensity;
  bool _findMinDensity;

  uint8_t _numMaxStdDev;
  uint32_t _maxDensity;
  bool _findMaxDensity;
  bool _findStdDev;

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
        bool allowsAccumulator = false,
        uint8_t accumTypeMask = static_cast<uint8_t>(AccumulatorType::ALL))
      : BaseScale(obj,
                  objPath,
                  ctx,
                  getDataTypeForType<DomainType>(),
                  getDataTypeForType<RangeType>(),
                  name,
                  type,
                  allowsAccumulator,
                  accumTypeMask),
        _domainPtr("domain", false),
        _rangePtr("range", true),
        _nullVal(),
        _useNullVal(false) {
    _initGLTypes();
  }

  virtual ~Scale() {}

  BaseScaleDomainRangeData* getDomainData() { return &_domainPtr; };
  BaseScaleDomainRangeData* getRangeData() { return &_rangePtr; };

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix) const {
    std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> rtn(_domainPtr.getType(), {});

    if (_useNullVal) {
      rtn.second.emplace("nullDomainVal_" + this->_name + extraSuffix, _domainPtr.getNullValue());
    }

    return rtn;
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getRangeTypeUniforms(
      const std::string& extraSuffix) const {
    std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> rtn(_rangePtr.getType(), {});

    if (_useNullVal) {
      rtn.second.emplace("nullRangeVal_" + this->_name + extraSuffix, _nullVal);
    }

    return rtn;
  }

 protected:
  std::string _getGLSLCode(const std::string& extraSuffix,
                           bool ignoreDomain,
                           bool ignoreRange,
                           bool ignoreAccum,
                           const BaseScale::ScaleShaderType scaleShader) {
    RUNTIME_EX_ASSERT(_domainPtr.size() > 0 && _rangePtr.size() > 0,
                      std::string(*this) + " getGLSLCode(): domain/range of scale \"" + _name + "\" has no value.");

    std::string shaderCode = scaleVertexShaderSource[static_cast<int>(scaleShader)];
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

    bool hasAccum = (ignoreAccum ? false : hasAccumulator());
    boost::replace_all(shaderCode, "<doAccum>", std::to_string(hasAccum));

    shaderCode += '\n';

    return shaderCode;
  }

  bool _updateFromJSONObj(const rapidjson::Value& obj,
                          const rapidjson::Pointer& objPath,
                          std::function<void(DomainType&)> domainValConvert = [](DomainType& domainVal) { return; },
                          std::function<void(RangeType&)> rangeValConvert = [](RangeType& rangeVal) { return; }) {
    bool rtn = false;

    if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
      RUNTIME_EX_ASSERT(
          obj.IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "scale items must be objects."));

      _domainPtr.initializeFromJSONObj(
          obj, objPath, _ctx, _type, _domainSizeChanged, _domainValsChanged, domainValConvert);
      _rangePtr.initializeFromJSONObj(obj, objPath, _ctx, _type, _rangeSizeChanged, _rangeValsChanged, rangeValConvert);

      rtn = hasDomainDataChanged() || hasRangeDataChanged();

      // TODO(croot): expose "nullValue" as a constant somewhere;
      std::string nullstr = "nullValue";
      rapidjson::Value::ConstMemberIterator mitr;

      if ((mitr = obj.FindMember(nullstr.c_str())) != obj.MemberEnd()) {
        QueryDataType itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value);

        RUNTIME_EX_ASSERT(
            areTypesCompatible(_rangePtr.getType(), itemType),
            RapidJSONUtils::getJsonParseErrorStr(
                this->_ctx->getUserWidgetIds(),
                mitr->value,
                "the scale \"" + this->_name + "\" has a range of type " + to_string(_rangePtr.getType()) +
                    " which is not compatible with a nullValue of type " + to_string(itemType) + "."));

        _nullVal = this->_rangePtr.getDataValueFromJSONObj(mitr->value);
        _useNullVal = true;
      } else {
        // set an undefined default
        _nullVal = RangeType();
        _useNullVal = false;
      }

      if (_allowsAccumulator) {
        bool accumUpdated = _updateAccumulatorFromJSONObj(obj, objPath);

        // TODO(croot): Properly handle null values in accumulations.
        // This would require adding the null value to the accumulation arrays
        // for ordinal and quantize scales. If we properly handle that case,
        // then remove the following LOG_IF warning.
        LOG_IF(WARNING, accumUpdated && hasAccumulator() && _useNullVal)
            << "Render scale " << std::string(*this)
            << ", accumulation scale with null values defined may have unwanted results";

        rtn = rtn || accumUpdated;

        QueryDataType dtype = getDomainDataType();
        RUNTIME_EX_ASSERT(
            getAccumulatorType() != AccumulatorType::DENSITY || dtype == QueryDataType::FLOAT ||
                dtype == QueryDataType::DOUBLE,
            RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                 obj,
                                                 "density accumulator scales must have floats/doubles as its domain "
                                                 "values which are used as percentages of the final accumulation "
                                                 "counts."));
      }
    } else if (_jsonPath != objPath) {
      _domainSizeChanged = _domainValsChanged = _rangeSizeChanged = _rangeValsChanged = true;

      _domainPtr.updateJSONPath(objPath);
      _rangePtr.updateJSONPath(objPath);
    }

    return rtn;
  }

  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                               std::unordered_map<std::string, std::string>& subroutineMap,
                               const std::string& extraSuffix,
                               bool ignoreDomain,
                               bool ignoreRange,
                               bool ignoreAccum,
                               bool ignoreNull = false) {
    if (!ignoreDomain) {
      activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorDataRef());

      if (_useNullVal) {
        activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix, _domainPtr.getNullValue());
      }
    }

    if (!ignoreRange) {
      activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorDataRef());

      if (_useNullVal) {
        activeShader->setUniformAttribute("nullRangeVal_" + this->_name + extraSuffix, _nullVal);
      }
    }

    // NOTE: when rendering the final accumulation pass, the null val func
    // can be turned off.
    // TODO(croot): Should we find a better way to deactive null vals for
    // accumulation passes?
    if (!ignoreNull) {
      if (_useNullVal && !ignoreAccum) {
        subroutineMap["isNullValFunc_" + this->_name + extraSuffix] = "isNullVal_" + this->_name + extraSuffix;
      } else {
        subroutineMap["isNullValFunc_" + this->_name + extraSuffix] = "isNullValPassThru_" + this->_name + extraSuffix;
      }
    }

    BaseScale::_bindUniformsToRenderer(activeShader, subroutineMap, ignoreAccum);
  }

  std::string _printInfo() const {
    return "<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + ">" +
           BaseScale::_printInfo();
  }

  ScaleDomainRangeData<DomainType> _domainPtr;
  ScaleDomainRangeData<RangeType> _rangePtr;

 private:
  RangeType _nullVal;
  bool _useNullVal;

  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initGLTypes() {
    _domainTypeGL = _domainPtr.getTypeGL();
    _rangeTypeGL = _rangePtr.getTypeGL();
  }
};

template <typename DomainType, typename RangeType>
class QuantitativeScale : public Scale<DomainType, RangeType> {
 public:
  QuantitativeScale(const rapidjson::Value& obj,
                    const rapidjson::Pointer& objPath,
                    const QueryRendererContextShPtr& ctx,
                    const std::string& name,
                    const ScaleType scaleType)
      : Scale<DomainType,
              RangeType>(obj, objPath, ctx, name, scaleType, true, static_cast<uint8_t>(AccumulatorType::DENSITY)),
        _useClamp(false) {}

  virtual ~QuantitativeScale() {}

  bool hasPropertiesChanged() const final { return hasClampChanged(); }
  bool hasClampChanged() const { return _clampChanged; }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      std::unordered_map<std::string, std::string>& subroutineMap,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false,
                                      bool ignoreAccum = false) {
    bool doAccum = false;
    if (!ignoreAccum && this->hasAccumulator()) {
      ignoreDomain = true;
      ignoreRange = true;
      doAccum = true;
    }
    this->_bindUniformsToRenderer(
        activeShader, subroutineMap, extraSuffix, ignoreDomain, ignoreRange, ignoreAccum, doAccum);

    if (!doAccum) {
      std::string transformFunc;
      switch (this->_type) {
        case ScaleType::LINEAR:
          transformFunc = "passThruTransform";
          break;
        case ScaleType::LOG:
          transformFunc = "logTransform";
          break;
        case ScaleType::POW:
          transformFunc = "powTransform";
          break;
        case ScaleType::SQRT:
          transformFunc = "sqrtTransform";
          break;
        default:
          THROW_RUNTIME_EX("ScaleType " + to_string(this->_type) + " does not have a supported glsl transform func.");
      }

      transformFunc += "_" + this->_name + extraSuffix;

      subroutineMap["quantTransform_" + this->_name + extraSuffix] = transformFunc;
    }
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false,
                          bool ignoreAccum = false) final {
    std::string shaderCode = this->_getGLSLCode(
        extraSuffix, ignoreDomain, ignoreRange, ignoreAccum, BaseScale::ScaleShaderType::QUANTITATIVE);

    std::ostringstream ss;
    ss << _useClamp;
    boost::replace_all(shaderCode, "<useClamp>", ss.str());

    return shaderCode;
  }

  bool _updateQuantitativeFromJSONObj(const rapidjson::Value& obj,
                                      const rapidjson::Pointer& objPath,
                                      std::function<void(DomainType&)> domainValConvert = [](DomainType& domainVal) {
    return;
  }) {
    bool updated = this->_updateFromJSONObj(obj, objPath, domainValConvert);

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
        // Clamping by default makes sense for DENSITY accumulation.
        // TODO(croot): should that be a default for other types?
        _useClamp = (this->getAccumulatorType() == AccumulatorType::DENSITY ? true : false);
      }

      if (prevClamp != _useClamp) {
        _clampChanged = true;
      } else {
        _clampChanged = false;
      }
    }

    if (this->hasAccumulator()) {
      this->_setNumAccumulatorVals(this->_rangePtr.size());
    }

    this->_jsonPath = objPath;

    return updated;
  }

 private:
  bool _useClamp;
  bool _clampChanged;

  void bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                             const std::string& attrName,
                             bool checkFullSize = true) final {
    RUNTIME_EX_ASSERT(this->_rangePtr.getType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation range types.");

    auto& data = this->_rangePtr.getVectorDataRef();
    CHECK(static_cast<int>(data.size()) == this->getNumAccumulatorValues());
    shaderPtr->setUniformAttribute(attrName, data, checkFullSize);
  }
};

template <typename DomainType, typename RangeType>
class LinearScale : public QuantitativeScale<DomainType, RangeType> {
 public:
  LinearScale(const rapidjson::Value& obj,
              const rapidjson::Pointer& objPath,
              const QueryRendererContextShPtr& ctx,
              const std::string& name = "")
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::LINEAR) {
    updateFromJSONObj(obj, objPath);
  }

  ~LinearScale() {}

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool updated = this->_updateQuantitativeFromJSONObj(obj, objPath);
    this->_jsonPath = objPath;
    return updated;
  }

  operator std::string() const final { return "LinearScale" + this->_printInfo(); }
};

template <typename DomainType, typename RangeType>
class LogScale : public QuantitativeScale<DomainType, RangeType> {
 public:
  LogScale(const rapidjson::Value& obj,
           const rapidjson::Pointer& objPath,
           const QueryRendererContextShPtr& ctx,
           const std::string& name = "")
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::LOG) {
    updateFromJSONObj(obj, objPath);
  }

  ~LogScale() {}

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool updated = this->_updateQuantitativeFromJSONObj(obj, objPath, logFunc);
    this->_jsonPath = objPath;
    return updated;
  }

  operator std::string() const final { return "LogScale" + this->_printInfo(); }

 private:
  static void logFunc(DomainType& val) {
    RUNTIME_EX_ASSERT(val > 0, std::to_string(val) + " is <= 0. Log scales only work with positive values");
    val = std::log(val);
  }
};

template <typename DomainType, typename RangeType>
class PowScale : public QuantitativeScale<DomainType, RangeType> {
 public:
  PowScale(const rapidjson::Value& obj,
           const rapidjson::Pointer& objPath,
           const QueryRendererContextShPtr& ctx,
           const std::string& name = "")
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::POW), _exponent(1.0) {
    updateFromJSONObj(obj, objPath);
  }

  ~PowScale() {}

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              std::unordered_map<std::string, std::string>& subroutineMap,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false,
                              bool ignoreAccum = false) final {
    QuantitativeScale<DomainType, RangeType>::bindUniformsToRenderer(
        activeShader, subroutineMap, extraSuffix, ignoreDomain, ignoreRange, ignoreAccum);

    activeShader->setUniformAttribute("uExponent_" + this->_name + extraSuffix, _exponent);
  }

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    if (!this->_ctx->isJSONCacheUpToDate(this->_jsonPath, obj)) {
      rapidjson::Value::ConstMemberIterator itr;

      // TODO(croot): move the "exponent" prop name into a const somewhere.
      std::string expProp = "exponent";

      if ((itr = obj.FindMember(expProp.c_str())) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(
            itr->value.IsNumber(),
            RapidJSONUtils::getJsonParseErrorStr(
                this->_ctx->getUserWidgetIds(), obj, "the \"exponent\" property for pow scales must be a number."));

        _exponent = RapidJSONUtils::getNumValFromJSONObj<float>(itr->value);
      } else {
        // TODO(croot): set a const default for _exponent somewhere
        _exponent = DomainType(1);
      }
    }

    bool updated =
        this->_updateQuantitativeFromJSONObj(obj, objPath, std::bind(powFunc, std::placeholders::_1, _exponent));

    this->_jsonPath = objPath;
    return updated;
  }

  operator std::string() const final { return "PowScale" + this->_printInfo(); }

 private:
  // TODO(croot): should this be any type? Right now, it seems we can only do
  // a pow() of a float in glsl. That, and because of polymorphism issues when
  // there's a ScaleRef() object doing type coercion, it's easist to just
  // keep this as a float.
  float _exponent;

  static void powFunc(DomainType& val, float exponent) { val = std::pow(val, exponent); }
};

template <typename DomainType, typename RangeType>
class SqrtScale : public QuantitativeScale<DomainType, RangeType> {
 public:
  SqrtScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            const std::string& name = "")
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::SQRT) {
    updateFromJSONObj(obj, objPath);
  }

  ~SqrtScale() {}

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final {
    bool updated = this->_updateQuantitativeFromJSONObj(obj, objPath, sqrtFunc);
    this->_jsonPath = objPath;
    return updated;
  }

  operator std::string() const final { return "SqrtScale" + this->_printInfo(); }

 private:
  static void sqrtFunc(DomainType& val) {
    RUNTIME_EX_ASSERT(val >= 0, std::to_string(val) + " is < 0. sqrt scales only work with positive values");
    val = std::sqrt(val);
  }
};

template <typename DomainType, typename RangeType>
class OrdinalScale : public Scale<DomainType, RangeType> {
 public:
  OrdinalScale(const rapidjson::Value& obj,
               const rapidjson::Pointer& objPath,
               const QueryRendererContextShPtr& ctx,
               const std::string& name = "")
      : Scale<DomainType, RangeType>(
            obj,
            objPath,
            ctx,
            name,
            ScaleType::ORDINAL,
            true,
            (static_cast<uint8_t>(AccumulatorType::MIN) | static_cast<uint8_t>(AccumulatorType::MAX) |
             static_cast<uint8_t>(AccumulatorType::BLEND))),
        _defaultVal() {
    updateFromJSONObj(obj, objPath);
  }

  ~OrdinalScale() {}

  std::string getRangeDefaultGLSLUniformName() const { return "uDefault_" + this->_name; }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              std::unordered_map<std::string, std::string>& subroutineMap,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false,
                              bool ignoreAccum = false) final {
    if (!ignoreAccum && this->hasAccumulator()) {
      this->_bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix, ignoreDomain, true, ignoreAccum);
    } else {
      this->_bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix, ignoreDomain, ignoreRange, ignoreAccum);

      if (!ignoreRange) {
        activeShader->setUniformAttribute(this->getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);
      }
    }
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false,
                          bool ignoreAccum = false) final {
    std::string shaderCode =
        this->_getGLSLCode(extraSuffix, ignoreDomain, ignoreRange, ignoreAccum, BaseScale::ScaleShaderType::ORDINAL);

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

  std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getRangeTypeUniforms(
      const std::string& extraSuffix) const {
    auto rtn = Scale<DomainType, RangeType>::getRangeTypeUniforms(extraSuffix);

    rtn.second.emplace(this->getRangeDefaultGLSLUniformName() + extraSuffix, _defaultVal);

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
                              std::unordered_map<std::string, std::string>& subroutineMap,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false,
                              bool ignoreAccum = false) final {
    if (!ignoreAccum && this->hasAccumulator()) {
      this->_bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix, ignoreDomain, true, ignoreAccum);
    } else {
      this->_bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix, ignoreDomain, ignoreRange, ignoreAccum);
    }

    double diff = this->_domainPtr.getDifference(this->_rangePtr.size());
    activeShader->setUniformAttribute<double>("quantizeDiff", diff);
  }

  std::string getGLSLCode(const std::string& extraSuffix = "",
                          bool ignoreDomain = false,
                          bool ignoreRange = false,
                          bool ignoreAccum = false) final {
    std::string shaderCode =
        this->_getGLSLCode(extraSuffix, ignoreDomain, ignoreRange, ignoreAccum, BaseScale::ScaleShaderType::QUANTIZE);
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

 private:
  void bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                             const std::string& attrName,
                             bool checkFullSize = true) final {
    RUNTIME_EX_ASSERT(this->_rangePtr.getType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation range types.");

    auto& data = this->_rangePtr.getVectorDataRef();
    CHECK(static_cast<int>(data.size()) == this->getNumAccumulatorValues());
    shaderPtr->setUniformAttribute(attrName, data, checkFullSize);
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALE_H_
