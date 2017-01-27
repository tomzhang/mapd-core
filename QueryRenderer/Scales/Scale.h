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

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/any.hpp>

namespace QueryRenderer {

ScaleInterpType getScaleInterpTypeFromJSONObj(const rapidjson::Value& obj);

class BaseScale : public JSONRefObject {
 public:
  static const size_t maxAccumTextures;
  static size_t convertNumAccumValsToNumAccumTextures(size_t numAccumVals, AccumulatorType accumType);
  static size_t convertNumAccumTexturesToNumAccumVals(size_t numAccumTxts, AccumulatorType accumType);

  enum class ScaleShaderType { QUANTITATIVE = 0, ORDINAL, QUANTIZE };
  const static std::vector<std::string> scaleVertexShaderSource;

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

  ScaleType getType() { return _type; }
  AccumulatorType getAccumulatorType() const { return _accumType; }
  bool hasAccumulator() const { return _accumType != AccumulatorType::UNDEFINED; }
  int getNumAccumulatorValues() const { return _numAccumulatorVals; }
  size_t getNumAccumulatorTextures() const {
    return convertNumAccumValsToNumAccumTextures(_numAccumulatorVals, _accumType);
  }

  QueryDataType getDomainDataType() { return _domainDataType; }
  QueryDataType getRangeDataType() { return _rangeDataType; }
  virtual ::Rendering::Colors::ColorType getRangeColorType() const = 0;

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();
  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");
  std::string getAccumulator1stPassFragSrc() const;

  virtual std::string getGLSLCode(const std::string& extraSuffix = "",
                                  bool ignoreDomain = false,
                                  bool ignoreRange = false,
                                  bool ignoreAccum = false) = 0;

  std::string getDomainGLSLTypeName(const std::string& extraSuffix = "") { return "domainType_" + _name + extraSuffix; }
  std::string getRangeGLSLTypeName(const std::string& extraSuffix = "") { return "rangeType_" + _name + extraSuffix; }

  std::string getDomainGLSLUniformName() { return "uDomains_" + _name; }
  std::string getRangeGLSLUniformName() { return "uRanges_" + _name; }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      std::unordered_map<std::string, std::string>& subroutineMap,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false,
                                      bool ignoreAccum = false) = 0;

  virtual BaseScaleDomainRangeData* getDomainData(const bool getOrig = false) = 0;
  virtual BaseScaleDomainRangeData* getRangeData(const bool getOrig = false) = 0;

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
      const std::string& extraSuffix,
      const ScaleDomainRangeData<int>* domainOverride) const = 0;

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<unsigned int>* domainOverride) const = 0;

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<float>* domainOverride) const = 0;

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<double>* domainOverride) const = 0;

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<int64_t>* domainOverride) const = 0;

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<uint64_t>* domainOverride) const = 0;

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

  bool _allowsAccumulator;

  std::string _printInfo() const;

  bool _updateAccumulatorFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _setNumAccumulatorVals(int numAccumulatorVals);
  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                               std::unordered_map<std::string, std::string>& subroutineMap,
                               bool ignoreAccum);

  bool hasNumAccumulatorValsChanged() const { return _numAccumulatorValsChanged; }

  struct OverrideData {
    ScaleDomainRangeDataShPtr dataPtr;
    QueryDataTableSqlShPtr dataTablePtr;

    void reset() {
      dataPtr = nullptr;
      dataTablePtr = nullptr;
    }
  };

  OverrideData _domainOverrideData;
  ScaleDomainRangeDataShPtr _rangeOverrideData;

 private:
  struct PerGpuData : BasePerGpuData {
    std::vector<::Rendering::GL::Resources::GLTexture2dShPtr> accumulatorTexPtrArray;

    ::Rendering::GL::Resources::GLShaderShPtr accumulator2ndPassShaderPtr;
    ::Rendering::GL::Resources::GLVertexBufferShPtr rectvbo;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vao;

    PerGpuData() : BasePerGpuData() {}
    explicit PerGpuData(const RootPerGpuDataShPtr& rootData,
                        const ::Rendering::GL::Resources::GLShaderShPtr& accumulator2ndPassShaderPtr = nullptr,
                        const ::Rendering::GL::Resources::GLVertexBufferShPtr& rectvbo = nullptr,
                        const ::Rendering::GL::Resources::GLVertexArrayShPtr& vao = nullptr)
        : BasePerGpuData(rootData),
          accumulator2ndPassShaderPtr(accumulator2ndPassShaderPtr),
          rectvbo(rectvbo),
          vao(vao) {}
    explicit PerGpuData(const BasePerGpuData& data,
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

  void _setDomainOverride(const ScaleDomainRangeDataShPtr& domainOverridePtr,
                          const QueryDataTableSqlShPtr& domainOverrideTablePtr);
  void _setRangeOverride(const ScaleDomainRangeDataShPtr& rangeOverridePtr);
  bool _hasDomainOverride() const;
  bool _hasRangeOverride() const;
  std::string _getDomainOverrideTableName() const;

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
        _domainPtr(ctx, "domain", false),
        _rangePtr(ctx, "range", true),
        _nullVal(),
        _useNullVal(false) {
    _initGLTypes();
  }

  virtual ~Scale() {}

  BaseScaleDomainRangeData* getDomainData(const bool getOrig = false) {
    if (!getOrig && _domainOverrideData.dataPtr) {
      return _domainOverrideData.dataPtr.get();
    }
    return &_domainPtr;
  };

  BaseScaleDomainRangeData* getRangeData(const bool getOrig = false) {
    if (!getOrig && _rangeOverrideData) {
      return _rangeOverrideData.get();
    }
    return &_rangePtr;
  };

  ::Rendering::Colors::ColorType getRangeColorType() const final { return _getRangeColorTypeInternal(); }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<int>* domainOverride) const {
    return _getDomainTypeUniforms<int>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<unsigned int>* domainOverride) const {
    return _getDomainTypeUniforms<unsigned int>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<float>* domainOverride) const {
    return _getDomainTypeUniforms<float>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<double>* domainOverride) const {
    return _getDomainTypeUniforms<double>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<int64_t>* domainOverride) const {
    return _getDomainTypeUniforms<int64_t>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<uint64_t>* domainOverride) const {
    return _getDomainTypeUniforms<uint64_t>(extraSuffix, domainOverride);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getDomainTypeUniforms(
      const std::string& extraSuffix) const {
    auto overridePtr = _domainOverrideData.dataPtr;
    if (overridePtr) {
      auto uintDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<unsigned int>>(overridePtr);
      if (uintDomain) {
        return _getDomainTypeUniforms<unsigned int>(extraSuffix, uintDomain.get());
      }
      auto intDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<int>>(overridePtr);
      if (intDomain) {
        return _getDomainTypeUniforms<int>(extraSuffix, intDomain.get());
      }
      auto floatDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<float>>(overridePtr);
      if (floatDomain) {
        return _getDomainTypeUniforms<float>(extraSuffix, floatDomain.get());
      }
      auto doubleDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<double>>(overridePtr);
      if (doubleDomain) {
        return _getDomainTypeUniforms<double>(extraSuffix, doubleDomain.get());
      }
      auto stringDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<std::string>>(overridePtr);
      if (stringDomain) {
        return _getDomainTypeUniforms<std::string>(extraSuffix, stringDomain.get());
      }
      auto colorRGBADomain =
          std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>>(overridePtr);
      if (colorRGBADomain) {
        return _getDomainTypeUniforms<::Rendering::Colors::ColorRGBA>(extraSuffix, colorRGBADomain.get());
      }
      auto colorHSLDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>>(overridePtr);
      if (colorHSLDomain) {
        return _getDomainTypeUniforms<::Rendering::Colors::ColorHSL>(extraSuffix, colorHSLDomain.get());
      }
      auto colorLABDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>>(overridePtr);
      if (colorLABDomain) {
        return _getDomainTypeUniforms<::Rendering::Colors::ColorLAB>(extraSuffix, colorLABDomain.get());
      }
      auto colorHCLDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>>(overridePtr);
      if (colorHCLDomain) {
        return _getDomainTypeUniforms<::Rendering::Colors::ColorHCL>(extraSuffix, colorHCLDomain.get());
      }

      THROW_RUNTIME_EX("Override domain type not supported.");
    }

    return _getDomainTypeUniforms<DomainType>(extraSuffix, &_domainPtr);
  }

  virtual std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> getRangeTypeUniforms(
      const std::string& extraSuffix) const {
    std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> rtn(
        _rangePtr.getType(), std::unordered_map<std::string, boost::any>());

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
    CHECK(!_domainOverrideData.dataPtr || _domainOverrideData.dataPtr->size() == _domainPtr.size());
    RUNTIME_EX_ASSERT(_domainPtr.size() > 0 && _rangePtr.size() > 0,
                      std::string(*this) + " getGLSLCode(): domain/range of scale \"" + _name + "\" has no value.");

    std::string shaderCode = scaleVertexShaderSource[static_cast<int>(scaleShader)];

    std::ostringstream ss;

    if (!ignoreDomain) {
      auto typeGLToUse = _domainTypeGL;
      if (_domainOverrideData.dataPtr) {
        typeGLToUse = _domainOverrideData.dataPtr->getTypeGL();
      }

      CHECK(typeGLToUse);
      boost::replace_first(shaderCode, "<domainType>", typeGLToUse->glslType());
      boost::replace_all(shaderCode, "<domainTypeEnum>", std::to_string(typeGLToUse->glslGLType()));
    }

    if (!ignoreRange) {
      // NOTE: not using the range override data here because it should
      // be the same type as the original range data
      boost::replace_first(shaderCode, "<rangeType>", _rangeTypeGL->glslType());
      boost::replace_all(shaderCode, "<rangeTypeEnum>", std::to_string(_rangeTypeGL->glslGLType()));
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

  std::pair<bool, bool> _updateFromJSONObj(
      const rapidjson::Value& obj,
      const rapidjson::Pointer& objPath,
      std::function<void(DomainType&)> domainValConvert = [](DomainType& domainVal) { return; },
      std::function<void(RangeType&)> rangeValConvert = [](RangeType& rangeVal) { return; }) {
    bool domainRangeUpdated = false;
    bool cacheUpdated = false;

    if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
      cacheUpdated = true;
      RUNTIME_EX_ASSERT(
          obj.IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "scale items must be objects."));

      _domainPtr.initializeFromJSONObj(obj, objPath, _type, _domainSizeChanged, _domainValsChanged, domainValConvert);
      _rangePtr.initializeFromJSONObj(obj, objPath, _type, _rangeSizeChanged, _rangeValsChanged, rangeValConvert);

      domainRangeUpdated = hasDomainDataChanged() || hasRangeDataChanged();

      if (hasDomainDataChanged()) {
        _domainOverrideData.reset();
      }

      if (hasRangeDataChanged()) {
        _rangeOverrideData.reset();
      }

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

        domainRangeUpdated = domainRangeUpdated || accumUpdated;

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

    return std::make_pair(domainRangeUpdated, cacheUpdated);
  }

  void _bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                               std::unordered_map<std::string, std::string>& subroutineMap,
                               const std::string& extraSuffix,
                               bool ignoreDomain,
                               bool ignoreRange,
                               bool ignoreAccum,
                               bool ignoreNull = false) {
    if (!ignoreDomain) {
      auto overridePtr = _domainOverrideData.dataPtr;
      if (overridePtr) {
        auto uintDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<unsigned int>>(overridePtr);
        if (uintDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, uintDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix, uintDomain->getNullValue());
          }
        }
        auto intDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<int>>(overridePtr);
        if (intDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, intDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix, intDomain->getNullValue());
          }
        }
        auto floatDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<float>>(overridePtr);
        if (floatDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, floatDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              floatDomain->getNullValue());
          }
        }
        auto doubleDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<double>>(overridePtr);
        if (doubleDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, doubleDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              doubleDomain->getNullValue());
          }
        }
        auto stringDomain = std::dynamic_pointer_cast<ScaleDomainRangeData<std::string>>(overridePtr);
        if (stringDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, stringDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              stringDomain->getNullValue());
          }
        }
        auto colorRGBADomain =
            std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>>(overridePtr);
        if (colorRGBADomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix,
                                            colorRGBADomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              colorRGBADomain->getNullValue());
          }
        }

        auto colorHSLDomain =
            std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>>(overridePtr);
        if (colorHSLDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix,
                                            colorHSLDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              colorHSLDomain->getNullValue());
          }
        }

        auto colorLABDomain =
            std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>>(overridePtr);
        if (colorLABDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix,
                                            colorLABDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              colorLABDomain->getNullValue());
          }
        }

        auto colorHCLDomain =
            std::dynamic_pointer_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>>(overridePtr);
        if (colorHCLDomain) {
          activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix,
                                            colorHCLDomain->getVectorDataRef());
          if (_useNullVal) {
            activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix,
                                              colorHCLDomain->getNullValue());
          }
        }

        THROW_RUNTIME_EX("Override domain type not supported.");
      } else {
        activeShader->setUniformAttribute(getDomainGLSLUniformName() + extraSuffix, _domainPtr.getVectorDataRef());

        if (_useNullVal) {
          activeShader->setUniformAttribute("nullDomainVal_" + this->_name + extraSuffix, _domainPtr.getNullValue());
        }
      }
    }

    if (!ignoreRange) {
      if (_rangeOverrideData) {
        auto overrideData = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(_rangeOverrideData);
        CHECK(overrideData);
        activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, overrideData->getVectorDataRef());
      } else {
        activeShader->setUniformAttribute(getRangeGLSLUniformName() + extraSuffix, _rangePtr.getVectorDataRef());
      }

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
  RangeType _nullVal;
  bool _useNullVal;

 private:
  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initGLTypes() {
    _domainTypeGL = _domainPtr.getTypeGL();
    _rangeTypeGL = _rangePtr.getTypeGL();
  }

  template <typename T>
  std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> _getDomainTypeUniforms(
      const std::string& extraSuffix,
      const ScaleDomainRangeData<T>* domainPtr) const {
    std::pair<QueryDataType, std::unordered_map<std::string, boost::any>> rtn(
        domainPtr->getType(), std::unordered_map<std::string, boost::any>());

    if (_useNullVal) {
      rtn.second.emplace("nullDomainVal_" + this->_name + extraSuffix, domainPtr->getNullValue());
    }

    return rtn;
  }

  template <typename Q = RangeType, typename std::enable_if<::Rendering::Colors::is_color<Q>::value>::type* = nullptr>
  ::Rendering::Colors::ColorType _getRangeColorTypeInternal() const {
    return ::Rendering::Colors::getColorType<Q>();
  }

  template <typename Q = RangeType, typename std::enable_if<std::is_arithmetic<Q>::value>::type* = nullptr>
  ::Rendering::Colors::ColorType _getRangeColorTypeInternal() const {
    THROW_RUNTIME_EX(std::string(*this) + " The range values are not color types.");
  }
};

template <typename DomainType, typename RangeType>
class QuantitativeScale : public Scale<DomainType, RangeType> {
 public:
  QuantitativeScale(const rapidjson::Value& obj,
                    const rapidjson::Pointer& objPath,
                    const QueryRendererContextShPtr& ctx,
                    const std::string& name,
                    const ScaleType scaleType,
                    const ScaleInterpType interpType = ScaleInterpType::UNDEFINED)
      : Scale<DomainType,
              RangeType>(obj, objPath, ctx, name, scaleType, true, static_cast<uint8_t>(AccumulatorType::DENSITY)),
        _useClamp(false),
        _interpType(interpType) {}

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
      // TODO(croot): cache these name bindings so we're not building up these strings
      // every time.

      std::string transformFunc;
      std::string interpFunc;
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

      switch (_interpType) {
        case ScaleInterpType::InterpolateHsl:
        case ScaleInterpType::InterpolateHcl:
          interpFunc = "colorInterpHslHcl";
          break;
        case ScaleInterpType::InterpolateHslLong:
        case ScaleInterpType::InterpolateHclLong:
          interpFunc = "colorInterpHslHclLong";
          break;
        default:
          interpFunc = "defaultInterp";
          break;
      }

      transformFunc += "_" + this->_name + extraSuffix;
      interpFunc += "_" + this->_name + extraSuffix;

      subroutineMap["quantTransform_" + this->_name + extraSuffix] = transformFunc;
      subroutineMap["quantInterp_" + this->_name + extraSuffix] = interpFunc;
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
    bool drUpdated, cacheUpdated;
    std::tie(drUpdated, cacheUpdated) = this->_updateFromJSONObj(obj, objPath, domainValConvert);

    if (drUpdated || cacheUpdated) {
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

      auto interpType = getScaleInterpTypeFromJSONObj(obj);
      if (interpType != ScaleInterpType::UNDEFINED) {
        RUNTIME_EX_ASSERT(
            _validateInterpolator(interpType),
            RapidJSONUtils::getJsonParseErrorStr(
                this->_ctx->getUserWidgetIds(),
                obj,
                "the \"interpolator\" property has an invalid type " + to_string(interpType) + ". " +
                    to_string(this->_type) + " scales with a range of type " + to_string(this->getRangeDataType()) +
                    " only support the following interpolators: [" +
                    boost::algorithm::join(getScaleInterpTypes(_getSupportedInterpolators()), ", ") + "]"));
      }
      _interpType = interpType;
    }

    if (this->hasAccumulator()) {
      // null values will be accumulated separately, where appropriate
      this->_setNumAccumulatorVals(this->_rangePtr.size() + (this->_useNullVal ? 1 : 0));
    }

    this->_jsonPath = objPath;

    return drUpdated;
  }

 private:
  bool _useClamp;
  ScaleInterpType _interpType;
  bool _clampChanged;

  bool _validateInterpolator(const ScaleInterpType interpType) {
    auto interps = _getSupportedInterpolators();
    for (auto& interp : interps) {
      if (interp == interpType) {
        return true;
      }
    }
    return false;
  }

  template <typename Q = RangeType, typename std::enable_if<std::is_arithmetic<Q>::value>::type* = nullptr>
  std::vector<ScaleInterpType> _getSupportedInterpolators() const {
    return {};
  }

  template <typename Q = RangeType,
            typename std::enable_if<
                ::Rendering::Colors::is_specific_color<Q, ::Rendering::Colors::ColorRGBA>::value>::type* = nullptr>
  std::vector<ScaleInterpType> _getSupportedInterpolators() const {
    return {ScaleInterpType::InterpolateRgb};
  }

  template <typename Q = RangeType,
            typename std::enable_if<
                ::Rendering::Colors::is_specific_color<Q, ::Rendering::Colors::ColorHSL>::value>::type* = nullptr>
  std::vector<ScaleInterpType> _getSupportedInterpolators() const {
    return {ScaleInterpType::InterpolateHsl, ScaleInterpType::InterpolateHslLong};
  }

  template <typename Q = RangeType,
            typename std::enable_if<
                ::Rendering::Colors::is_specific_color<Q, ::Rendering::Colors::ColorLAB>::value>::type* = nullptr>
  std::vector<ScaleInterpType> _getSupportedInterpolators() const {
    return {ScaleInterpType::InterpolateLab};
  }

  template <typename Q = RangeType,
            typename std::enable_if<
                ::Rendering::Colors::is_specific_color<Q, ::Rendering::Colors::ColorHCL>::value>::type* = nullptr>
  std::vector<ScaleInterpType> _getSupportedInterpolators() const {
    return {ScaleInterpType::InterpolateHcl, ScaleInterpType::InterpolateHclLong};
  }

  void bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                             const std::string& attrName,
                             bool checkFullSize = true) final {
    RUNTIME_EX_ASSERT(this->_rangePtr.getType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation range types.");

    auto overrideData = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(this->_rangeOverrideData);
    auto data = (overrideData ? overrideData->getVectorData() : this->_rangePtr.getVectorData());
    if (this->_useNullVal) {
      data.push_back(this->_nullVal);
    }

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
              const std::string& name = "",
              const ScaleInterpType interpType = ScaleInterpType::UNDEFINED)
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::LINEAR, interpType) {
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
           const std::string& name = "",
           const ScaleInterpType interpType = ScaleInterpType::UNDEFINED)
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::LOG, interpType) {
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
           const std::string& name = "",
           const ScaleInterpType interpType = ScaleInterpType::UNDEFINED)
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::POW, interpType), _exponent(1.0) {
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
            const std::string& name = "",
            const ScaleInterpType interpType = ScaleInterpType::UNDEFINED)
      : QuantitativeScale<DomainType, RangeType>(obj, objPath, ctx, name, ScaleType::SQRT, interpType) {
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
    bool rtn = this->_updateFromJSONObj(obj, objPath).first;

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

    this->_jsonPath = objPath;
    _updateDefaultJSONPath(objPath);

    // add an additional accumulator for "defaults"
    if (this->hasAccumulator()) {
      // null values will be accumulated separately, where appropriate
      this->_setNumAccumulatorVals(this->_rangePtr.size() + 1 + (this->_useNullVal ? 1 : 0));
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

    auto overrideData = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(this->_rangeOverrideData);
    auto data = (overrideData ? overrideData->getVectorData() : this->_rangePtr.getVectorData());
    data.push_back(_defaultVal);

    // nulls will be last here. The ordinal scale shader accounds for this by doing a
    // and the default val by doing a numDomains_<name> + 1
    if (this->_useNullVal) {
      data.push_back(this->_nullVal);
    }

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

    // TODO(croot): do I need to worry about any domain overrides here?
    // I don't believe I do, since string domains would not be used in quantize scales
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
    bool rtn = this->_updateFromJSONObj(obj, objPath).first;
    this->_jsonPath = objPath;

    if (this->hasAccumulator()) {
      // null values will be accumulated separately, where appropriate
      this->_setNumAccumulatorVals(this->_rangePtr.size() + (this->_useNullVal ? 1 : 0));
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

    auto overrideData = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(this->_rangeOverrideData);
    auto data = (overrideData ? overrideData->getVectorData() : this->_rangePtr.getVectorData());
    if (this->_useNullVal) {
      data.push_back(this->_nullVal);
    }
    CHECK(static_cast<int>(data.size()) == this->getNumAccumulatorValues());
    shaderPtr->setUniformAttribute(attrName, data, checkFullSize);
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALE_H_
