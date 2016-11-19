#include "Scale.h"
#include "Utils.h"
#include "../Utils/ShaderUtils.h"
#include "shaders/quantitativeScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"
#include "shaders/quantizeScaleTemplate_vert.h"
#include "shaders/accumulatorScale_1stPass_frag.h"
#include "shaders/accumulatorScale_2ndPass_vert.h"
#include "shaders/accumulatorScale_2ndPass_frag.h"
#include "../Rendering/QueryAccumTxPool.h"

#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLTexture2dArray.h>

#include <Shared/measure.h>

namespace QueryRenderer {

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    QuantitativeScaleTemplate_vert::source,  // LINEAR, LOG, POW, SQRT
    OrdinalScaleTemplate_vert::source,       // ORDINAL
    QuantizeScaleTemplate_vert::source,      // QUANTIZE
};

const size_t BaseScale::maxAccumTextures = 30;
size_t BaseScale::convertNumAccumValsToNumAccumTextures(size_t numAccumVals, AccumulatorType accumType) {
  if (accumType == AccumulatorType::DENSITY) {
    return 1;
  }
  return (numAccumVals + 1) / 2;
}

size_t BaseScale::convertNumAccumTexturesToNumAccumVals(size_t numAccumTxts, AccumulatorType accumType) {
  RUNTIME_EX_ASSERT(accumType != AccumulatorType::DENSITY,
                    "Cannot determine the number of accumulator values from a density accumulator type.");
  return numAccumTxts * 2;
}

BaseScale::BaseScale(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type,
                     bool allowsAccumulator,
                     uint8_t accumTypeMask)
    : JSONRefObject(RefType::SCALE, name, objPath),
      _type(type),
      _domainDataType(domainDataType),
      _domainTypeGL(nullptr),
      _rangeDataType(rangeDataType),
      _rangeTypeGL(nullptr),
      _ctx(ctx),
      _allowsAccumulator(allowsAccumulator),
      _accumType(AccumulatorType::UNDEFINED),
      _accumTypeChanged(false),
      _accumTypeMask(accumTypeMask),
      _numAccumulatorVals(0),
      _numAccumulatorValsChanged(false),
      _numAccumulatorTxtsChanged(false),
      _numMinStdDev(0),
      _minDensity(0),
      _findMinDensity(false),
      _numMaxStdDev(0),
      _maxDensity(0),
      _findMaxDensity(false),
      _findStdDev(false),
      _justBuilt(false) {}

BaseScale::~BaseScale() {
  _clearResources();
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
    case ScaleType::LOG:
    case ScaleType::POW:
    case ScaleType::SQRT:
      scaleName = "Quantitative";
      break;
    case ScaleType::ORDINAL:
      scaleName = "Ordinal";
      break;
    case ScaleType::QUANTIZE:
      scaleName = "Quantize";
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + " getScaleGLSLFuncName(): scale type is not supported.");
  }

  return "get" + scaleName + "Scale_" + _name + extraSuffix;
}

std::string BaseScale::getAccumulator1stPassFragSrc() const {
  CHECK(hasAccumulator() && _perGpuData.size());

  std::string fragSrc = AccumulatorScale_1stPass_frag::source;

  boost::replace_all(fragSrc, "<name>", _name);
  boost::replace_all(fragSrc, "<numAccumTextures>", std::to_string(getNumAccumulatorTextures()));

  return fragSrc;
}

void BaseScale::accumulationPreRender(const GpuId& gpuId) {
  CHECK(hasAccumulator());

  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  size_t textureArraySize = getNumAccumulatorTextures();
  auto& txPool = itr->second.getAccumTxPool();

  CHECK(itr->second.accumulatorTexPtrArray.size() == 0);
  itr->second.accumulatorTexPtrArray =
      txPool->getInactiveAccumTx(_ctx->getWidth(), _ctx->getHeight(), textureArraySize);
}

void BaseScale::accumulationPostRender(const GpuId& gpuId) {
  CHECK(hasAccumulator());

  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  auto& txPool = itr->second.getAccumTxPool();
  txPool->setAccumTxInactive(itr->second.accumulatorTexPtrArray);

  itr->second.accumulatorTexPtrArray.clear();
}

void BaseScale::renderAccumulation(::Rendering::GL::GLRenderer* glRenderer,
                                   const GpuId& gpuId,
                                   ::Rendering::GL::Resources::GLTexture2dArray* compTxArrayPtr) {
  CHECK(hasAccumulator());

  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  // unneccesary checks?
  // TODO(croot): should I pass the fbo as an argument
  auto fbo = glRenderer->getBoundFramebuffer(::Rendering::GL::Resources::FboBind::READ_AND_DRAW);
  CHECK(fbo != nullptr);

  // there will be no multisampling for rendering based on
  // accumulation textures
  glRenderer->disable(GL_MULTISAMPLE);

  // the 2nd pass of the accumulation, when using 1 gpu, does not
  // do anything with the ids, or in other words GL_COLOR_ATTACHMENT1 of
  // the fbo, so disable it.

  // TODO(croot): re-activate GL_COLOR_ATTACHMENT1/2 after we're all done?
  // TODO(croot): improve this API:
  // 1) the enable/disableAttachment functions should maybe return true/false
  //    if the attachment was changed? But how would we deal with the multiple
  //    attachment methods?
  // 2) add a separate activate method just for drawing and only temporarily
  //    enables/disables the attachments provided?
  // 3) improve the activateEnabledAttachmentsForDrawing method name?
  fbo->disableAttachment(GL_COLOR_ATTACHMENT1);
  fbo->disableAttachment(GL_COLOR_ATTACHMENT2);
  fbo->activateEnabledAttachmentsForDrawing();

  ::Rendering::GL::Resources::GLTexture2dShPtr extentTx;
  if (_accumType == AccumulatorType::DENSITY && (_findMinDensity || _findMaxDensity || _findStdDev)) {
    auto& accumPool = itr->second.getAccumTxPool();

    // there should only be 1 accumulation texture when the accumulation
    // type is DENSITY
    CHECK(itr->second.accumulatorTexPtrArray.size() == 1);

    if (compTxArrayPtr) {
      extentTx = accumPool->getInactiveExtentsTx(compTxArrayPtr, _findStdDev);
    } else {
      extentTx = accumPool->getInactiveExtentsTx(itr->second.accumulatorTexPtrArray[0], _findStdDev);
    }
  }

  glRenderer->setClearColor(0, 0, 0, 0);
  glRenderer->clearAll();
  glRenderer->bindShader(itr->second.accumulator2ndPassShaderPtr);

  itr->second.accumulator2ndPassShaderPtr->setImageLoadStoreAttribute("inTxPixelCounter",
                                                                      itr->second.accumulatorTexPtrArray);

  if (compTxArrayPtr) {
    // match the binding unit from the shader
    itr->second.accumulator2ndPassShaderPtr->setImageLoadStoreImageUnit("inTxArrayPixelCounter",
                                                                        itr->second.accumulatorTexPtrArray.size());
    itr->second.accumulator2ndPassShaderPtr->setImageLoadStoreAttribute("inTxArrayPixelCounter", compTxArrayPtr);
  }

  std::unordered_map<std::string, std::string> subroutines(
      {{"getAccumulatedColor", ""},
       {"getAccumulatedCnt", (compTxArrayPtr != nullptr ? "getTxArrayAccumulatedCnt" : "getTxAccumulatedCnt")},
       {"getMinDensity", "getUniformMinDensity"},
       {"getMaxDensity", "getUniformMaxDensity"},
       {"calcMeanStdDev", "getEmptyMeanStdDev"}});

  if (_findMinDensity || _findMaxDensity || _findStdDev) {
    CHECK(extentTx);
    itr->second.accumulator2ndPassShaderPtr->setSamplerTextureImageUnit("densityExtents", GL_TEXTURE0);
    itr->second.accumulator2ndPassShaderPtr->setSamplerAttribute("densityExtents", extentTx);

    if (_findMinDensity) {
      if (_numMinStdDev > 0) {
        subroutines["getMinDensity"] = "getStdDevMinDensity";
      } else {
        subroutines["getMinDensity"] = "getTextureMinDensity";
      }
    }

    if (_findMaxDensity) {
      if (_numMaxStdDev > 0) {
        subroutines["getMaxDensity"] = "getStdDevMaxDensity";

      } else {
        subroutines["getMaxDensity"] = "getTextureMaxDensity";
      }
    }

    if (_findStdDev) {
      subroutines["calcMeanStdDev"] = "getTextureMeanStdDev";
    }
  }

  std::string accumFunc;
  bool ignoreDomain = true, ignoreRange = true;
  switch (_accumType) {
    case AccumulatorType::MIN:
      accumFunc = "getMinAccumulatedColor";
      break;
    case AccumulatorType::MAX:
      accumFunc = "getMaxAccumulatedColor";
      break;
    case AccumulatorType::BLEND:
      accumFunc = "getBlendAccumulatedColor";
      break;
    case AccumulatorType::DENSITY:
      accumFunc = "getDensityAccumulatedColor";
      ignoreDomain = false;
      ignoreRange = false;
      break;
    default:
      THROW_RUNTIME_EX("Accumulator type " + to_string(_accumType) + " is currently unsupported for rendering");
  }
  subroutines["getAccumulatedColor"] = accumFunc;

  bindUniformsToRenderer(itr->second.accumulator2ndPassShaderPtr.get(),
                         subroutines,
                         "_ACCUMULATION",  // TODO(croot): expose as a global
                         ignoreDomain,
                         ignoreRange,
                         true);

  itr->second.accumulator2ndPassShaderPtr->setSubroutines(subroutines);
  itr->second.accumulator2ndPassShaderPtr->setUniformAttribute<uint32_t>("minDensity", _minDensity);
  itr->second.accumulator2ndPassShaderPtr->setUniformAttribute<uint32_t>("maxDensity", _maxDensity);
  itr->second.accumulator2ndPassShaderPtr->setUniformAttribute<float>("minStdDev", static_cast<float>(_numMinStdDev));
  itr->second.accumulator2ndPassShaderPtr->setUniformAttribute<float>("maxStdDev", static_cast<float>(_numMaxStdDev));

  bindAccumulatorColors(itr->second.accumulator2ndPassShaderPtr, "inColors");

  glRenderer->bindVertexArray(itr->second.vao);
  glRenderer->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // restore state/multi-sampling
  glRenderer->enable(GL_MULTISAMPLE);

  if (extentTx) {
    auto& accumTxPool = itr->second.getAccumTxPool();
    accumTxPool->setExtentsTxInactive(extentTx);
  }
}

const std::vector<::Rendering::GL::Resources::GLTexture2dShPtr>& BaseScale::getAccumulatorTextureArrayRef(
    const GpuId& gpuId) {
  RUNTIME_EX_ASSERT(hasAccumulator(),
                    std::string(*this) + ": The scale is not an accumulator scale. Cannot get accumulator textures.");

  auto itr = _perGpuData.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": " + std::to_string(gpuId) + " is not a valid gpu id.");

  return itr->second.accumulatorTexPtrArray;
}

void BaseScale::bindAccumulatorColors(::Rendering::GL::Resources::GLShaderShPtr& shaderPtr,
                                      const std::string& attrName,
                                      bool checkFullSize) {
  THROW_RUNTIME_EX("Accumulation is not activated for the scale type: " + to_string(_type));
}

std::string BaseScale::_printInfo() const {
  return "(name: " + _name + ") " + to_string(_ctx->getUserWidgetIds());
}

bool BaseScale::_updateAccumulatorFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  // NOTE: obj should be a JSON object by the time it reaches here
  CHECK(obj.IsObject());

  AccumulatorType type;
  bool rtn;

  // TODO(croot): expose "accumulator" as a constant somewhere;
  static const std::string accumulatorStr = "accumulator";
  rapidjson::Value::ConstMemberIterator mitr;

  if ((mitr = obj.FindMember(accumulatorStr.c_str())) != obj.MemberEnd()) {
    QueryDataType itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value, true);

    RUNTIME_EX_ASSERT(
        itemType == QueryDataType::STRING,
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), mitr->value, "scale \"" + _name + "\" accumulator must be a string."));

    std::string accumTypeStr(makeUpperCase(mitr->value.GetString()));

    if (accumTypeStr == "MIN") {
      type = AccumulatorType::MIN;
    } else if (accumTypeStr == "MAX") {
      type = AccumulatorType::MAX;
    } else if (accumTypeStr == "BLEND") {
      type = AccumulatorType::BLEND;
    } else if (accumTypeStr == "DENSITY") {
      type = AccumulatorType::DENSITY;

      _findStdDev = false;
      _numMinStdDev = 0;
      _numMaxStdDev = 0;

      static const std::string densityMin = "minDensityCnt";
      static const std::string densityMax = "maxDensityCnt";

      RUNTIME_EX_ASSERT(
          (mitr = obj.FindMember(densityMax.c_str())) != obj.MemberEnd(),
          RapidJSONUtils::getJsonParseErrorStr(
              _ctx->getUserWidgetIds(),
              obj,
              "A \"" + densityMax + "\" attribute is missing which is required for density-based accumulations"));

      itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value, true);
      RUNTIME_EX_ASSERT(
          itemType == QueryDataType::UINT || itemType == QueryDataType::INT || itemType == QueryDataType::STRING,
          RapidJSONUtils::getJsonParseErrorStr(
              _ctx->getUserWidgetIds(), obj, "Density max must be an integer or the string \"max\"."));

      if (mitr->value.IsString()) {
        std::string val = std::string(mitr->value.GetString());
        RUNTIME_EX_ASSERT(
            val == "max" || (_findStdDev = (val == "1stStdDev")) || (_findStdDev = (val == "2ndStdDev")),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(),
                obj,
                "Density max must be an integer or the string \"max\", \"1stStdDev\", or \"2ndStdDev\"."));

        if (_findStdDev) {
          switch (val[0]) {
            case '1':
              _numMaxStdDev = 1;
              break;
            case '2':
              _numMaxStdDev = 2;
              break;
            default:
              THROW_RUNTIME_EX(std::string(*this) + ": stddev string: " + val + " is unsupported.");
          }
        }

        _findMaxDensity = true;
      } else if (mitr->value.IsInt()) {
        RUNTIME_EX_ASSERT(
            mitr->value.GetInt() >= 0,
            RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "Density max must be an integer > 0."));

        _maxDensity = static_cast<uint32_t>(mitr->value.GetInt());
      } else {
        _maxDensity = static_cast<uint32_t>(mitr->value.GetUint());
      }

      if ((mitr = obj.FindMember(densityMin.c_str())) != obj.MemberEnd()) {
        itemType = RapidJSONUtils::getDataTypeFromJSONObj(mitr->value, true);
        RUNTIME_EX_ASSERT(
            itemType == QueryDataType::UINT || itemType == QueryDataType::INT || itemType == QueryDataType::STRING,
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "Density min must be an integer or the string \"min\"."));

        if (mitr->value.IsString()) {
          bool useStdDev = false;
          std::string val = std::string(mitr->value.GetString());
          RUNTIME_EX_ASSERT(
              val == "min" || (useStdDev = (val == "-1stStdDev")) || (useStdDev = (val == "-2ndStdDev")),
              RapidJSONUtils::getJsonParseErrorStr(
                  _ctx->getUserWidgetIds(),
                  obj,
                  "Density min must be an integer or the string \"min\", \"-1stStdDev\", or \"-2ndStdDev\"."));
          _findMinDensity = true;

          if (useStdDev) {
            switch (val[1]) {
              case '1':
                _numMinStdDev = 1;
                break;
              case '2':
                _numMinStdDev = 2;
                break;
              default:
                THROW_RUNTIME_EX(std::string(*this) + ": stddev string: " + val + " is unsupported.");
            }
          }
          _findStdDev = (_findStdDev || useStdDev);
        } else if (mitr->value.IsInt()) {
          RUNTIME_EX_ASSERT(mitr->value.GetInt() >= 0,
                            RapidJSONUtils::getJsonParseErrorStr(
                                _ctx->getUserWidgetIds(), obj, "Density min must be an integer > 0."));

          _minDensity = static_cast<uint32_t>(mitr->value.GetInt());
        } else {
          _minDensity = static_cast<uint32_t>(mitr->value.GetUint());
        }
      } else {
        _minDensity = 0;
      }
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(),
          mitr->value,
          "scale \"" + _name + "\" accumulator \"" + accumTypeStr + "\" is not a supported accumulation type."));
    }

    RUNTIME_EX_ASSERT(
        (_accumTypeMask & static_cast<uint8_t>(type)) > 0,
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(),
            obj,
            "Scale of type " + to_string(_type) + " does not support an accumulator of type " + to_string(type) + "."));
  } else {
    // set an undefined default
    type = AccumulatorType::UNDEFINED;
  }

  rtn = (type != _accumType);

  _accumType = type;
  _accumTypeChanged = rtn;

  if (rtn && !hasAccumulator()) {
    // clean out the gpu resources if the scale has changed to not be an accumulator
    _clearResources();
    _perGpuData.clear();
    _ctx->removeAccumulatorScale(_name);
  }

  return rtn;
}

void BaseScale::_setNumAccumulatorVals(int numAccumulatorVals) {
  RUNTIME_EX_ASSERT(numAccumulatorVals > 0, std::string(*this) + ", need to have at least 1 accumulator.");
  if (numAccumulatorVals != _numAccumulatorVals) {
    _numAccumulatorValsChanged = true;
    auto prevNumTxts = getNumAccumulatorTextures();

    RUNTIME_EX_ASSERT(convertNumAccumValsToNumAccumTextures(numAccumulatorVals, _accumType) <= maxAccumTextures,
                      std::string(*this) +
                          " There are too many accumulator values to do a render-based accumation as it requires too "
                          "many textures. There are " +
                          std::to_string(numAccumulatorVals) + " values requested for accumulation requiring " +
                          std::to_string(convertNumAccumValsToNumAccumTextures(numAccumulatorVals, _accumType)) +
                          " but there's a limit of " + std::to_string(maxAccumTextures) + " textures");

    _numAccumulatorVals = numAccumulatorVals;
    auto newNumTxts = getNumAccumulatorTextures();

    _numAccumulatorTxtsChanged = (prevNumTxts != newNumTxts);
  } else {
    _numAccumulatorValsChanged = false;
    _numAccumulatorTxtsChanged = false;
  }
}

void BaseScale::_bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                        std::unordered_map<std::string, std::string>& subroutineMap,
                                        bool ignoreAccum) {
  if (!ignoreAccum && hasAccumulator()) {
    auto gpuId = activeShader->getGLRenderer()->getGpuId();
    auto itr = this->_perGpuData.find(gpuId);
    CHECK(itr != this->_perGpuData.end());
    activeShader->setImageLoadStoreAttribute("inTxPixelCounter", itr->second.accumulatorTexPtrArray);

    std::string accumFunc;
    switch (getAccumulatorType()) {
      case AccumulatorType::MIN:
      case AccumulatorType::MAX:
      case AccumulatorType::BLEND:
        accumFunc = "minMaxBlendAccumulate";
        break;
      case AccumulatorType::DENSITY:
        accumFunc = "densityAccumulate";
        break;
      default:
        THROW_RUNTIME_EX("Accumulator type " + to_string(getAccumulatorType()) +
                         " is not currently supported for mark rendering.");
        break;
    }

    // TODO(croot): should we verify that subroutines of the
    // same name don't exist? I'm choosing not to as I'm
    // assuming the shader compilation would fail if there are two
    // subroutines of the same name.
    subroutineMap["accumulate"] = accumFunc;
  }
}

::Rendering::GL::Resources::GLShaderShPtr BaseScale::_buildAccumulatorShader(
    ::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
    size_t numTextures) {
  std::string vertSrc = AccumulatorScale_2ndPass_vert::source;
  std::string fragSrc = AccumulatorScale_2ndPass_frag::source;

  boost::replace_all(fragSrc, "<name>", _name);
  boost::replace_all(fragSrc, "<numAccumColors>", std::to_string(_numAccumulatorVals));
  boost::replace_all(fragSrc, "<numAccumTextures>", std::to_string(numTextures));

  std::string suffix = "_ACCUMULATION";
  auto funcRange = ShaderUtils::getGLSLFunctionBounds(fragSrc, "getDensityColor");
  std::string scaleCode = getGLSLCode(suffix, false, false, true);
  boost::replace_range(fragSrc, funcRange, scaleCode);
  boost::replace_all(fragSrc, "getDensityColor(pct)", getScaleGLSLFuncName(suffix) + "(pct)");

  return rsrcMgr->createShader(vertSrc, fragSrc);
}

void BaseScale::_initGpuResources(QueryRendererContext* ctx, bool initializing) {
  if (!hasAccumulator()) {
    return;
  }

  ::Rendering::GL::GLRenderer* renderer;
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr;

  size_t textureArraySize = getNumAccumulatorTextures();

  if (!_perGpuData.size()) {
    if (!initializing) {
      // doing a lazy init, so only initialize when properly instructed to do so
      return;
    }

    auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;
    CHECK(qrmPerGpuDataPtr);

    ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr bufferLayout(
        new ::Rendering::GL::Resources::GLInterleavedBufferLayout());
    bufferLayout->addAttribute<float, 2>("pos");

    for (auto& itr : *qrmPerGpuDataPtr) {
      auto rtn =
          _perGpuData.emplace(std::piecewise_construct, std::forward_as_tuple(itr->gpuId), std::forward_as_tuple(itr));
      auto& gpuData = rtn.first->second;

      itr->makeActiveOnCurrentThread();
      renderer = itr->getGLRenderer();
      CHECK(renderer);

      rsrcMgr = renderer->getResourceManager();

      gpuData.accumulator2ndPassShaderPtr = _buildAccumulatorShader(rsrcMgr, textureArraySize);

      gpuData.rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);

      renderer->bindShader(gpuData.accumulator2ndPassShaderPtr);
      gpuData.vao = rsrcMgr->createVertexArray({{gpuData.rectvbo, {}}});
    }

    ctx->addAccumulatorScale(_name);
    _justBuilt = true;
  } else if (!_justBuilt && (hasNumAccumulatorValsChanged() || hasAccumulatorTypeChanged())) {
    for (auto& gpuItr : _perGpuData) {
      gpuItr.second.makeActiveOnCurrentThread();

      rsrcMgr = gpuItr.second.getGLRenderer()->getResourceManager();
      gpuItr.second.accumulator2ndPassShaderPtr = _buildAccumulatorShader(rsrcMgr, textureArraySize);
    }
  } else {
    _justBuilt = false;
  }

  // unset from update
  // TODO(croot): should scales have an update function much like a mark
  // that's called in QueryRendererContext::update?
  _accumTypeChanged = _numAccumulatorValsChanged = _numAccumulatorTxtsChanged = false;
}

void BaseScale::_setDomainOverride(const ScaleDomainRangeDataShPtr& domainOverridePtr,
                                   const QueryDataTableSqlShPtr& domainOverrideTablePtr) {
  _domainOverrideData.dataPtr = domainOverridePtr;
  _domainOverrideData.dataTablePtr = domainOverrideTablePtr;
}

bool BaseScale::_hasDomainOverride() const {
  return _domainOverrideData.dataPtr != nullptr;
}

std::string BaseScale::_getDomainOverrideTableName() const {
  auto tablePtr = std::dynamic_pointer_cast<BaseQueryDataTableSQL>(_domainOverrideData.dataTablePtr);
  if (tablePtr) {
    return tablePtr->getTableName();
  }
  return "";
}

void BaseScale::_clearResources() {
  // need to clear out the accumulation buffers if we still have them checked out
  auto currRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  for (auto& gpuData : _perGpuData) {
    if (gpuData.second.accumulatorTexPtrArray.size() && gpuData.second.hasAccumTxPool()) {
      gpuData.second.makeActiveOnCurrentThread();
      auto& accumTxPool = gpuData.second.getAccumTxPool();
      accumTxPool->setAccumTxInactive(gpuData.second.accumulatorTexPtrArray);
    }
  }
  if (currRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  } else {
    currRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (currRenderer) {
      currRenderer->makeInactive();
    }
  }
}

}  // namespace QueryRenderer
