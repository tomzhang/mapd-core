#include "Scale.h"
#include "Utils.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"
#include "shaders/quantizeScaleTemplate_vert.h"
#include "shaders/accumulatorScale_1stPass_frag.h"
#include "shaders/accumulatorScale_2ndPass_vert.h"
#include "shaders/accumulatorScale_2ndPass_frag.h"
#include "../Rendering/QueryAccumTxPool.h"

#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLTexture2dArray.h>

namespace QueryRenderer {

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    LinearScaleTemplate_vert::source,    // LINEAR
    OrdinalScaleTemplate_vert::source,   // ORDINAL
    QuantizeScaleTemplate_vert::source,  // QUANTIZE
};

const size_t BaseScale::maxAccumTextures = 30;
size_t BaseScale::convertNumAccumValsToNumAccumTextures(size_t numAccumVals) {
  return (numAccumVals + 1) / 2;
}

size_t BaseScale::convertNumAccumTexturesToNumAccumVals(size_t numAccumTxts) {
  return numAccumTxts * 2;
}

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type,
                     bool allowsAccumulator)
    : _name(name),
      _type(type),
      _domainDataType(domainDataType),
      _domainTypeGL(nullptr),
      _rangeDataType(rangeDataType),
      _rangeTypeGL(nullptr),
      _ctx(ctx),
      _allowsAccumulator(allowsAccumulator),
      _accumType(AccumulatorType::UNDEFINED),
      _accumTypeChanged(false),
      _numAccumulatorVals(0),
      _numAccumulatorValsChanged(false),
      _numAccumulatorTxtsChanged(false),
      _justBuilt(false) {
}

BaseScale::BaseScale(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type,
                     bool allowsAccumulator)
    : BaseScale(ctx, domainDataType, rangeDataType, name, type, allowsAccumulator) {
}

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
      scaleName = "Linear";
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
                                   const ::Rendering::GL::Resources::GLTexture2dShPtr& idTex) {
  CHECK(hasAccumulator());

  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  // unneccesary checks?
  auto fbo = glRenderer->getBoundFramebuffer(::Rendering::GL::Resources::FboBind::READ_AND_DRAW);
  CHECK(fbo != nullptr);

  // the 2nd pass of the accumulation, when using 1 gpu, does not
  // do anything with the ids, or in other words GL_COLOR_ATTACHMENT1 of
  // the fbo, so disable it.

  // TODO(croot): re-activate GL_COLOR_ATTACHMENT1 after we're all done?
  // TODO(croot): improve this API:
  // 1) the enable/disableAttachment functions should maybe return true/false
  //    if the attachment was changed? But how would we deal with the multiple
  //    attachment methods?
  // 2) add a separate activate method just for drawing and only temporarily
  //    enables/disables the attachments provided?
  // 3) improve the activateEnabledAttachmentsForDrawing method name?
  fbo->disableAttachment(GL_COLOR_ATTACHMENT1);
  fbo->activateEnabledAttachmentsForDrawing();

  glRenderer->setClearColor(0, 0, 0, 0);
  glRenderer->clearAll();

  glRenderer->bindShader(itr->second.accumulator2ndPassShaderPtr);

  itr->second.accumulator2ndPassShaderPtr->setImageLoadStoreAttribute("inTxPixelCounter",
                                                                      itr->second.accumulatorTexPtrArray);

  bindAccumulatorColors(itr->second.accumulator2ndPassShaderPtr, "inColors");

  glRenderer->bindVertexArray(itr->second.vao);
  glRenderer->drawVertexBuffers(GL_TRIANGLE_STRIP);
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
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(),
          mitr->value,
          "scale \"" + _name + "\" accumulator \"" + accumTypeStr + "\" is not a supported accumulation type."));
    }
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

    RUNTIME_EX_ASSERT(convertNumAccumValsToNumAccumTextures(numAccumulatorVals) <= maxAccumTextures,
                      std::string(*this) +
                          " There are too many accumulator values to do a render-based accumation as it requires too "
                          "many textures. There are " +
                          std::to_string(numAccumulatorVals) + " values requested for accumulation requiring " +
                          std::to_string(convertNumAccumValsToNumAccumTextures(numAccumulatorVals)) +
                          " but there's a limit of " + std::to_string(maxAccumTextures) + " textures");

    _numAccumulatorVals = numAccumulatorVals;
    auto newNumTxts = getNumAccumulatorTextures();

    _numAccumulatorTxtsChanged = (prevNumTxts != newNumTxts);
  } else {
    _numAccumulatorValsChanged = false;
    _numAccumulatorTxtsChanged = false;
  }
}

void BaseScale::_bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader) {
  if (hasAccumulator()) {
    auto gpuId = activeShader->getGLRenderer()->getGpuId();
    auto itr = this->_perGpuData.find(gpuId);
    CHECK(itr != this->_perGpuData.end());
    activeShader->setImageLoadStoreAttribute("inTxPixelCounter", itr->second.accumulatorTexPtrArray);
  }
}

::Rendering::GL::Resources::GLShaderShPtr BaseScale::_buildAccumulatorShader(
    ::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
    size_t numTextures) {
  std::string vertSrc = AccumulatorScale_2ndPass_vert::source;
  std::string fragSrc = AccumulatorScale_2ndPass_frag::source;

  boost::replace_all(fragSrc, "<name>", _name);
  boost::replace_all(fragSrc, "<accumType>", std::to_string(static_cast<int>(_accumType)));
  boost::replace_all(fragSrc, "<numAccumColors>", std::to_string(_numAccumulatorVals));

  boost::replace_all(fragSrc, "<numAccumTextures>", std::to_string(numTextures));

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
      PerGpuData gpuData(itr);

      itr->makeActiveOnCurrentThread();
      renderer = itr->getGLRenderer();
      CHECK(renderer);

      rsrcMgr = renderer->getResourceManager();

      gpuData.accumulator2ndPassShaderPtr = _buildAccumulatorShader(rsrcMgr, textureArraySize);

      gpuData.rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);

      renderer->bindShader(gpuData.accumulator2ndPassShaderPtr);
      gpuData.vao = rsrcMgr->createVertexArray({{gpuData.rectvbo, {}}});

      _perGpuData.emplace(itr->gpuId, std::move(gpuData));
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
