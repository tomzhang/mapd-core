#include "ScaleRef.h"
#include "Scale.h"
#include "../Marks/RenderProperty.h"

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;

BaseScaleRef::BaseScaleRef(const QueryRendererContextShPtr& ctx,
                           const ScaleShPtr& scalePtr,
                           BaseRenderProperty* rndrProp)
    : _ctx(ctx), _scalePtr(scalePtr), _rndrPropPtr(rndrProp) {
  // if we get to this point, that means the scale is going to be used by a mark.
  // So we'll initialize any gpu resources the scale may use now.
  _initScalePtr();
}

std::string BaseScaleRef::getName() const {
  _verifyScalePointer();
  return _scalePtr->getName();
}

const std::string& BaseScaleRef::getNameRef() {
  _verifyScalePointer();
  return _scalePtr->getNameRef();
}

const ::Rendering::GL::TypeGLShPtr& BaseScaleRef::getDomainTypeGL() {
  _verifyScalePointer();
  return _scalePtr->getDomainTypeGL();
}

const ::Rendering::GL::TypeGLShPtr& BaseScaleRef::getRangeTypeGL() {
  _verifyScalePointer();
  return _scalePtr->getRangeTypeGL();
}

std::string BaseScaleRef::getDomainGLSLTypeName(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getDomainGLSLTypeName(extraSuffix);
}

std::string BaseScaleRef::getRangeGLSLTypeName(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getRangeGLSLTypeName(extraSuffix);
}

std::string BaseScaleRef::getScaleGLSLFuncName(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getScaleGLSLFuncName(extraSuffix);
}

const ::Rendering::GL::TypeGLShPtr& BaseScaleRef::getDomainTypeGL() const {
  _verifyScalePointer();
  return _scalePtr->getDomainTypeGL();
}

const ::Rendering::GL::TypeGLShPtr& BaseScaleRef::getRangeTypeGL() const {
  _verifyScalePointer();
  return _scalePtr->getRangeTypeGL();
}

std::string BaseScaleRef::getGLSLCode(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getGLSLCode(extraSuffix);
}

void BaseScaleRef::bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                          std::unordered_map<std::string, std::string>& subroutineMap,
                                          const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix);
}

void BaseScaleRef::_verifyScalePointer() const {
  RUNTIME_EX_ASSERT(_scalePtr != nullptr, std::string(*this) + ": The scale reference object is uninitialized.");
}

const QueryDataTableShPtr& BaseScaleRef::_getDataTablePtr() {
  return _rndrPropPtr->getDataTablePtr();
}

std::string BaseScaleRef::_getDataColumnName() {
  return _rndrPropPtr->getDataColumnName();
}

std::string BaseScaleRef::_getRndrPropName() {
  return _rndrPropPtr->getName();
}

void BaseScaleRef::_initScalePtr() {
  // doing this here because we only need to initialize
  // the gpu resources for a scale when it is being used
  // by a mark and it is guaranteed to be used by a mark
  // when a scale ref is created/updated, so we can
  // initialize the gpu resources here.
  _scalePtr->_initGpuResources(_ctx.get(), true);

  if (_scalePtr->hasAccumulator()) {
    _rndrPropPtr->_setAccumulatorFromScale(_scalePtr);
  } else {
    _rndrPropPtr->_clearAccumulatorFromScale(_scalePtr);
  }
}

void BaseScaleRef::_deleteScalePtr() {
  _rndrPropPtr->_unsubscribeFromRefEvent(_scalePtr);
  _rndrPropPtr->_clearAccumulatorFromScale(_scalePtr);
  _rndrPropPtr->_setShaderDirty();
  _scalePtr = nullptr;
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

AccumulatorType BaseScaleRef::getAccumulatorType() const {
  _verifyScalePointer();
  return _scalePtr->getAccumulatorType();
}

bool BaseScaleRef::hasAccumulator() const {
  _verifyScalePointer();
  return _scalePtr->hasAccumulator();
}

void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<ColorRGBA>>& destData,
                            ScaleDomainRangeData<ColorRGBA>* srcData) {
  std::vector<ColorRGBA>& srcVec = srcData->getVectorDataRef();

  destData.reset(new ScaleDomainRangeData<ColorRGBA>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<ColorRGBA>& destVec = destData->getVectorDataRef();
  for (size_t i = 0; i < srcVec.size(); ++i) {
    destVec[i] = srcVec[i];
  }
}

}  // namespace QueryRenderer
