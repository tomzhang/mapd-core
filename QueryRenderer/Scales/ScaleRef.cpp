#include "ScaleRef.h"
#include "Scale.h"
#include "../Marks/RenderProperty.h"

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;

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

std::string BaseScaleRef::getScaleGLSLFuncName(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getScaleGLSLFuncName(extraSuffix);
}

std::string BaseScaleRef::getGLSLCode(const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->getGLSLCode(extraSuffix);
}

void BaseScaleRef::bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                          const std::string& extraSuffix) {
  _verifyScalePointer();
  return _scalePtr->bindUniformsToRenderer(activeShader, extraSuffix);
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

void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<ColorRGBA>>& destData,
                            ScaleDomainRangeData<ColorRGBA>* srcData) {
  std::vector<ColorRGBA>& srcVec = srcData->getVectorData();

  destData.reset(new ScaleDomainRangeData<ColorRGBA>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<ColorRGBA>& destVec = destData->getVectorData();
  for (size_t i = 0; i < srcVec.size(); ++i) {
    destVec[i] = srcVec[i];
  }
}

}  // namespace QueryRenderer
