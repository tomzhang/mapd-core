#include "ScaleRef.h"
#include "Scale.h"
#include "../Marks/RenderProperty.h"

namespace QueryRenderer {

using ::Rendering::Colors::ColorRGBA;
using ::Rendering::Colors::ColorHSL;
using ::Rendering::Colors::ColorLAB;
using ::Rendering::Colors::ColorHCL;

BaseScaleRef::BaseScaleRef(const QueryRendererContextShPtr& ctx,
                           const ScaleShPtr& scalePtr,
                           BaseRenderProperty* rndrProp)
    : _ctx(ctx), _scalePtr(scalePtr), _rndrPropPtr(rndrProp) {}

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

void BaseScaleRef::_initScalePtr(const ScaleDomainRangeDataShPtr& domainDataPtr,
                                 const ScaleDomainRangeDataShPtr& rangeDataPtr) {
  // doing this here because we only need to initialize
  // the gpu resources for a scale when it is being used
  // by a mark and it is guaranteed to be used by a mark
  // when a scale ref is created/updated, so we can
  // initialize the gpu resources here.
  auto accumulates = _scalePtr->hasAccumulator();
  if (accumulates) {
    auto scaleDomainDataPtr = _scalePtr->getDomainData(true);
    CHECK(scaleDomainDataPtr);
    if (scaleDomainDataPtr->getType() == QueryDataType::STRING) {
      // we've got a scale that is accumulating, but it has
      // strings for its domain. That means the strings are
      // dictionary encoded and have been converted to
      // their encoded values here in this scale ref. We need
      // to explicitly set the parent scale's domain to those
      // encoded values so the accumulation shader can get
      // set appropriately. We're going to override the
      // domain values on the scales

      // TODO(croot): is there ever a case where range values
      // are dictionary encoded? If so, we need to support that
      // here too. Or, if there if a day ever comes where non
      // dictionary-encoded strings are supported, then that
      // needs handling too. This logic could possibly be
      // removed once data references are supported in scales.
      // If that's the case, then the dictionary-encoding
      // conversion can take place on the parent scale and
      // therefore this wouldn't be needed.

      auto dataTable = std::dynamic_pointer_cast<BaseQueryDataTableSQL>(_getDataTablePtr());
      CHECK(dataTable);

      bool hasOverride = _scalePtr->_hasDomainOverride();
      RUNTIME_EX_ASSERT(!hasOverride || _scalePtr->_getDomainOverrideTableName() == dataTable->getTableName(),
                        std::string(*this) +
                            "Error trying to initialize a scale reference object. The accumulator scale \"" +
                            _scalePtr->getName() +
                            "\" is being referenced, but its domain has dictionary-encoded strings and it's already "
                            "tied to table: " +
                            _scalePtr->_getDomainOverrideTableName() + " whereas this reference is tied to table: " +
                            dataTable->getTableName() +
                            ". Any references to accumulation scales that have a dictionary-encoded string domain "
                            "must use the same "
                            "data table.");

      if (!hasOverride) {
        _scalePtr->_setDomainOverride(domainDataPtr, dataTable);
      }

      _scalePtr->_setRangeOverride(rangeDataPtr);
    }
  }

  _scalePtr->_initGpuResources(_ctx.get(), true);

  if (accumulates) {
    _rndrPropPtr->_setAccumulatorFromScale(_scalePtr);
  } else {
    _rndrPropPtr->_clearAccumulatorFromScale(_scalePtr);
  }
}

void BaseScaleRef::_deleteScalePtr() {
  _rndrPropPtr->_unsubscribeFromScaleEvent(_scalePtr);
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

}  // namespace QueryRenderer
