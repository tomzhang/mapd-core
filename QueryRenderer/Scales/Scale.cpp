#include "Scale.h"
#include "Utils.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"
#include "shaders/quantizeScaleTemplate_vert.h"

namespace QueryRenderer {

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    LinearScaleTemplate_vert::source,    // LINEAR
    OrdinalScaleTemplate_vert::source,   // ORDINAL
    QuantizeScaleTemplate_vert::source,  // QUANTIZE
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type)
    : _name(name),
      _type(type),
      _domainDataType(domainDataType),
      _domainTypeGL(nullptr),
      _rangeDataType(rangeDataType),
      _rangeTypeGL(nullptr),
      _ctx(ctx) {
}

BaseScale::BaseScale(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type)
    : BaseScale(ctx, domainDataType, rangeDataType, name, type) {
}

BaseScale::~BaseScale() {
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

std::string BaseScale::_printInfo() const {
  return "(name: " + _name + ") " + to_string(_ctx->getUserWidgetIds());
}

}  // namespace QueryRenderer
