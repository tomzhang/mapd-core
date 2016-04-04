#include "Scale.h"
#include "Utils.h"
#include "shaders/linearScaleTemplate_vert.h"
#include "shaders/ordinalScaleTemplate_vert.h"

namespace QueryRenderer {

const std::vector<std::string> BaseScale::scaleVertexShaderSource = {
    LinearScaleTemplate_vert::source,  // LINEAR
    OrdinalScaleTemplate_vert::source  // ORDINAL
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx,
                     QueryDataType domainDataType,
                     QueryDataType rangeDataType,
                     const std::string& name,
                     ScaleType type)
    : _name(name),
      _type(type),
      _useClamp(false),
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
  _initFromJSONObj(obj, objPath);
}

BaseScale::~BaseScale() {
}

bool BaseScale::_initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = false;
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    RUNTIME_EX_ASSERT(
        obj.IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "scale items must be objects."));

    if (!_name.length()) {
      _name = getScaleNameFromJSONObj(obj);
    }

    if (_type == ScaleType::UNDEFINED) {
      _type = getScaleTypeFromJSONObj(obj);
    }

    if (_type == ScaleType::LINEAR) {
      // TODO(croot): create a separate derived class per scale?
      // once we do this, there's no need to do any updates to the BaseScale class
      // since it would only handle _name & _type & if either of those ever
      // change, a new object is built. Therefore the _initFromJSONObj() call in
      // the derived class's updateFromJSONObj() can be removed and this _initFromJSONObj()
      // method can be made private.
      // Also, by doing this, the _jsonPath property can be fullty managed
      // by the derived class, so this base class can provide a "_setJsonPath()"
      // method or something.
      rapidjson::Value::ConstMemberIterator itr;

      // TODO(croot): move the "clamp" prop name into a const somewhere.
      std::string clampProp = "clamp";

      bool prevClamp = _useClamp;
      if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
        RUNTIME_EX_ASSERT(
            itr->value.IsBool(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "the \"clamp\" property for linear scales must be a boolean."));

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

    rtn = true;
  }

  return rtn;
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
    default:
      THROW_RUNTIME_EX(std::string(*this) + " getScaleGLSLFuncName(): scale type is not supported.");
  }

  return "get" + scaleName + "Scale_" + _name + extraSuffix;
}

std::string BaseScale::_printInfo() const {
  return "(name: " + _name + ", type: " + to_string(_type) + ") " + to_string(_ctx->getUserWidgetIds());
}

}  // namespace QueryRenderer
