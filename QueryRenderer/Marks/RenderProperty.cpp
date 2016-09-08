#include "RenderProperty.h"
#include "BaseMark.h"
#include "../Data/BaseQueryDataTable.h"
#include "../Data/QueryDataTable.h"
#include "../Data/QueryPolyDataTable.h"
#include "../Scales/Scale.h"

#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;
using ::Rendering::GL::Resources::GLShader;
using ::Rendering::GL::TypeGLShPtr;

void BaseRenderProperty::initializeFromJSONObj(const rapidjson::Value& obj,
                                               const rapidjson::Pointer& objPath,
                                               const QueryDataTableShPtr& dataPtr) {
  if (obj.IsObject()) {
    rapidjson::Value::ConstMemberIterator mitr;

    // TODO(croot): move the following prop strings to a const somewhere
    std::string fieldProp = "field";
    std::string valueProp = "value";
    std::string scaleProp = "scale";

    bool hasData = false;

    if ((mitr = obj.FindMember(fieldProp.c_str())) != obj.MemberEnd()) {
      // need to clear out the value path
      _valueJsonPath = rapidjson::Pointer();

      if (!_ctx->isJSONCacheUpToDate(_fieldJsonPath, mitr->value)) {
        RUNTIME_EX_ASSERT(dataPtr != nullptr,
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
                              obj,
                              "a data reference for the mark is not defined. Cannot access \"field\"."));
        RUNTIME_EX_ASSERT(mitr->value.IsString(),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(), obj, "\"field\" property for mark must be a string."));

        // TODO(croot): need to update references when a data
        // ptr has changed, but the scale reference hasn't
        // changed.
        initializeFromData(mitr->value.GetString(), dataPtr);
      }

      _fieldJsonPath = objPath.Append(fieldProp.c_str(), fieldProp.length());

      hasData = true;
    } else if ((mitr = obj.FindMember(valueProp.c_str())) != obj.MemberEnd()) {
      // need to clear out the field path
      _fieldJsonPath = rapidjson::Pointer();
      _dataPtr = nullptr;
      _vboAttrName = "";

      if (!_ctx->isJSONCacheUpToDate(_valueJsonPath, mitr->value)) {
        _initValueFromJSONObj(mitr->value);
      }
      _valueJsonPath = objPath.Append(valueProp.c_str(), valueProp.length());

      hasData = true;
    } else {
    }

    if ((mitr = obj.FindMember(scaleProp.c_str())) != obj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_scaleJsonPath, mitr->value)) {
        RUNTIME_EX_ASSERT(
            _useScale,
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "render property \"" + _name + "\" does not support scale references."));

        _initScaleFromJSONObj(mitr->value);
        _verifyScale();
      }

      _scaleJsonPath = objPath.Append(scaleProp.c_str(), scaleProp.length());
    } else {
      // need some value source, either by "field" or by "value" if there's no scale reference
      RUNTIME_EX_ASSERT(hasData,
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            obj,
                            "invalid mark property object. Must contain a data reference via a \"field\" property "
                            "or "
                            "a \"value\" property."));

      // TODO(croot): If a scale was used previously but now it's not, we need to
      // indicate that a shader rebuild is needed. Two possible approaches:
      // 1) have a local pointer that points back to the mark that encapsulates
      //    this render property and have a public function on the mark object
      //    to mark the shader as dirty.
      // 2) mark a local dirty flag and leave it up to the mark to traverse
      //    all its render properties looking for the dirty flag.

      // need to clear out the _scaleJsonPath
      _scaleJsonPath = rapidjson::Pointer();

      _scaleConfigPtr = nullptr;
    }

    _initFromJSONObj(obj);

  } else {
    // need to clear out the object paths
    _fieldJsonPath = rapidjson::Pointer();
    _valueJsonPath = rapidjson::Pointer();
    _scaleJsonPath = rapidjson::Pointer();
    _dataPtr = nullptr;
    _vboAttrName = "";

    _scaleConfigPtr = nullptr;

    _initValueFromJSONObj(obj);
  }
}

void BaseRenderProperty::initializeFromData(const std::string& attrName, const QueryDataTableShPtr& dataPtr) {
  RUNTIME_EX_ASSERT(dataPtr != nullptr,
                    std::string(*this) + ": Cannot initialize mark property " + _name +
                        " from data. A valid data reference hasn't been initialized.");

  _dataPtr = dataPtr;
  _vboAttrName = attrName;

  _initBuffers(_dataPtr->getAttributeDataBuffers(attrName));
  _vboInitType = VboInitType::FROM_DATAREF;

  _prntMark->setPropsDirty();

  _initTypeFromBuffer();
}

int BaseRenderProperty::size(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end()) {
    return itr->second.vbo->numItems();
  }
  return 0;
}

bool BaseRenderProperty::hasVboPtr() {
  for (auto& itr : _perGpuData) {
    if (itr.second.vbo != nullptr) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasVboPtr(const GpuId& gpuId) {
  auto itr = _perGpuData.find(gpuId);

  return (itr != _perGpuData.end() && itr->second.vbo != nullptr);
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end()) {
    return itr->second.vbo;
  }

  return nullptr;
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr() const {
  auto itr = _perGpuData.begin();
  if (itr != _perGpuData.end()) {
    return itr->second.vbo;
  }

  return nullptr;
}

bool BaseRenderProperty::hasUboPtr() {
  for (auto& itr : _perGpuData) {
    if (itr.second.ubo != nullptr) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasUboPtr(const GpuId& gpuId) {
  auto itr = _perGpuData.find(gpuId);

  return (itr != _perGpuData.end() && itr->second.ubo != nullptr);
}

QueryUniformBufferShPtr BaseRenderProperty::getUboPtr(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end()) {
    return itr->second.ubo;
  }

  return nullptr;
}

QueryUniformBufferShPtr BaseRenderProperty::getUboPtr() const {
  auto itr = _perGpuData.begin();
  if (itr != _perGpuData.end()) {
    return itr->second.ubo;
  }

  return nullptr;
}

std::string BaseRenderProperty::getInGLSLType() const {
  RUNTIME_EX_ASSERT(_inType != nullptr,
                    std::string(*this) + " getInGLSLType(): input type for \"" + _name + "\" is uninitialized.");

  return _inType->glslType();
}

const ::Rendering::GL::TypeGLShPtr& BaseRenderProperty::getInTypeGL() const {
  RUNTIME_EX_ASSERT(_inType != nullptr,
                    std::string(*this) + " getInTypeGL(): input type for \"" + _name + "\" is uninitialized.");

  return _inType;
}

std::string BaseRenderProperty::getOutGLSLType() const {
  if (_scaleConfigPtr != nullptr) {
    return _scaleConfigPtr->getRangeTypeGL()->glslType();
  }

  RUNTIME_EX_ASSERT(
      _outType != nullptr,
      std::string(*this) + " getOutGLSLType(): output type for mark property \"" + _name + "\" is uninitialized.");

  return (_outType->glslType());
}

const ::Rendering::GL::TypeGLShPtr& BaseRenderProperty::getOutTypeGL() const {
  RUNTIME_EX_ASSERT(_outType != nullptr,
                    std::string(*this) + " getOutTypeGL(): input type for \"" + _name + "\" is uninitialized.");

  return _outType;
}

void BaseRenderProperty::addToVboAttrMap(const GpuId& gpuId,
                                         ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) const {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr->second.vbo != nullptr,
                    std::string(*this) +
                        " addToVboAttrMap(): A vertex buffer is not defined. Cannot add vbo attrs to "
                        "vbo->shader attr map.");

  ::Rendering::GL::Resources::GLVertexBufferShPtr glVbo = itr->second.vbo->getGLVertexBufferPtr();
  auto attrMapItr = attrMap.find(glVbo);
  if (attrMapItr == attrMap.end()) {
    attrMap.insert({glVbo, {{_vboAttrName, _name}}});
  } else {
    attrMapItr->second.push_back({_vboAttrName, _name});
  }
}

void BaseRenderProperty::initGpuResources(const QueryRendererContext* ctx,
                                          const std::set<GpuId>& usedGpus,
                                          const std::set<GpuId>& unusedGpus) {
  auto qrmPerGpuData = ctx->getRootGpuCache()->perGpuData;

  for (auto gpuId : usedGpus) {
    if (_perGpuData.find(gpuId) == _perGpuData.end()) {
      auto qrmItr = qrmPerGpuData->find(gpuId);
      CHECK(qrmItr != qrmPerGpuData->end());
      PerGpuData gpuData(*qrmItr);
      if (_dataPtr) {
        QueryBufferShPtr bufferPtr = _dataPtr->getAttributeDataBuffer(gpuId, _vboAttrName);
        switch (bufferPtr->getGLResourceType()) {
          case ::Rendering::GL::Resources::GLResourceType::VERTEX_BUFFER:
            gpuData.vbo = std::dynamic_pointer_cast<QueryVertexBuffer>(bufferPtr);
            CHECK(gpuData.vbo);
            break;
          case ::Rendering::GL::Resources::GLResourceType::UNIFORM_BUFFER:
            gpuData.ubo = std::dynamic_pointer_cast<QueryUniformBuffer>(bufferPtr);
            CHECK(gpuData.ubo);
            break;
          default:
            CHECK(false) << "Unsupported resource type " << bufferPtr->getGLResourceType() << " for render properties.";
        }
        _perGpuData.emplace(gpuId, std::move(gpuData));
      }
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }
}

void BaseRenderProperty::_setShaderDirty() {
  _prntMark->setShaderDirty();
}

void BaseRenderProperty::_setPropsDirty() {
  _prntMark->setPropsDirty();
}

std::string BaseRenderProperty::_printInfo() const {
  return "(name: " + _name + ", vbo attr name: " + _vboAttrName + ") " + to_string(_ctx->getUserWidgetIds());
}

std::set<GpuId> BaseRenderProperty::_initUnusedGpus(const std::map<GpuId, QueryBufferShPtr>& bufferMap) {
  std::set<GpuId> usedGpus;
  std::set<GpuId> unusedGpus;
  for (const auto& kv : _perGpuData) {
    if (bufferMap.find(kv.first) == bufferMap.end()) {
      unusedGpus.insert(kv.first);
    }
  }
  return unusedGpus;
}

std::set<GpuId> BaseRenderProperty::_initUnusedGpus(const std::set<GpuId>& usedGpus) {
  std::set<GpuId> unusedGpus;
  for (const auto& kv : _perGpuData) {
    if (usedGpus.find(kv.first) == usedGpus.end()) {
      unusedGpus.insert(kv.first);
    }
  }
  return unusedGpus;
}

void BaseRenderProperty::_initBuffers(const std::map<GpuId, QueryBufferShPtr>& bufferMap) {
  std::set<GpuId> usedGpus;
  for (auto itr : bufferMap) {
    usedGpus.insert(itr.first);
  }
  auto unusedGpus = _initUnusedGpus(bufferMap);

  CHECK(_prntMark->numGpus() == bufferMap.size());
  initGpuResources(_ctx.get(), usedGpus, unusedGpus);
  CHECK(bufferMap.size() == _perGpuData.size());

  ::Rendering::GL::Resources::GLResourceType rsrcType = bufferMap.begin()->second->getGLResourceType();

  switch (rsrcType) {
    case ::Rendering::GL::Resources::GLResourceType::VERTEX_BUFFER: {
      QueryVertexBufferShPtr qvbo;
      for (const auto& itr : bufferMap) {
        auto myItr = _perGpuData.find(itr.first);
        CHECK(myItr != _perGpuData.end() && rsrcType == itr.second->getGLResourceType());
        qvbo = std::dynamic_pointer_cast<QueryVertexBuffer>(itr.second);
        CHECK(qvbo);
        myItr->second.vbo = qvbo;
      }
    } break;
    case ::Rendering::GL::Resources::GLResourceType::UNIFORM_BUFFER: {
      QueryUniformBufferShPtr qubo;
      for (const auto& itr : bufferMap) {
        auto myItr = _perGpuData.find(itr.first);
        CHECK(myItr != _perGpuData.end() && rsrcType == itr.second->getGLResourceType());
        qubo = std::dynamic_pointer_cast<QueryUniformBuffer>(itr.second);
        myItr->second.ubo = qubo;
      }
    } break;

    default:
      CHECK(false) << "Unsupported resource type " << rsrcType << " for render properties.";
  }
}

void BaseRenderProperty::_setAccumulatorFromScale(const ScaleShPtr& scalePtr) {
  if (scalePtr && scalePtr->hasAccumulator()) {
    // TODO(croot): what if there are two accumulator scales in the
    // same mark properties?
    RUNTIME_EX_ASSERT(
        !_prntMark->hasAccumulator() || scalePtr->getName() == _prntMark->getAccumulatorScaleName(),
        std::string(*_prntMark) +
            " has more that one accumulator scale attached to it. There can only be one accumulator scale per mark.");
    _prntMark->setAccumulatorScale(scalePtr->getName());
  }
}

void BaseRenderProperty::_clearAccumulatorFromScale(const ScaleShPtr& scalePtr) {
  if (_prntMark && scalePtr && scalePtr->getName() == _prntMark->getAccumulatorScaleName()) {
    // TODO(croot): what if there are two accumulator scales in the
    // same mark properties?
    _prntMark->clearAccumulatorScale();
  }
}

void BaseRenderProperty::_unsubscribeFromRefEvent(const ScaleShPtr& scalePtr) {
  if (scalePtr) {
    _ctx->unsubscribeFromRefEvent(RefEventType::ALL, scalePtr, _scaleRefSubscriptionCB);
  }
}

void BaseRenderProperty::_updateScalePtr(const ScaleShPtr& scalePtr) {
  _setAccumulatorFromScale(scalePtr);
}

template <>
RenderProperty<ColorRGBA, 1>::RenderProperty(BaseMark* prntMark,
                                             const std::string& name,
                                             const QueryRendererContextShPtr& ctx,
                                             bool useScale,
                                             bool flexibleType,
                                             bool allowsAccumulatorScale)
    : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType, allowsAccumulatorScale),
      _mult(),
      _offset(),
      _uniformVal() {
  _inType.reset(new ::Rendering::GL::TypeGL<float, 4>());
  _outType.reset(new ::Rendering::GL::TypeGL<float, 4>());
}

template <>
void RenderProperty<ColorRGBA, 1>::initializeValue(const ColorRGBA& val) {
  // TODO: combine all the different types into a utility file somewhere.
  // i.e. this is already defined in BufferLayout.h, so let's find a
  // good way to consolidate these definitions

  // TODO(croot): make thread safe

  if (_vboInitType != VboInitType::FROM_VALUE) {
    _inType.reset(new ::Rendering::GL::TypeGL<float, 4>());
    _outType.reset(new ::Rendering::GL::TypeGL<float, 4>());

    for (auto& itr : _perGpuData) {
      itr.second.vbo = nullptr;
    }

    _vboInitType = VboInitType::FROM_VALUE;
    _setPropsDirty();
  } else {
    // TODO(croot): do we need to set props dirty on the parent?
  }

  _uniformVal = val;
}

template <>
void RenderProperty<ColorRGBA, 1>::bindUniformToRenderer(GLShader* activeShader,
                                                         const std::string& uniformAttrName) const {
  // TODO(croot): deal with numComponents here by using a vector instead?
  activeShader->setUniformAttribute<std::array<float, 4>>(uniformAttrName, _uniformVal.getColorArray());
}

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj) {
  // TODO: what about offsets / mults for colors?
}

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsString(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "value for color property \"" + _name + "\" must be a string."));

  ColorRGBA color(obj.GetString());

  initializeValue(color);
}

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale() {
  RUNTIME_EX_ASSERT(_scaleConfigPtr != nullptr,
                    std::string(*this) + ": Cannot verify scale for mark property \"" + _name +
                        "\". Scale reference is uninitialized.");

  TypeGLShPtr vboType = _scaleConfigPtr->getRangeTypeGL();

  // colors need to be a specific type
  RUNTIME_EX_ASSERT(ColorRGBA::isValidTypeGL(vboType),
                    std::string(*this) + ": Vertex buffer to use for mark property \"" + _name +
                        "\" is not an appropriate type for colors.");
}

void EnumRenderProperty::_initValueFromJSONObj(const rapidjson::Value& obj) {
  int val = 0;
  if (obj.IsString()) {
    RUNTIME_EX_ASSERT(_stringConvertFunc != nullptr, std::string(*this) + ": Enum property does not support strings.");

    val = _stringConvertFunc(obj.GetString());

    RUNTIME_EX_ASSERT(val >= 0, std::string(*this) + ": Enum value " + obj.GetString() + " is not supported.");
  } else {
    val = RapidJSONUtils::getNumValFromJSONObj<int>(obj);
  }

  initializeValue(val);
}

}  // namespace QueryRenderer
