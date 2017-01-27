#include "RenderProperty.h"
#include "BaseMark.h"
#include "../Data/BaseQueryDataTable.h"
#include "../Data/QueryDataTable.h"
#include "../Data/QueryPolyDataTable.h"
#include "../Scales/Scale.h"

#include <Rendering/Renderer/GL/Resources/GLShader.h>

#include <boost/algorithm/string/join.hpp>

namespace QueryRenderer {

using ::Rendering::Colors::ColorUnion;
using ::Rendering::GL::Resources::GLShader;
using ::Rendering::GL::TypeGLShPtr;

BaseRenderProperty::~BaseRenderProperty() {
  _unsubscribeFromDataEvent();
}

void BaseRenderProperty::initializeFromJSONObj(const rapidjson::Value& obj,
                                               const rapidjson::Pointer& objPath,
                                               const QueryDataTableShPtr& dataPtr) {
  if (obj.IsObject()) {
    rapidjson::Value::ConstMemberIterator mitr, scalemitr;

    // TODO(croot): move the following prop strings to a const somewhere
    std::string fieldProp = "field";
    std::string valueProp = "value";
    std::string scaleProp = "scale";

    bool hasData = false;
    bool hasScale = false, updateScale = false;

    if ((scalemitr = obj.FindMember(scaleProp.c_str())) != obj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_scaleJsonPath, scalemitr->value)) {
        RUNTIME_EX_ASSERT(
            _useScale,
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), obj, "render property \"" + _name + "\" does not support scale references."));
        updateScale = true;
      } else if (_scalePtr && _ctx->getScale(_scalePtr->getName()) != _scalePtr) {
        updateScale = true;
      }

      _scaleJsonPath = objPath.Append(scaleProp.c_str(), scaleProp.length());
      hasScale = true;
    } else {
      // need to clear out the _scaleJsonPath
      _scaleJsonPath = rapidjson::Pointer();
    }

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

        bool dataPtrChanged = dataPtr != _dataPtr;
        if (dataPtrChanged) {
          _unsubscribeFromDataEvent();
        }

        _internalInitFromData(mitr->value.GetString(), dataPtr, hasScale, updateScale);

        if (dataPtrChanged) {
          // setup callbacks for data updates
          auto dataJSONPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(dataPtr);
          CHECK(dataJSONPtr);
          auto cb =
              std::bind(&BaseRenderProperty::_dataRefUpdateCB, this, std::placeholders::_1, std::placeholders::_2);
          _dataRefSubscriptionId = _ctx->subscribeToRefEvent(RefEventType::ALL, dataJSONPtr, cb);
        }
      }

      _fieldJsonPath = objPath.Append(fieldProp.c_str(), fieldProp.length());

      hasData = true;
    } else {
      // need to clear out the field path
      _clearFieldPath();
    }

    if ((mitr = obj.FindMember(valueProp.c_str())) != obj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_valueJsonPath, mitr->value)) {
        _initValueFromJSONObj(mitr->value, hasScale, !hasData);
      }
      _valueJsonPath = objPath.Append(valueProp.c_str(), valueProp.length());

      hasData = true;
    } else {
      _resetValue();
      _valueJsonPath = rapidjson::Pointer();
    }

    if (updateScale) {
      // initialize according to the scale
      _scaleConfigPtr = nullptr;
      _scalePtr = nullptr;
      _initScaleFromJSONObj(scalemitr->value);
      _validateScale();
    }

    if (!hasScale) {
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

      _scaleConfigPtr = nullptr;
      _scalePtr = nullptr;

      if (_vboInitType == VboInitType::FROM_SCALEREF) {
        _resetTypes();
        _validateValue(false);
      }
    }

    _initFromJSONObj(obj);

  } else {
    // need to clear out the object paths
    _clearFieldPath();
    _valueJsonPath = rapidjson::Pointer();
    _scaleJsonPath = rapidjson::Pointer();

    _scaleConfigPtr = nullptr;
    _scalePtr = nullptr;

    _initValueFromJSONObj(obj, false, true);
  }
}

bool BaseRenderProperty::initializeFromData(const std::string& attrName, const QueryDataTableShPtr& dataPtr) {
  return _internalInitFromData(attrName, dataPtr, false, false);
}

bool BaseRenderProperty::_internalInitFromData(const std::string& attrName,
                                               const QueryDataTableShPtr& dataPtr,
                                               const bool hasScale,
                                               const bool updatingScale) {
  RUNTIME_EX_ASSERT(dataPtr != nullptr,
                    std::string(*this) + ": Cannot initialize mark property " + _name +
                        " from data. A valid data reference hasn't been initialized.");

  _dataPtr = dataPtr;
  _vboAttrName = attrName;

  _initBuffers(_dataPtr->getAttributeDataBuffers(attrName));
  _vboInitType = VboInitType::FROM_DATAREF;

  bool inchanged, outchanged;
  std::tie(inchanged, outchanged) = _initTypeFromBuffer(hasScale);
  if (inchanged || outchanged) {
    if (inchanged && !updatingScale && _scalePtr) {
      // need to make sure our scale reference is properly
      // adjusted for a possible new data type
      _updateScalePtr(_scalePtr);
    }
    _setPropsDirty();
  }

  return inchanged || outchanged;
}

int BaseRenderProperty::size(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  if (itr != _perGpuData.end() && itr->second.vbo) {
    auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(_dataPtr);
    return itr->second.vbo->numVertices(dataPtr ? dataPtr->getVboQueryDataLayout() : nullptr);
  }
  return 0;
}

bool BaseRenderProperty::hasVboPtr() const {
  for (auto& itr : _perGpuData) {
    if (itr.second.vbo != nullptr) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasVboPtr(const GpuId& gpuId) const {
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

bool BaseRenderProperty::hasUboPtr() const {
  for (auto& itr : _perGpuData) {
    if (itr.second.ubo != nullptr) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasUboPtr(const GpuId& gpuId) const {
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
  if (_scaleConfigPtr) {
    return _scaleConfigPtr->getRangeTypeGL()->glslType();
  } else if (_scalePtr) {
    return _scalePtr->getRangeTypeGL()->glslType();
  }

  RUNTIME_EX_ASSERT(
      _outType != nullptr,
      std::string(*this) + " getOutGLSLType(): output type for mark property \"" + _name + "\" is uninitialized.");

  return (_outType->glslType());
}

const ::Rendering::GL::TypeGLShPtr& BaseRenderProperty::getOutTypeGL() const {
  if (_scaleConfigPtr) {
    return _scaleConfigPtr->getRangeTypeGL();
  } else if (_scalePtr) {
    return _scalePtr->getRangeTypeGL();
  }

  RUNTIME_EX_ASSERT(_outType != nullptr,
                    std::string(*this) + " getOutTypeGL(): input type for \"" + _name + "\" is uninitialized.");

  return _outType;
}

void BaseRenderProperty::addToVboAttrMap(const GpuId& gpuId,
                                         ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap& attrMap) const {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr->second.vbo != nullptr,
                    std::string(*this) +
                        " addToVboAttrMap(): A vertex buffer is not defined. Cannot add vbo attrs to "
                        "vbo->shader attr map.");

  ::Rendering::GL::Resources::GLVertexBufferShPtr glVbo = itr->second.vbo->getGLVertexBufferPtr();
  auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(_dataPtr);
  auto pair = std::make_pair(glVbo, (dataPtr ? dataPtr->getVboGLBufferLayout() : nullptr));
  auto attrMapItr = attrMap.find(pair);
  if (attrMapItr == attrMap.end()) {
    attrMap.insert({pair, {{_vboAttrName, _name}}});
  } else {
    attrMapItr->second.push_back({_vboAttrName, _name});
  }
}

void BaseRenderProperty::bindUniformScaleToRenderer(::Rendering::GL::Resources::GLShader* activeShader) const {
  if (_decimalExpScale > 0 && _outType) {
    if (_ctx->supportsInt64()) {
      activeShader->setUniformAttribute<uint64_t>(_name + "_ExpScale", _decimalExpScale);
    } else {
      activeShader->setUniformAttribute<unsigned int>(_name + "_ExpScale", static_cast<unsigned int>(_decimalExpScale));
    }
  }
}

void BaseRenderProperty::initGpuResources(const QueryRendererContext* ctx,
                                          const std::set<GpuId>& usedGpus,
                                          const std::set<GpuId>& unusedGpus) {
  auto qrmPerGpuData = ctx->getRootGpuCache()->perGpuData;

  bool reset = false;
  for (auto gpuId : usedGpus) {
    if (_perGpuData.find(gpuId) == _perGpuData.end()) {
      auto qrmItr = qrmPerGpuData->find(gpuId);
      CHECK(qrmItr != qrmPerGpuData->end());
      auto rtn =
          _perGpuData.emplace(std::piecewise_construct, std::forward_as_tuple(gpuId), std::forward_as_tuple(*qrmItr));
      auto& gpuData = rtn.first->second;
      if (_dataPtr) {
        if (_dataPtr->hasAttribute(_vboAttrName)) {
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
              CHECK(false) << "Unsupported resource type " << bufferPtr->getGLResourceType()
                           << " for render properties.";
          }
        } else {
          reset = true;
        }
      }
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }

  if (reset) {
    // reaching this code likely means that there was an attempt to initGpuResources
    // before a complete update was performed, so clearing everything out first.
    LOG(WARNING) << "Possible sync issue when updating render property " << _printInfo();
    _clearFieldPath();
    _resetTypes();
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

  initGpuResources(_ctx.get(), usedGpus, unusedGpus);
  CHECK(bufferMap.size() == _perGpuData.size());

  if (!bufferMap.size()) {
    return;
  }

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

void BaseRenderProperty::_dataRefUpdateCB(RefEventType refEventType, const RefObjShPtr& refObjPtr) {
  auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTable>(refObjPtr);
  CHECK(dataPtr);
  switch (refEventType) {
    case RefEventType::UPDATE:
      CHECK(dataPtr == _dataPtr);
    // pass thru to the REPLACE code
    case RefEventType::REPLACE:
      CHECK(_vboAttrName.size());
      if (_internalInitFromData(_vboAttrName, dataPtr, _scaleConfigPtr != nullptr || _scalePtr != nullptr, false)) {
        _setShaderDirty();
      }
      break;
    case RefEventType::REMOVE:
      THROW_RUNTIME_EX(std::string(*this) + ": Error, data table " + refObjPtr->getName() +
                       " has been removed but is still being referenced by this render property.")
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " + std::to_string(static_cast<int>(refEventType)) +
                       " isn't currently supported for data reference updates.");
      break;
  }
}

void BaseRenderProperty::_clearFieldPath() {
  _fieldJsonPath = rapidjson::Pointer();
  _clearDataPtr();
  _vboAttrName = "";
  _decimalExpScale = 0;
}

void BaseRenderProperty::_clearDataPtr() {
  _unsubscribeFromDataEvent();
  _dataPtr = nullptr;
}

bool BaseRenderProperty::_checkAccumulator(const ScaleShPtr& scalePtr) {
  bool scaleAccumulation = scalePtr->hasAccumulator();
  RUNTIME_EX_ASSERT(_allowsAccumulatorScale || !scaleAccumulation,
                    std::string(*this) + " The scale \"" + scalePtr->getName() +
                        "\" is an accumulator scale but this mark property doesn't accept accumulator scales.");
  return scaleAccumulation;
}

void BaseRenderProperty::_setAccumulatorFromScale(const ScaleShPtr& scalePtr) {
  if (scalePtr && scalePtr->hasAccumulator()) {
    // TODO(croot): what if there are two accumulator scales in the
    // same mark properties?
    RUNTIME_EX_ASSERT(
        !_prntMark->hasAccumulator() || scalePtr->getName() == _prntMark->getAccumulatorScaleName(),
        std::string(*_prntMark) +
            " has more that one accumulator scale attached to it. There can only be one accumulator scale per mark.");
    _prntMark->setAccumulatorScale(scalePtr, _scaleConfigPtr);
  }
}

void BaseRenderProperty::_clearAccumulatorFromScale(const ScaleShPtr& scalePtr) {
  if (_prntMark && scalePtr && scalePtr->getName() == _prntMark->getAccumulatorScaleName()) {
    // TODO(croot): what if there are two accumulator scales in the
    // same mark properties?
    _prntMark->clearAccumulatorScale();
  }
}

void BaseRenderProperty::_unsubscribeFromScaleEvent(const ScaleShPtr& scalePtr) {
  if (scalePtr) {
    _ctx->unsubscribeFromRefEvent(RefEventType::ALL, scalePtr, _scaleRefSubscriptionId);
  }
}

void BaseRenderProperty::_unsubscribeFromDataEvent() {
  if (_dataPtr) {
    auto dataJSONPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(_dataPtr);
    CHECK(dataJSONPtr);
    _ctx->unsubscribeFromRefEvent(RefEventType::ALL, dataJSONPtr, _dataRefSubscriptionId);
  }
}

void BaseRenderProperty::_updateScalePtr(const ScaleShPtr& scalePtr) {
  _scalePtr = scalePtr;
  _setAccumulatorFromScale(scalePtr);
}

template <>
RenderProperty<ColorUnion, 1>::RenderProperty(BaseMark* prntMark,
                                              const std::string& name,
                                              const QueryRendererContextShPtr& ctx,
                                              bool useScale,
                                              bool flexibleType,
                                              bool allowsAccumulatorScale,
                                              bool allowsNonColorStrings)
    : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType, allowsAccumulatorScale, allowsNonColorStrings),
      _mult(),
      _offset() {
  _resetValue();

  auto rootGpuCache = ctx->getRootGpuCache();
  CHECK(rootGpuCache);
  _inType = ColorUnion::getTypeGLPtr(rootGpuCache->supportedExtensions);
  _outType = ColorUnion::getTypeGLPtr(rootGpuCache->supportedExtensions);
}

template <>
bool RenderProperty<ColorUnion, 1>::_resetTypes() {
  if (_vboInitType != VboInitType::FROM_VALUE) {
    auto rootGpuCache = _ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    auto newInType = ColorUnion::getTypeGLPtr(rootGpuCache->supportedExtensions);
    auto newOutType = ColorUnion::getTypeGLPtr(rootGpuCache->supportedExtensions);

    for (auto& itr : _perGpuData) {
      itr.second.vbo = nullptr;
      itr.second.ubo = nullptr;
    }

    if (_vboInitType == VboInitType::FROM_DATAREF) {
      _setPropsDirty();
    }
    _vboInitType = VboInitType::FROM_VALUE;

    if (!_inType || !_outType || (*_inType) != (*newInType) || (*_outType) != (*newOutType)) {
      _setShaderDirty();
      _inType = newInType;
      _outType = newOutType;
      return true;
    }
  }
  return false;
}

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_resetValue() {
  _uniformVal.set(QueryDataType::COLOR, ColorUnion());
}

template <>
ColorUnion RenderProperty<ColorUnion, 1>::getUniformValue() const {
  if (_uniformVal.getType() == QueryDataType::COLOR) {
    return _uniformVal.getColorRef<ColorUnion>();
  }

  THROW_RUNTIME_EX(std::string(*this) + " Cannot get the uniform value of the color property \"" + _name +
                   "\". The property is referencing a scale with a domain that is not a color type and therefore its "
                   "uniform value is not a color.");

  return ColorUnion();
}

template <>
void RenderProperty<ColorUnion, 1>::bindUniformToRenderer(GLShader* activeShader,
                                                          const std::string& uniformAttrName) const {
  if (_inType) {
    if (_vboInitType == VboInitType::FROM_SCALEREF) {
      if (dynamic_cast<::Rendering::GL::TypeGL<int, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<int>(uniformAttrName, _uniformVal.getVal<int>());
      } else if (dynamic_cast<::Rendering::GL::TypeGL<unsigned int, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<unsigned int>(uniformAttrName, _uniformVal.getVal<unsigned int>());
      } else if (dynamic_cast<::Rendering::GL::TypeGL<float, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<float>(uniformAttrName, _uniformVal.getVal<float>());
      } else if (dynamic_cast<::Rendering::GL::TypeGL<double, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<double>(uniformAttrName, _uniformVal.getVal<double>());
      } else if (dynamic_cast<::Rendering::GL::TypeGL<int64_t, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<int64_t>(uniformAttrName, _uniformVal.getVal<int64_t>());
      } else if (dynamic_cast<::Rendering::GL::TypeGL<uint64_t, 1>*>(_inType.get())) {
        activeShader->setUniformAttribute<uint64_t>(uniformAttrName, _uniformVal.getVal<uint64_t>());
      } else {
        CHECK(false) << "Unsupported type: " << _uniformVal.getType() << " " << std::string(*_inType);
      }
    } else {
      activeShader->setUniformAttribute<std::array<float, 4>>(uniformAttrName,
                                                              _uniformVal.getColorRef<ColorUnion>().getColorArrayRef());
    }
  }
}

void ColorRenderProperty::_initFromJSONObj(const rapidjson::Value& obj) {
  // TODO: what about offsets / mults for colors?

  // NOTE: this function is called by the base class during initialization, so we
  // know at this point that obj is an json Object
  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot): move the following prop strings to a const somewhere
  std::string colorSpaceProp = "colorSpace";

  if ((mitr = obj.FindMember(colorSpaceProp.c_str())) != obj.MemberEnd()) {
    auto colorType = ::Rendering::Colors::ColorType::RGBA;

    RUNTIME_EX_ASSERT(
        mitr->value.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), mitr->value, "\"" + colorSpaceProp + "\" color property must be a string"));

    auto colorString = std::string(mitr->value.GetString());
    try {
      colorType = ::Rendering::Colors::getColorTypeFromColorPrefix(colorString);
    } catch (::Rendering::RenderError& err) {
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(),
          mitr->value,
          "The string \"" + colorString + "\" is not a valid color space. The supported color spaces are: [" +
              boost::algorithm::join(::Rendering::Colors::getAllColorPrefixes(), ",") + "]"));
    }

    if (_vboInitType == VboInitType::FROM_VALUE) {
      auto& uniformColor = _uniformVal.getColorRef<ColorUnion>();
      if (_colorInitType == ColorInitType::FROM_PACKED_UINT) {
        // re-initialize the color based on this new space
        if (colorType != uniformColor.getType()) {
          auto packedColor = uniformColor.getPackedColor();
          try {
            uniformColor.initFromPackedUInt(packedColor, colorType);
          } catch (::Rendering::RenderError& err) {
            THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                                  obj,
                                                                  "The packed color " + std::to_string(packedColor) +
                                                                      " is not a valid packed color. " +
                                                                      err.getDetails()));
          }
        }
      } else {
        RUNTIME_EX_ASSERT(
            uniformColor.getType() == colorType,
            RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                 obj,
                                                 "The color space \"" + colorString + "\" does not match the " +
                                                     ::Rendering::Colors::to_string(uniformColor.getType()) +
                                                     " color space defined by value"));
      }
    } else if (_vboInitType == VboInitType::FROM_DATAREF) {
      CHECK(_inType);

      if (ColorUnion::isPackedTypeGL(_inType)) {
        // verify that the color type is valid
        RUNTIME_EX_ASSERT(
            ColorUnion::isValidPackedType(colorType),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(),
                obj,
                "The data attribute \"" + _vboAttrName + "\" is being used as a packed color, but the color space \"" +
                    colorString + "\" is not a valid packed color type. The supported packed color types are [" +
                    boost::algorithm::join(ColorUnion::getPackedColorPrefixes(), ",") + "]"));
      }

      // TODO(croot)
      _uniformVal.set(QueryDataType::COLOR, ColorUnion(0.0f, 0.0f, 0.0f, 1.0f, colorType));
    }
  }
}

void ColorRenderProperty::_validateValue(const bool hasScale) {
  if (!hasScale) {
    auto type = _uniformVal.getType();
    RUNTIME_EX_ASSERT(
        type == QueryDataType::COLOR || type == QueryDataType::INT || type == QueryDataType::UINT,
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(),
            "value for color property \"" + _name + "\" must be a string or a color packed into an int/uint."));

    if (type == QueryDataType::INT || type == QueryDataType::UINT) {
      ColorUnion color;
      auto num = _uniformVal.getVal<uint32_t>();
      try {
        color.initFromPackedUInt(num);
      } catch (::Rendering::RenderError& err) {
        THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(),
            "The packed color " + std::to_string(num) + " is not a valid packed color. " + err.what()));
      }
      _colorInitType = ColorInitType::FROM_PACKED_UINT;
      _uniformVal.set(QueryDataType::COLOR, color);
    }
  }
}

template <typename T>
static ScaleRefShPtr createColorScaleRef(const ::Rendering::Colors::ColorType colorType,
                                         const QueryRendererContextShPtr& ctx,
                                         const ScaleShPtr& scalePtr,
                                         BaseRenderProperty* rndrProp) {
  switch (colorType) {
    case ::Rendering::Colors::ColorType::RGBA:
      return ScaleRefShPtr(new ScaleRef<T, ::Rendering::Colors::ColorRGBA>(ctx, scalePtr, rndrProp));
      break;
    case ::Rendering::Colors::ColorType::HSL:
      return ScaleRefShPtr(new ScaleRef<T, ::Rendering::Colors::ColorHSL>(ctx, scalePtr, rndrProp));
      break;
    case ::Rendering::Colors::ColorType::LAB:
      return ScaleRefShPtr(new ScaleRef<T, ::Rendering::Colors::ColorLAB>(ctx, scalePtr, rndrProp));
      break;
    case ::Rendering::Colors::ColorType::HCL:
      return ScaleRefShPtr(new ScaleRef<T, ::Rendering::Colors::ColorHCL>(ctx, scalePtr, rndrProp));
      break;
    default:
      THROW_RUNTIME_EX("Unsupported color type: " + std::to_string(static_cast<int>(colorType)) +
                       ". Cannot create a color scale ref.");
  }

  return nullptr;
}

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_updateScalePtr(const ScaleShPtr& scalePtr) {
  bool scaleAccumulation = _checkAccumulator(scalePtr);
  bool scaleDensityAccumulation = (scaleAccumulation && scalePtr->getAccumulatorType() == AccumulatorType::DENSITY);

  const ::Rendering::GL::TypeGLShPtr* inTypeToUse = &_inType;
  if (*inTypeToUse) {
    if (!_scaleConfigPtr) {
      _setShaderDirty();
    }

    if (_vboInitType != VboInitType::FROM_DATAREF) {
      inTypeToUse = &(scalePtr->getDomainTypeGL());
      if (_inType != *inTypeToUse && (*_inType) != (**inTypeToUse)) {
        _setShaderDirty();
      }
      _inType = *inTypeToUse;
      _vboInitType = VboInitType::FROM_SCALEREF;
      _validateValue(true);
    }

    auto* rangeData = scalePtr->getRangeData();

    ::Rendering::Colors::ColorType colorType = ::Rendering::Colors::ColorType::RGBA;
    if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>*>(rangeData)) {
      colorType = ::Rendering::Colors::ColorType::RGBA;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>*>(rangeData)) {
      colorType = ::Rendering::Colors::ColorType::HSL;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>*>(rangeData)) {
      colorType = ::Rendering::Colors::ColorType::LAB;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>*>(rangeData)) {
      colorType = ::Rendering::Colors::ColorType::HCL;
    } else {
      THROW_RUNTIME_EX(std::string(*this) +
                       ": Trying to add a color scale with an unsupported color type for its range.");
    }

    if (isDecimal()) {
      _scaleConfigPtr = createColorScaleRef<double>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<unsigned int, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<unsigned int>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<int, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<int>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<float, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<float>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<double, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<double>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<int64_t, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<int64_t>(colorType, _ctx, scalePtr, this);
    } else if (dynamic_cast<::Rendering::GL::TypeGL<uint64_t, 1>*>((*inTypeToUse).get())) {
      _scaleConfigPtr = createColorScaleRef<uint64_t>(colorType, _ctx, scalePtr, this);
    } else {
      RUNTIME_EX_ASSERT(scaleDensityAccumulation,
                        std::string(*this) + ": Scale domain with shader type \"" +
                            scalePtr->getDomainTypeGL()->glslType() + "\" and data with shader type \"" +
                            (*inTypeToUse)->glslType() + "\" are not supported to work together.");

      switch (scalePtr->getDomainDataType()) {
        case QueryDataType::DOUBLE:
          _scaleConfigPtr = createColorScaleRef<double>(colorType, _ctx, scalePtr, this);
          break;
        default:
          THROW_RUNTIME_EX(std::string(*this) + ": Unsupported density accumulator scale with domain of type " +
                           to_string(scalePtr->getDomainDataType()));
      }
    }
  } else {
    if (_scaleConfigPtr) {
      _setShaderDirty();
    }
    _scaleConfigPtr = nullptr;
  }

  BaseRenderProperty::_updateScalePtr(scalePtr);
  _setupScaleCB(scalePtr);
}

template <>
void RenderProperty<ColorUnion, 1>::_validateType(const ::Rendering::GL::TypeGLShPtr& type) {
  auto& uniformColor = _uniformVal.getColorRef<ColorUnion>();
  RUNTIME_EX_ASSERT(uniformColor.isValidTypeGL(type),
                    std::string(*this) + ": The vertex buffer type " + (type ? std::string(*type) : "\"null\"") +
                        " is not a valid color type for mark property \"" + _name + "\".");
}

template <>
void RenderProperty<ColorUnion, 1>::_validateScale() {
  RUNTIME_EX_ASSERT(_scaleConfigPtr != nullptr || _scalePtr != nullptr,
                    std::string(*this) + ": Cannot verify scale for mark property \"" + _name +
                        "\". Scale reference is uninitialized.");

  // colors need to be a specific type
  auto* rangeData = (_scaleConfigPtr ? _scaleConfigPtr->getRangeData() : _scalePtr->getRangeData());

  RUNTIME_EX_ASSERT(dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>*>(rangeData) ||
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>*>(rangeData) ||
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>*>(rangeData) ||
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>*>(rangeData),
                    std::string(*this) + ": The scale \"" +
                        (_scaleConfigPtr ? _scaleConfigPtr->getName() : _scalePtr->getName()) +
                        " does not have a range with a valid color type for mark property \"" + _name + "\".");
}

::Rendering::Colors::ColorType ColorRenderProperty::getColorType() const {
  if (_scaleConfigPtr || _scalePtr) {
    auto* rangeData = (_scaleConfigPtr ? _scaleConfigPtr->getRangeData() : _scalePtr->getRangeData());

    if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>*>(rangeData)) {
      return ::Rendering::Colors::ColorType::RGBA;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>*>(rangeData)) {
      return ::Rendering::Colors::ColorType::HSL;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>*>(rangeData)) {
      return ::Rendering::Colors::ColorType::LAB;
    } else if (dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>*>(rangeData)) {
      return ::Rendering::Colors::ColorType::HCL;
    } else {
      THROW_RUNTIME_EX(std::string(*this) +
                       ": Trying to add a color scale with an unsupported color type for its range.");
    }
  } else if (_dataPtr) {
    auto embeddedData = std::dynamic_pointer_cast<DataTable>(_dataPtr);
    if (embeddedData) {
      auto column = embeddedData->getColumn(_vboAttrName);
      if (std::dynamic_pointer_cast<TDataColumn<::Rendering::Colors::ColorRGBA>>(column)) {
        return ::Rendering::Colors::ColorType::RGBA;
      } else if (std::dynamic_pointer_cast<TDataColumn<::Rendering::Colors::ColorHSL>>(column)) {
        return ::Rendering::Colors::ColorType::HSL;
      } else if (std::dynamic_pointer_cast<TDataColumn<::Rendering::Colors::ColorLAB>>(column)) {
        return ::Rendering::Colors::ColorType::LAB;
      } else if (std::dynamic_pointer_cast<TDataColumn<::Rendering::Colors::ColorHCL>>(column)) {
        return ::Rendering::Colors::ColorType::HCL;
      } else if (std::dynamic_pointer_cast<TDataColumn<int>>(column) ||
                 std::dynamic_pointer_cast<TDataColumn<unsigned int>>(column)) {
        // The color is packed into an int here, so the type of the packed color will be determined
        // by the uniform, which holds the type of the color
        auto& uniformColor = _uniformVal.getColorRef<ColorUnion>();
        return uniformColor.getType();
      } else {
        THROW_RUNTIME_EX(std::string(*this) +
                         ": Trying to use a color embedded in the data with an unsupported color type.");
      }
    }

    // NOTE: if a color is provided via a SQL query, it is defined by either a packed uint, or a
    // vec4f. In either case, we don't know what color space this color is defined in. This is
    // determined in Vega like so:
    //
    // fillColor: {
    //     field: "color",    // the name of the attr from the resulting sql
    //     colorSpace: "rgb"  // the space the color is defined in, in this case "rgb".
    //                        // Can also be "hsl", "lab", or "hcl"
    // },
    //
    // So the actual space will be determined in the _uniformVal that will be set to the
    // appropriate space already.
  }

  auto& uniformColor = _uniformVal.getColorRef<ColorUnion>();
  return uniformColor.getType();
}

bool ColorRenderProperty::isColorPacked() const {
  if (_scaleConfigPtr || _scalePtr) {
    // TODO(croot): support packed colors in scales?
    return false;
  }
  CHECK(_inType);
  return ColorUnion::isPackedTypeGL(_inType);
}

void EnumRenderProperty::_validateValue(const bool hasScale) {
  int val = 0;
  if (_uniformVal.getType() == QueryDataType::STRING) {
    RUNTIME_EX_ASSERT(_stringConvertFunc != nullptr, std::string(*this) + ": Enum property does not support strings.");

    auto strval = _uniformVal.getStringVal();
    val = _stringConvertFunc(strval);

    RUNTIME_EX_ASSERT(val >= 0, std::string(*this) + ": Enum value " + strval + " is not supported.");
  } else {
    val = _uniformVal.getVal<int>();
  }

  _uniformVal.set(QueryDataType::INT, val);
}

}  // namespace QueryRenderer
