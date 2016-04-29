#ifndef QUERYRENDERER_MARKS_RENDERPROPERTY_H_
#define QUERYRENDERER_MARKS_RENDERPROPERTY_H_

#include "Types.h"
#include "../Types.h"
#include "../Data/Types.h"
#include "../Scales/Types.h"
#include "../Scales/Scale.h"
#include "../Scales/ScaleRef.h"
#include "../QueryRenderManager.h"
#include "../QueryRendererContext.h"

#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Renderer/GL/Resources/GLVertexArray.h>

#include <set>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

class BaseRenderProperty {
 public:
  BaseRenderProperty(BaseMark* prntMark,
                     const std::string& name,
                     const QueryRendererContextShPtr& ctx,
                     bool useScale = true,
                     bool flexibleType = true)
      : _prntMark(prntMark),
        _name(name),
        _useScale(useScale),
        _vboAttrName(""),
        _perGpuData(),
        _vboInitType(VboInitType::UNDEFINED),
        _dataPtr(nullptr),
        _ctx(ctx),
        _inType(nullptr),
        _outType(nullptr),
        _scaleConfigPtr(nullptr),
        _flexibleType(flexibleType) {}

  virtual ~BaseRenderProperty() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryDataTableShPtr& dataPtr);
  void initializeFromData(const std::string& attrName, const QueryDataTableShPtr& dataPtr);

  int size(const GpuId& gpuId) const;

  std::string getName() const { return _name; }

  std::string getGLSLFunc() const { return "get" + _name; }

  std::string getInGLSLName() const { return "inT" + _name; }

  std::string getInGLSLType() const;

  std::string getOutGLSLName() const { return "outT" + _name; }

  std::string getOutGLSLType() const;

  bool hasVboPtr();
  bool hasVboPtr(const GpuId& gpuId);

  bool hasUboPtr();
  bool hasUboPtr(const GpuId& gpuId);

  QueryVertexBufferShPtr getVboPtr(const GpuId& gpuId) const;
  QueryVertexBufferShPtr getVboPtr() const;

  QueryUniformBufferShPtr getUboPtr(const GpuId& gpuId) const;
  QueryUniformBufferShPtr getUboPtr() const;

  bool usesScaleConfig() { return (_scaleConfigPtr != nullptr); }

  const ScaleRefShPtr& getScaleReference() { return _scaleConfigPtr; }

  void addToVboAttrMap(const GpuId& gpuId, ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) const;

  virtual void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                     const std::string& uniformAttrName) const = 0;

  std::string getDataColumnName() { return _vboAttrName; }
  const QueryDataTableShPtr& getDataTablePtr() { return _dataPtr; }

  void initGpuResources(const QueryRendererContext* ctx,
                        const std::set<GpuId>& usedGpus,
                        const std::set<GpuId>& unusedGpus);

  virtual operator std::string() const = 0;

 protected:
  enum class VboInitType { FROM_VALUE = 0, FROM_DATAREF, UNDEFINED };

  BaseMark* _prntMark;
  std::string _name;
  bool _useScale;

  std::string _vboAttrName;

  struct PerGpuData : BasePerGpuData {
    QueryVertexBufferShPtr vbo;
    QueryUniformBufferShPtr ubo;

    PerGpuData() : BasePerGpuData(), vbo(nullptr), ubo(nullptr) {}
    explicit PerGpuData(const BasePerGpuData& perGpuData,
                        const QueryVertexBufferShPtr& vbo = nullptr,
                        const QueryUniformBufferShPtr& ubo = nullptr)
        : BasePerGpuData(perGpuData), vbo(vbo), ubo(ubo) {}
    PerGpuData(const PerGpuData& data) : BasePerGpuData(data), vbo(data.vbo), ubo(data.ubo) {}
    PerGpuData(PerGpuData&& data)
        : BasePerGpuData(std::move(data)), vbo(std::move(data.vbo)), ubo(std::move(data.ubo)) {}

    ~PerGpuData() {
      // need to make active to properly delete GL resources per-gpu
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;

  VboInitType _vboInitType;
  QueryDataTableShPtr _dataPtr;

  QueryRendererContextShPtr _ctx;

  ::Rendering::GL::TypeGLShPtr _inType;
  ::Rendering::GL::TypeGLShPtr _outType;

  // ScaleShPtr _scaleConfigPtr;
  ScaleRefShPtr _scaleConfigPtr;

  bool _flexibleType;

  rapidjson::Pointer _fieldJsonPath;
  rapidjson::Pointer _valueJsonPath;
  rapidjson::Pointer _scaleJsonPath;

  // TODO(croot): redefining RefEventCallback here, so we should have a "types"
  // header file somewhere
  typedef std::function<void(RefEventType, const ScaleShPtr&)> RefEventCallback;
  RefEventCallback _scaleRefSubscriptionCB;

  virtual void _initScaleFromJSONObj(const rapidjson::Value& obj) = 0;
  virtual void _initFromJSONObj(const rapidjson::Value& obj) {}
  virtual void _initValueFromJSONObj(const rapidjson::Value& obj) = 0;
  virtual void _initTypeFromBuffer() = 0;
  virtual void _verifyScale() = 0;
  virtual void _scaleRefUpdateCB(RefEventType refEventType, const ScaleShPtr& scalePtr) = 0;

  void _setShaderDirty();
  void _setPropsDirty();

  std::string _printInfo() const;

 private:
  std::set<GpuId> _initUnusedGpus(const std::map<GpuId, QueryBufferShPtr>& bufferMap);
  std::set<GpuId> _initUnusedGpus(const std::set<GpuId>& usedGpus);
  void _initBuffers(const std::map<GpuId, QueryBufferShPtr>& bufferMap);
};

template <typename T, int numComponents = 1>
class RenderProperty : public BaseRenderProperty {
 public:
  RenderProperty(BaseMark* prntMark,
                 const std::string& name,
                 const QueryRendererContextShPtr& ctx,
                 bool useScale = true,
                 bool flexibleType = true)
      : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType), _mult(), _offset() {
    _inType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
    _outType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
  }

  ~RenderProperty() {
    if (_scaleConfigPtr) {
      _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
    }
  }

  void initializeValue(const T& val) {
    // TODO: this is a public function.. should I protect from already existing data?

    if (_vboInitType != VboInitType::FROM_VALUE) {
      _inType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());
      _outType.reset(new ::Rendering::GL::TypeGL<T, numComponents>());

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

  void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                             const std::string& uniformAttrName) const {
    // TODO(croot): deal with numComponents here by using a vector instead?
    activeShader->setUniformAttribute<T>(uniformAttrName, _uniformVal);
  }

  operator std::string() const final {
    return "RenderProperty<" + std::string(typeid(T).name()) + ", " + std::to_string(numComponents) + "> " +
           _printInfo();
  }

 private:
  T _mult;
  T _offset;
  T _uniformVal;

  void _initScaleFromJSONObj(const rapidjson::Value& obj) {
    RUNTIME_EX_ASSERT(
        obj.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), obj, "scale reference for mark property \"" + _name + "\" must be a string."));

    RUNTIME_EX_ASSERT(_ctx != nullptr && _scaleConfigPtr == nullptr,
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(),
                          obj,
                          "cannot initialize mark property \"" + _name +
                              "\" from a scale. The context is uninitialized or a scale is already being referenced."));

    // TODO(croot): We probably need a better way to deal with types. We've got an _inType that is either defined
    // by an incoming data reference or an explicit value (or set of values). The latter is easy. We already
    // have the type in T/numComponents of the template. But the data reference is trickier.

    ScaleShPtr scalePtr = _ctx->getScale(obj.GetString());
    RUNTIME_EX_ASSERT(scalePtr != nullptr,
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(),
                          obj,
                          "the scale \"" + std::string(obj.GetString()) + "\" does not exist in the json."));

    _updateScalePtr(scalePtr);
  }

  void _updateScalePtr(const ScaleShPtr& scalePtr) {
    if (!_scaleConfigPtr) {
      _setShaderDirty();
    } else {
      ScaleShPtr prevPtr = _scaleConfigPtr->getScalePtr();
      if (!prevPtr || scalePtr.get() != prevPtr.get()) {
        _setShaderDirty();
        _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
      }
    }

    if (dynamic_cast<::Rendering::GL::TypeGL<unsigned int, 1>*>(_inType.get())) {
      _scaleConfigPtr.reset(new ScaleRef<unsigned int, T>(_ctx, scalePtr, this));
    } else if (dynamic_cast<::Rendering::GL::TypeGL<int, 1>*>(_inType.get())) {
      _scaleConfigPtr.reset(new ScaleRef<int, T>(_ctx, scalePtr, this));
    } else if (dynamic_cast<::Rendering::GL::TypeGL<float, 1>*>(_inType.get())) {
      _scaleConfigPtr.reset(new ScaleRef<float, T>(_ctx, scalePtr, this));
    } else if (dynamic_cast<::Rendering::GL::TypeGL<double, 1>*>(_inType.get())) {
      _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
    } else {
      THROW_RUNTIME_EX(std::string(*this) + ": Scale domain with shader type \"" +
                       scalePtr->getDomainTypeGL()->glslType() + "\" and data with shader type \"" +
                       _inType->glslType() + "\" are not supported to work together");
    }

    // setup callbacks for scale updates
    _scaleRefSubscriptionCB = std::bind(
        &RenderProperty<T, numComponents>::_scaleRefUpdateCB, this, std::placeholders::_1, std::placeholders::_2);
    _ctx->subscribeToRefEvent(RefEventType::ALL, scalePtr, _scaleRefSubscriptionCB);
  }

  void _initFromJSONObj(const rapidjson::Value& obj) {
    // this is internally called at the appropriate time from
    // the base class's initialization function, so there's
    // no need to check that obj is valid since that should've
    // already been done.
    rapidjson::Value::ConstMemberIterator mitr;

    // TODO(croot) - fill this in with mult/offset json path caches.
    // These values will ultimately act as uniforms, so modifications
    // to them should not force a full shader build/compile.
    // if ((mitr = obj.FindMember("mult")) != obj.MemberEnd()) {
    //   _mult = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
    // }

    // if ((mitr = obj.FindMember("offset")) != obj.MemberEnd()) {
    //   _offset = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
    // }
  }

  void _initValueFromJSONObj(const rapidjson::Value& obj) {
    T val = RapidJSONUtils::getNumValFromJSONObj<T>(obj);

    initializeValue(val);
  }

  void _initTypeFromBuffer() {
    auto itr = _perGpuData.begin();
    CHECK(itr != _perGpuData.end());

    QueryBufferShPtr bufToUse = (itr->second.vbo ? std::dynamic_pointer_cast<QueryBuffer>(itr->second.vbo)
                                                 : std::dynamic_pointer_cast<QueryBuffer>(itr->second.ubo));
    RUNTIME_EX_ASSERT(bufToUse != nullptr,
                      std::string(*this) +
                          ": Vertex/uniform buffer is uninitialized. Cannot initialize type for mark property \"" +
                          _name + "\".");

    ::Rendering::GL::TypeGLShPtr vboType = bufToUse->getAttributeTypeGL(_vboAttrName);

    if (_flexibleType) {
      _inType = vboType;
      _outType = vboType;
    } else {
      // the render property is rigid, meaning it cannot accept
      // different types. So verify that the type of the attribute
      // in the vbo is appropriate.
      RUNTIME_EX_ASSERT((*_outType) == (*vboType),
                        std::string(*this) +
                            ": The vertex buffer type does not match the output type for mark property \"" + _name +
                            "\".");
    }
  }

  void _verifyScale() {}

  void _scaleRefUpdateCB(RefEventType refEventType, const ScaleShPtr& scalePtr) {
    switch (refEventType) {
      case RefEventType::UPDATE:
        _scaleConfigPtr->updateScaleRef(scalePtr);

        if (scalePtr->hasClampChanged() || scalePtr->hasDomainChangedInSize() || scalePtr->hasRangeChangedInSize()) {
          _setShaderDirty();
        }
        break;
      case RefEventType::REPLACE:
        _updateScalePtr(scalePtr);
        break;
      case RefEventType::REMOVE:
        break;
      default:
        THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " + std::to_string(static_cast<int>(refEventType)) +
                         " isn't currently supported for scale reference updates.");
        break;
    }
  }
};

template <>
RenderProperty<::Rendering::Objects::ColorRGBA, 1>::RenderProperty(
    BaseMark* prntMark,
    const std::string& name,
    const QueryRendererContextShPtr& ctx,
    bool useScale,
    // TODO(croot): perhaps remove flexibleType? it ultimately is saying
    // whether or not we can use a scale, right, which we have defined
    // with useScale?
    bool flexibleType);

template <>
void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::initializeValue(const ::Rendering::Objects::ColorRGBA& val);

template <>
void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::bindUniformToRenderer(
    ::Rendering::GL::Resources::GLShader* activeShader,
    const std::string& uniformAttrName) const;

template <>
void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj);

template <>
void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj);

// template <>
// void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::_initTypeFromVbo();

template <>
void RenderProperty<::Rendering::Objects::ColorRGBA, 1>::_verifyScale();

}  // namespace QueryRendererclass BaseRenderProperty {

#endif  // QUERYRENDERER_MARKS_RENDERPROPERTY_H_
