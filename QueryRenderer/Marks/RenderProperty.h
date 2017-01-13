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

#include <Rendering/Colors/ColorUnion.h>
#include <Rendering/Renderer/GL/Resources/GLVertexArray.h>

#include <set>
#include <boost/any.hpp>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

class BaseRenderProperty {
 public:
  BaseRenderProperty(BaseMark* prntMark,
                     const std::string& name,
                     const QueryRendererContextShPtr& ctx,
                     bool useScale = true,
                     bool flexibleType = true,
                     bool allowsAccumulatorScale = false,
                     bool allowsNonColorStrings = false)
      : _prntMark(prntMark),
        _name(name),
        _useScale(useScale),
        _allowsAccumulatorScale(allowsAccumulatorScale),
        _allowsNonColorStrings(allowsNonColorStrings),
        _decimalExpScale(0),
        _vboAttrName(""),
        _perGpuData(),
        _vboInitType(VboInitType::UNDEFINED),
        _dataPtr(nullptr),
        _ctx(ctx),
        _inType(nullptr),
        _outType(nullptr),
        _scaleConfigPtr(nullptr),
        _flexibleType(flexibleType),
        _scaleRefSubscriptionId(0),
        _dataRefSubscriptionId(0) {}

  virtual ~BaseRenderProperty();

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryDataTableShPtr& dataPtr);

  bool initializeFromData(const std::string& attrName, const QueryDataTableShPtr& dataPtr);

  int size(const GpuId& gpuId) const;

  std::string getName() const { return _name; }

  std::string getGLSLFunc() const { return "get" + _name; }

  std::string getInGLSLName() const { return "inT" + _name; }
  std::string getInGLSLType() const;
  const ::Rendering::GL::TypeGLShPtr& getInTypeGL() const;

  std::string getOutGLSLName() const { return "outT" + _name; }
  std::string getOutGLSLType() const;
  const ::Rendering::GL::TypeGLShPtr& getOutTypeGL() const;

  bool hasVboPtr() const;
  bool hasVboPtr(const GpuId& gpuId) const;

  bool hasUboPtr() const;
  bool hasUboPtr(const GpuId& gpuId) const;

  bool isDecimal() const { return _decimalExpScale != 0; }

  QueryVertexBufferShPtr getVboPtr(const GpuId& gpuId) const;
  QueryVertexBufferShPtr getVboPtr() const;

  QueryUniformBufferShPtr getUboPtr(const GpuId& gpuId) const;
  QueryUniformBufferShPtr getUboPtr() const;

  bool usesScaleConfig() const { return (_scaleConfigPtr != nullptr || _scalePtr != nullptr); }

  const ScaleRefShPtr& getScaleReference() const { return _scaleConfigPtr; }

  void addToVboAttrMap(const GpuId& gpuId, ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap& attrMap) const;

  virtual void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                     const std::string& uniformAttrName) const = 0;

  void bindUniformScaleToRenderer(::Rendering::GL::Resources::GLShader* activeShader) const;

  std::string getDataColumnName() { return _vboAttrName; }
  const QueryDataTableShPtr& getDataTablePtr() { return _dataPtr; }

  void initGpuResources(const QueryRendererContext* ctx,
                        const std::set<GpuId>& usedGpus,
                        const std::set<GpuId>& unusedGpus);

  virtual operator std::string() const = 0;

 protected:
  enum class VboInitType { FROM_VALUE = 0, FROM_DATAREF, FROM_SCALEREF, UNDEFINED };

  BaseMark* _prntMark;
  std::string _name;
  bool _useScale;
  bool _allowsAccumulatorScale;
  bool _allowsNonColorStrings;
  uint64_t _decimalExpScale;

  std::string _vboAttrName;

  struct PerGpuData : BasePerGpuData {
    QueryVertexBufferShPtr vbo;
    QueryUniformBufferShPtr ubo;

    PerGpuData() : BasePerGpuData(), vbo(nullptr), ubo(nullptr) {}
    explicit PerGpuData(const RootPerGpuDataShPtr& rootData,
                        const QueryVertexBufferShPtr& vbo = nullptr,
                        const QueryUniformBufferShPtr& ubo = nullptr)
        : BasePerGpuData(rootData), vbo(vbo), ubo(ubo) {}

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

  ScaleRefShPtr _scaleConfigPtr;
  ScaleShPtr _scalePtr;

  bool _flexibleType;

  rapidjson::Pointer _fieldJsonPath;
  rapidjson::Pointer _valueJsonPath;
  rapidjson::Pointer _scaleJsonPath;

  RefCallbackId _scaleRefSubscriptionId;
  RefCallbackId _dataRefSubscriptionId;

  virtual void _initScaleFromJSONObj(const rapidjson::Value& obj) = 0;
  virtual void _initFromJSONObj(const rapidjson::Value& obj) {}
  virtual void _initValueFromJSONObj(const rapidjson::Value& obj,
                                     const bool hasScale,
                                     const bool resetTypes = false) = 0;
  virtual void _validateValue(const bool hasScale) = 0;
  virtual void _resetValue() = 0;
  virtual bool _resetTypes() = 0;
  virtual std::pair<bool, bool> _initTypeFromBuffer(const bool hasScale = false) = 0;
  virtual void _validateScale() = 0;
  virtual void _scaleRefUpdateCB(RefEventType refEventType, const RefObjShPtr& refObjPtr) = 0;
  virtual void _updateScalePtr(const ScaleShPtr& scalePtr);

  void _dataRefUpdateCB(RefEventType refEventType, const RefObjShPtr& refObjPtr);
  void _clearFieldPath();
  void _clearDataPtr();

  bool _checkAccumulator(const ScaleShPtr& scalePtr);
  void _setAccumulatorFromScale(const ScaleShPtr& scalePtr);
  void _clearAccumulatorFromScale(const ScaleShPtr& scalePtr);
  void _unsubscribeFromScaleEvent(const ScaleShPtr& scalePtr);
  void _unsubscribeFromDataEvent();
  void _setShaderDirty();
  void _setPropsDirty();

  std::string _printInfo() const;

 private:
  std::set<GpuId> _initUnusedGpus(const std::map<GpuId, QueryBufferShPtr>& bufferMap);
  std::set<GpuId> _initUnusedGpus(const std::set<GpuId>& usedGpus);
  void _initBuffers(const std::map<GpuId, QueryBufferShPtr>& bufferMap);
  bool _internalInitFromData(const std::string& attrName, const QueryDataTableShPtr& dataPtr, const bool hasScale);

  friend class BaseScaleRef;
};

template <typename T, int numComponents = 1>
class RenderProperty : public BaseRenderProperty {
 public:
  RenderProperty(BaseMark* prntMark,
                 const std::string& name,
                 const QueryRendererContextShPtr& ctx,
                 bool useScale = true,
                 bool flexibleType = true,
                 bool allowsAccumulatorScale = false,
                 bool allowsNonColorStrings = false)
      : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType, allowsAccumulatorScale, allowsNonColorStrings),
        _mult(),
        _offset() {
    _resetValue();
    auto rootGpuCache = ctx->getRootGpuCache();
    CHECK(rootGpuCache);
    _inType.reset(new ::Rendering::GL::TypeGL<T, numComponents>(rootGpuCache->supportedExtensions));
    _outType.reset(new ::Rendering::GL::TypeGL<T, numComponents>(rootGpuCache->supportedExtensions));
  }

  virtual ~RenderProperty() {}

  void initializeValue(const T& val) {
    // TODO: this is a public function.. should I protect from already existing data?
    _resetTypes();
    _uniformVal.set(getDataTypeForType<T>(), val);
    _validateValue(false);
  }

  T getUniformValue() const { return _uniformVal.getVal<T>(); };

  void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
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
        auto checkptr = dynamic_cast<::Rendering::GL::TypeGL<T, 1>*>(_inType.get());
        CHECK(checkptr);
        activeShader->setUniformAttribute<T>(uniformAttrName, _uniformVal.getVal<T>());
      }
    }
  }

  operator std::string() const final {
    return "RenderProperty<" + std::string(typeid(T).name()) + ", " + std::to_string(numComponents) + "> " +
           _printInfo();
  }

 protected:
  AnyDataType _uniformVal;

 private:
  T _mult;
  T _offset;

  void _initScaleFromJSONObj(const rapidjson::Value& obj) {
    RUNTIME_EX_ASSERT(
        obj.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), obj, "scale reference for mark property \"" + _name + "\" must be a string."));

    RUNTIME_EX_ASSERT(_ctx != nullptr && !_scaleConfigPtr && !_scalePtr,
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

  void _setupScaleCB(const ScaleShPtr& scalePtr) {
    // setup callbacks for scale updates
    auto cb = std::bind(
        &RenderProperty<T, numComponents>::_scaleRefUpdateCB, this, std::placeholders::_1, std::placeholders::_2);
    _scaleRefSubscriptionId = _ctx->subscribeToRefEvent(RefEventType::ALL, scalePtr, cb);
  }

  void _updateScalePtr(const ScaleShPtr& scalePtr) {
    CHECK(scalePtr);

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

      if (isDecimal()) {
        _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<unsigned int, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<unsigned int, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<int, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<int, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<float, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<float, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<double, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<int64_t, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<int64_t, T>(_ctx, scalePtr, this));
      } else if (dynamic_cast<::Rendering::GL::TypeGL<uint64_t, 1>*>((*inTypeToUse).get())) {
        _scaleConfigPtr.reset(new ScaleRef<uint64_t, T>(_ctx, scalePtr, this));
      } else {
        RUNTIME_EX_ASSERT(scaleDensityAccumulation,
                          std::string(*this) + ": Scale domain with shader type \"" +
                              scalePtr->getDomainTypeGL()->glslType() + "\" and data with shader type \"" +
                              (*inTypeToUse)->glslType() + "\" are not supported to work together.");

        switch (scalePtr->getDomainDataType()) {
          case QueryDataType::DOUBLE:
            _scaleConfigPtr.reset(new ScaleRef<double, T>(_ctx, scalePtr, this));
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

  virtual void _initFromJSONObj(const rapidjson::Value& obj) {
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

  bool _resetTypes() {
    if (_vboInitType != VboInitType::FROM_VALUE) {
      auto rootGpuCache = _ctx->getRootGpuCache();
      CHECK(rootGpuCache);

      ::Rendering::GL::TypeGLShPtr newInType(
          new ::Rendering::GL::TypeGL<T, numComponents>(rootGpuCache->supportedExtensions));
      ::Rendering::GL::TypeGLShPtr newOutType(
          new ::Rendering::GL::TypeGL<T, numComponents>(rootGpuCache->supportedExtensions));

      for (auto& itr : _perGpuData) {
        itr.second.vbo = nullptr;
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

  void _initValueFromJSONObj(const rapidjson::Value& obj, const bool hasScale, const bool resetTypes = false) final {
    if (resetTypes) {
      _resetTypes();
    }
    _uniformVal = RapidJSONUtils::getAnyDataFromJSONObj(obj, _allowsNonColorStrings);
    _validateValue(hasScale);
  }

  virtual void _validateValue(const bool hasScale) {}

  void _resetValue() { _uniformVal.set(getDataTypeForType<T>(), T()); }

  std::pair<bool, bool> _initTypeFromBuffer(const bool hasScale = false) final {
    bool inchanged = false, outchanged = false;
    auto itr = _perGpuData.begin();
    if (itr == _perGpuData.end()) {
      // there's nothing in the data ptr
      if (_inType) {
        inchanged = true;
      }

      if (_outType) {
        outchanged = true;
      }

      _inType = nullptr;
      _outType = nullptr;

      return std::make_pair(inchanged, outchanged);
    }

    QueryBufferShPtr bufToUse = (itr->second.vbo ? std::dynamic_pointer_cast<QueryBuffer>(itr->second.vbo)
                                                 : std::dynamic_pointer_cast<QueryBuffer>(itr->second.ubo));
    RUNTIME_EX_ASSERT(bufToUse != nullptr,
                      std::string(*this) +
                          ": Vertex/uniform buffer is uninitialized. Cannot initialize type for mark property \"" +
                          _name + "\".");

    auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(_dataPtr);
    QueryDataLayoutShPtr layout;
    if (dataPtr) {
      layout = dataPtr->getVboQueryDataLayout();
      if (!layout || !layout->hasAttribute(_vboAttrName)) {
        layout = dataPtr->getUboQueryDataLayout();
        if (layout && !layout->hasAttribute(_vboAttrName)) {
          layout = nullptr;
        }
      }
    }
    auto vboType = bufToUse->getAttributeTypeGL(_vboAttrName, layout);

    _decimalExpScale = 0;
    if (!_flexibleType && (!_useScale || !hasScale)) {
      // if _flexibleType is false, then the render property is rigid,
      // meaning it cannot accept certain types. So validate the type of the attribute
      // in the vbo is appropriate. If validations succeeds, then make sure the
      // in/out types match the vbo
      _validateType(vboType);
    } else {
      auto typeToUse = vboType;
      if (layout && layout->isDecimalAttr(_vboAttrName)) {
        auto rootGpuCache = _ctx->getRootGpuCache();
        typeToUse.reset(new ::Rendering::GL::TypeGL<double, 1>(rootGpuCache->supportedExtensions));

        // NOTE: decimal types are only determined via data buffers.
        _decimalExpScale = layout->getDecimalExp(_vboAttrName);
      }
      if (!_outType || *_outType != *typeToUse) {
        outchanged = true;
      }
      _outType = typeToUse;
    }

    if (!_inType || *_inType != *vboType) {
      inchanged = true;
    }
    _inType = vboType;

    return std::make_pair(inchanged, outchanged);
  }

  void _validateType(const ::Rendering::GL::TypeGLShPtr& type) {
    RUNTIME_EX_ASSERT((*_outType) == (*type),
                      std::string(*this) + ": The vertex buffer type " + (type ? std::string(*type) : "\"null\"") +
                          " does not match the output type " + (type ? std::string(*_outType) : "\"null\"") +
                          " for mark property \"" + _name + "\".");
  }

  void _validateScale() {}

  void _scaleRefUpdateCB(RefEventType refEventType, const RefObjShPtr& refObjPtr) {
    auto scalePtr = std::dynamic_pointer_cast<BaseScale>(refObjPtr);
    CHECK(scalePtr);
    switch (refEventType) {
      case RefEventType::UPDATE: {
        bool accumulatorChanged = scalePtr->hasNumAccumulatorTexturesChanged();
        RUNTIME_EX_ASSERT(!accumulatorChanged || _allowsAccumulatorScale || !scalePtr->hasAccumulator(),
                          std::string(*this) + ": scale \"" + scalePtr->getName() +
                              "\" has been updated into an accumulator scale, but this property doesn't allow for "
                              "accumulator scales.");

        if (_scaleConfigPtr) {
          _scaleConfigPtr->updateScaleRef(scalePtr);
        }
        // TODO(croot): should we do something here if there is no _scaleConfigPtr?
        // Should we check if the data has changed or something?

        if (accumulatorChanged || scalePtr->hasPropertiesChanged() || scalePtr->hasDomainChangedInSize() ||
            scalePtr->hasRangeChangedInSize()) {
          _setShaderDirty();
        }
      } break;
      case RefEventType::REPLACE:
        _updateScalePtr(scalePtr);
        break;
      case RefEventType::REMOVE:
        THROW_RUNTIME_EX(std::string(*this) + ": Error, scale: " + refObjPtr->getName() +
                         " has been removed but is still being referenced by this render property.")
        break;
      default:
        THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " + std::to_string(static_cast<int>(refEventType)) +
                         " isn't currently supported for scale reference updates.");
        break;
    }
  }
};

template <>
RenderProperty<::Rendering::Colors::ColorUnion, 1>::RenderProperty(
    BaseMark* prntMark,
    const std::string& name,
    const QueryRendererContextShPtr& ctx,
    bool useScale,
    // TODO(croot): perhaps remove flexibleType? it ultimately is saying
    // whether or not we can use a scale, right, which we have defined
    // with useScale?
    bool flexibleType,
    bool allowsAccumulatorScale,
    bool allowsNonColorStrings);

template <>
::Rendering::Colors::ColorUnion RenderProperty<::Rendering::Colors::ColorUnion, 1>::getUniformValue() const;

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::bindUniformToRenderer(
    ::Rendering::GL::Resources::GLShader* activeShader,
    const std::string& uniformAttrName) const;

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_updateScalePtr(const ScaleShPtr& scalePtr);

template <>
bool RenderProperty<::Rendering::Colors::ColorUnion, 1>::_resetTypes();

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_resetValue();

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_validateType(const ::Rendering::GL::TypeGLShPtr& type);

template <>
void RenderProperty<::Rendering::Colors::ColorUnion, 1>::_validateScale();

class ColorRenderProperty : public RenderProperty<::Rendering::Colors::ColorUnion> {
 public:
  ColorRenderProperty(BaseMark* prntMark,
                      const std::string& name,
                      const QueryRendererContextShPtr& ctx,
                      bool useScale = true,
                      bool allowsAccumulatorScale = true)
      : RenderProperty<::Rendering::Colors::ColorUnion>(prntMark, name, ctx, useScale, false, allowsAccumulatorScale) {}

  ~ColorRenderProperty() {}

  ::Rendering::Colors::ColorType getColorType() const;
  bool isColorPacked() const;

 private:
  enum class ColorInitType { FROM_STRING = 0, FROM_PACKED_UINT };
  ColorInitType _colorInitType;

  void _initFromJSONObj(const rapidjson::Value& obj) final;
  void _validateValue(const bool hasScale) final;
};

class EnumRenderProperty : public RenderProperty<int> {
 public:
  EnumRenderProperty(BaseMark* prntMark,
                     const std::string& name,
                     const QueryRendererContextShPtr& ctx,
                     bool useScale = true,
                     bool flexibleType = true,
                     std::function<int(const std::string&)> stringConvertFunc = nullptr)
      : RenderProperty<int>(prntMark, name, ctx, useScale, flexibleType, false, true),
        _stringConvertFunc(stringConvertFunc) {}

  ~EnumRenderProperty() {}

 private:
  std::function<int(const std::string&)> _stringConvertFunc;
  void _validateValue(const bool hasScale) final;
};

}  // namespace QueryRendererclass BaseRenderProperty {

#endif  // QUERYRENDERER_MARKS_RENDERPROPERTY_H_
