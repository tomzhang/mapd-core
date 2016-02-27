#ifndef QUERYRENDERER_QUERYRENDEREROBJECTS_H_
#define QUERYRENDERER_QUERYRENDEREROBJECTS_H_

#include "Types.h"
#include "QueryDataTable.h"
#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/RenderError.h>
#include <vector>

// #include "QueryRendererError.h"
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
// #include <memory>  // std::unique_ptr
// #include "Shader.h"
// #include <string>
// #include <utility>  // std::pair
// #include <initializer_list>
// #include <GL/glew.h>
// #include "QueryRenderer.h"
// #include "VertexBuffer.h"
// #include "BufferLayout.h"
// #include "TypeGL.h"
// #include "RapidJSONUtils.h"
// #include "DataTable.h"
// #include "Color.h"
// #include <cstdint>
// #include <typeinfo>

namespace QueryRenderer {

enum class RefEventType;

class BaseScaleDomainRangeData {
 public:
  BaseScaleDomainRangeData(const std::string& name, bool useString = false) : _name(name), _useString(useString) {}
  virtual ~BaseScaleDomainRangeData() {}
  virtual int size() = 0;
  virtual const ::Rendering::GL::TypeGLShPtr& getTypeGL() = 0;
  virtual const std::type_info& getTypeInfo() = 0;

  std::string getName() { return _name; }
  bool useString() { return _useString; }

 protected:
  std::string _name;
  bool _useString;
};

typedef std::shared_ptr<BaseScaleDomainRangeData> ScaleDomainRangeDataShPtr;

class BaseScale {
 public:
  enum class ScaleType { LINEAR = 0, ORDINAL, UNDEFINED };
  const static std::vector<std::string> scaleVertexShaderSource;

  // QueryDataType domainType;
  // QueryDataType rangeType;

  BaseScale(const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);
  BaseScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            QueryDataType domainDataType,
            QueryDataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);

  virtual ~BaseScale();

  std::string getName() { return _name; }
  const std::string& getNameRef() const { return _name; }
  ScaleType getType() { return _type; }

  QueryDataType getDomainDataType() { return _domainDataType; }
  QueryDataType getRangeDataType() { return _rangeDataType; }

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL() {
    RUNTIME_EX_ASSERT(_domainTypeGL != nullptr, "BaseScale::getDomainTypeGL(): the domain type is uninitialized.");
    return _domainTypeGL;
  }

  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL() {
    RUNTIME_EX_ASSERT(_rangeTypeGL != nullptr, "BaseScale::getRangeTypeGL(): the range type is uninitialized.");

    return _rangeTypeGL;
  }

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");

  virtual std::string getGLSLCode(const std::string& extraSuffix = "",
                                  bool ignoreDomain = false,
                                  bool ignoreRange = false) = 0;

  std::string getDomainGLSLUniformName() { return "uDomains_" + _name; }

  std::string getRangeGLSLUniformName() { return "uRanges_" + _name; }

  std::string getRangeDefaultGLSLUniformName() { return "uDefault_" + _name; }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false) = 0;

  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  virtual bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

  bool hasClampChanged() { return _clampChanged; }
  bool hasDomainChangedInSize() { return _domainSizeChanged; }
  bool hasDomainValsChanged() { return _domainValsChanged; }
  bool hasDomainDataChanged() { return _domainSizeChanged || _domainValsChanged; }
  bool hasRangeChangedInSize() { return _rangeSizeChanged; }
  bool hasRangeValsChanged() { return _rangeValsChanged; }
  bool hasRangeDataChanged() { return _rangeSizeChanged || _rangeValsChanged; }

 protected:
  std::string _name;
  ScaleType _type;

  bool _useClamp;
  bool _clampChanged;

  // TODO(croot): somehow consolidate all the types and use typeid() or the like
  // to handle type-ness.
  QueryDataType _domainDataType;
  ::Rendering::GL::TypeGLShPtr _domainTypeGL;
  bool _domainSizeChanged;
  bool _domainValsChanged;

  QueryDataType _rangeDataType;
  ::Rendering::GL::TypeGLShPtr _rangeTypeGL;
  bool _rangeSizeChanged;
  bool _rangeValsChanged;

  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _jsonPath;

  bool _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
};

typedef std::unique_ptr<BaseScale> ScaleUqPtr;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

template <typename T>
class ScaleDomainRangeData : public BaseScaleDomainRangeData {
 public:
  static const QueryDataType dataType;
  ScaleDomainRangeData(const std::string& name, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(nullptr), _cachedTypeGL(nullptr) {}
  ScaleDomainRangeData(const std::string& name, size_t size, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(new std::vector<T>(size)), _cachedTypeGL(nullptr) {}
  ~ScaleDomainRangeData() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryRendererContextShPtr& ctx,
                             BaseScale::ScaleType type,
                             bool& sizeChanged,
                             bool& valsChanged);
  void updateJSONPath(const rapidjson::Pointer& objPath);

  int size() { return (_vectorPtr == nullptr ? 0 : _vectorPtr->size()); }

  std::vector<T>& getVectorData() { return *_vectorPtr; }

  inline const ::Rendering::GL::TypeGLShPtr& getTypeGL();
  inline const std::type_info& getTypeInfo() { return typeid(T); }

  static T getDataValueFromJSONObj(const rapidjson::Value& obj);

 private:
  std::shared_ptr<std::vector<T>> _vectorPtr;
  ::Rendering::GL::TypeGLShPtr _cachedTypeGL;
  rapidjson::Pointer _jsonPath;

  void _pushItem(const rapidjson::Value& obj);
  void _setItem(size_t idx, const rapidjson::Value& obj);
  void _setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);
  void _updateVectorDataByType(TDataColumn<T>* dataColumnPtr, BaseScale::ScaleType type);
};

template <>
inline const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getTypeGL();

template <>
inline const ::Rendering::GL::TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL();

template <>
unsigned int ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
int ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
float ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
double ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
::Rendering::Objects::ColorRGBA ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getDataValueFromJSONObj(
    const rapidjson::Value& obj);

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj);

// template <>
// ::Rendering::GL::TypeGLShPtr ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::getTypeGL() {
//     // TODO: combine all the different types into a utility file somewhere.
//     // i.e. this is already defined in BufferLayout.h, so let's find a
//     // good way to consolidate these definitions

//     return ::Rendering::GL::TypeGLShPtr(new TypeGL<uint8_t, 4>(true, true));
// }

template <>
void ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::_setFromStringValue(const std::string& strVal,
                                                                                BaseScale::ScaleType type);

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);

template <>
void ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>::_updateVectorDataByType(
    TDataColumn<::Rendering::Objects::ColorRGBA>* dataColumnPtr,
    BaseScale::ScaleType type);

template <typename DomainType, typename RangeType>
class Scale : public BaseScale {
 public:
  Scale(const rapidjson::Value& obj,
        const rapidjson::Pointer& objPath,
        const QueryRendererContextShPtr& ctx,
        const std::string& name = "",
        BaseScale::ScaleType type = BaseScale::ScaleType::UNDEFINED);
  ~Scale();

  std::string getGLSLCode(const std::string& extraSuffix = "", bool ignoreDomain = false, bool ignoreRange = false);
  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false);
  BaseScaleDomainRangeData* getDomainData() { return &_domainPtr; };
  BaseScaleDomainRangeData* getRangeData() { return &_rangePtr; };

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
  // std::vector<DomainType> _domain;
  // std::vector<RangeType> _range;

  // std::shared_ptr<std::vector<DomainType>> _domainPtr;
  // std::shared_ptr<std::vector<RangeType>> _rangePtr;

  ScaleDomainRangeData<DomainType> _domainPtr;
  ScaleDomainRangeData<RangeType> _rangePtr;

  RangeType _defaultVal;
  rapidjson::Pointer _defaultJsonPath;

  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initGLTypes();

  void _setDefaultFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _updateDefaultJSONPath(const rapidjson::Pointer& objPath);
};

// std::unique_ptr<BaseScale> createScaleConfig(QueryDataType domainType, QueryDataType rangeType);

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj);
BaseScale::ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
QueryDataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
ScaleShPtr createScale(const rapidjson::Value& obj,
                       const rapidjson::Pointer& objPath,
                       const QueryRendererContextShPtr& ctx,
                       const std::string& name = "",
                       // TODO(croot): expose default as a constant somewhere
                       BaseScale::ScaleType type = BaseScale::ScaleType::UNDEFINED);

class BaseRenderProperty;
class BaseMark;

class BaseScaleRef {
 public:
  BaseScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp)
      : _ctx(ctx), _scalePtr(scalePtr), _rndrPropPtr(rndrProp) {}
  virtual ~BaseScaleRef() {}

  const std::string& getNameRef() {
    _verifyScalePointer();
    return _scalePtr->getNameRef();
  }

  virtual const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL() {
    _verifyScalePointer();
    return _scalePtr->getDomainTypeGL();
  }

  virtual const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL() {
    _verifyScalePointer();
    return _scalePtr->getRangeTypeGL();
  }

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "") {
    _verifyScalePointer();
    return _scalePtr->getScaleGLSLFuncName(extraSuffix);
  }

  virtual std::string getGLSLCode(const std::string& extraSuffix = "") {
    _verifyScalePointer();
    return _scalePtr->getGLSLCode(extraSuffix);
  }

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::string& extraSuffix = "") {
    _verifyScalePointer();
    return _scalePtr->bindUniformsToRenderer(activeShader, extraSuffix);
  }

  virtual void updateScaleRef(const ScaleShPtr& scalePtr) = 0;

  ScaleShPtr getScalePtr() { return _scalePtr; }

 protected:
  void _verifyScalePointer() {
    RUNTIME_EX_ASSERT(_scalePtr != nullptr, "The scale reference object is uninitialized.");
  }

  QueryRendererContextShPtr _ctx;
  ScaleShPtr _scalePtr;
  BaseRenderProperty* _rndrPropPtr;
};

typedef std::shared_ptr<BaseScaleRef> ScaleRefShPtr;

template <typename DomainType, typename RangeType>
class ScaleRef : public BaseScaleRef {
 public:
  ScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp);
  ~ScaleRef() {}

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();
  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getGLSLCode(const std::string& extraSuffix = "");

  void updateScaleRef(const ScaleShPtr& scalePtr);
  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader, const std::string& extraSuffix = "");

 private:
  std::unique_ptr<ScaleDomainRangeData<DomainType>> _coercedDomainData;
  std::unique_ptr<ScaleDomainRangeData<RangeType>> _coercedRangeData;
  std::unique_ptr<RangeType> _coercedDefaultVal;

  void _updateDomainRange(bool updateDomain, bool updateRange, bool force = false);
  void _doStringToDataConversion(ScaleDomainRangeData<std::string>* domainData);
  void _sort();
};

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
        _flexibleType(flexibleType) {
    _initGpuResources(ctx);
  }

  virtual ~BaseRenderProperty() {
    // std::cerr << "IN BaseRenderProperty DESTRUCTOR " << _name << std::endl;
  }

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryDataTableVBOShPtr& dataPtr);
  void initializeFromData(const std::string& columnName, const QueryDataTableVBOShPtr& dataPtr);

  int size(const GpuId& gpuId) const {
    auto itr = _perGpuData.find(gpuId);
    if (itr != _perGpuData.end()) {
      return itr->second.vbo->numItems();
    }
    return 0;
  }

  std::string getName() const { return _name; }

  std::string getGLSLFunc() const { return "get" + _name; }

  std::string getInGLSLName() const { return "inT" + _name; }

  std::string getInGLSLType() const;

  std::string getOutGLSLName() const { return "outT" + _name; }

  std::string getOutGLSLType() const;

  bool hasVboPtr() {
    for (auto& itr : _perGpuData) {
      if (itr.second.vbo != nullptr) {
        return true;
      }
    }
    return false;
  }
  bool hasVboPtr(const GpuId& gpuId) {
    auto itr = _perGpuData.find(gpuId);

    return (itr != _perGpuData.end() && itr->second.vbo != nullptr);
  }

  QueryVertexBufferShPtr getVboPtr(const GpuId& gpuId) const {
    auto itr = _perGpuData.find(gpuId);
    if (itr != _perGpuData.end()) {
      return itr->second.vbo;
    }

    return nullptr;
  }

  QueryVertexBufferShPtr getVboPtr() const {
    auto itr = _perGpuData.begin();
    if (itr != _perGpuData.end()) {
      return itr->second.vbo;
    }

    return nullptr;
  }

  bool usesScaleConfig() { return (_scaleConfigPtr != nullptr); }

  // ScaleShPtr& getScaleConfig() { return _scaleConfigPtr; }
  const ScaleRefShPtr& getScaleReference() { return _scaleConfigPtr; }

  void addToVboAttrMap(const GpuId& gpuId, ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) const {
    auto itr = _perGpuData.find(gpuId);

    RUNTIME_EX_ASSERT(itr->second.vbo != nullptr,
                      "BaseRenderProperty::addToVboAttrMap(): A vertex buffer is not defined. Cannot add vbo attrs to "
                      "vbo->shader attr map.");

    ::Rendering::GL::Resources::GLVertexBufferShPtr glVbo = itr->second.vbo->getGLVertexBufferPtr();
    auto attrMapItr = attrMap.find(glVbo);
    if (attrMapItr == attrMap.end()) {
      attrMap.insert({glVbo, {{_vboAttrName, _name}}});
    } else {
      attrMapItr->second.push_back({_vboAttrName, _name});
    }

    // _vboPtr->bindToRenderer(activeShader, _vboAttrName, _name);
  }

  virtual void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                     const std::string& uniformAttrName) const = 0;

  std::string getDataColumnName() { return _vboAttrName; }
  const QueryDataTableVBOShPtr& getDataTablePtr() { return _dataPtr; }

 protected:
  enum class VboInitType { FROM_VALUE = 0, FROM_DATAREF, UNDEFINED };

  BaseMark* _prntMark;
  std::string _name;
  bool _useScale;

  std::string _vboAttrName;

  struct PerGpuData {
    QueryRenderManager::PerGpuDataWkPtr qrmGpuData;
    QueryVertexBufferShPtr vbo;

    PerGpuData() : qrmGpuData(), vbo(nullptr) {}
    explicit PerGpuData(const QueryRendererContext::PerGpuData& qrcGpuData, const QueryVertexBufferShPtr& vbo = nullptr)
        : qrmGpuData(qrcGpuData.qrmGpuData), vbo(vbo) {}
    PerGpuData(const PerGpuData& data) : qrmGpuData(data.qrmGpuData), vbo(data.vbo) {}
    PerGpuData(PerGpuData&& data) : qrmGpuData(std::move(data.qrmGpuData)), vbo(std::move(data.vbo)) {}

    ~PerGpuData() {
      // need to make active to properly delete GL resources per-gpu
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }

    QueryRenderManager::PerGpuDataShPtr getQRMGpuData() { return qrmGpuData.lock(); }

    void makeActiveOnCurrentThread() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuData = getQRMGpuData();
      if (qrmGpuData) {
        qrmGpuData->makeActiveOnCurrentThread();
      }
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;

  VboInitType _vboInitType;
  QueryDataTableVBOShPtr _dataPtr;

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
  virtual void _initTypeFromVbo() = 0;
  virtual void _verifyScale() = 0;
  virtual void _scaleRefUpdateCB(RefEventType refEventType, const ScaleShPtr& scalePtr) = 0;

 private:
  void _initGpuResources(const QueryRendererContextShPtr& ctx,
                         const std::unordered_set<GpuId> unusedGpus = std::unordered_set<GpuId>()) {
    const QueryRendererContext::PerGpuDataMap& qrcPerGpuData = ctx->getGpuDataMap();
    for (auto& itr : qrcPerGpuData) {
      if (_perGpuData.find(itr.first) == _perGpuData.end()) {
        _perGpuData.emplace(itr.first, PerGpuData(itr.second));
      }
    }

    for (auto gpuId : unusedGpus) {
      _perGpuData.erase(gpuId);
    }
  }

  void _initVBOs(const std::map<GpuId, QueryVertexBufferShPtr>& vboMap) {
    CHECK(vboMap.size() == _perGpuData.size());

    for (const auto& itr : vboMap) {
      auto myItr = _perGpuData.find(itr.first);
      CHECK(myItr != _perGpuData.end());
      myItr->second.vbo = itr.second;
    }
  }
};

template <typename T, int numComponents = 1>
class RenderProperty : public BaseRenderProperty {
 public:
  RenderProperty(BaseMark* prntMark,
                 const std::string& name,
                 const QueryRendererContextShPtr& ctx,
                 bool useScale = true,
                 bool flexibleType = true);
  //     : BaseRenderProperty(prntMark, name, ctx, useScale, flexibleType), _mult(), _offset() {
  //   _inType.reset(new TypeGL<T, numComponents>());
  //   _outType.reset(new TypeGL<T, numComponents>());
  // }
  ~RenderProperty();  //{
  // _ctx->unsubscribeFromRefEvent(RefEventType::ALL, _scaleConfigPtr->getScalePtr(), _scaleRefSubscriptionCB);
  //}

  void initializeValue(const T& val);
  void bindUniformToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                             const std::string& uniformAttrName) const;

 private:
  T _mult;
  T _offset;
  T _uniformVal;

  void _initScaleFromJSONObj(const rapidjson::Value& obj);
  void _updateScalePtr(const ScaleShPtr& scalePtr);
  void _initFromJSONObj(const rapidjson::Value& obj);
  void _initValueFromJSONObj(const rapidjson::Value& obj);
  void _initTypeFromVbo();
  void _verifyScale();
  void _scaleRefUpdateCB(RefEventType refEventType, const ScaleShPtr& scalePtr);
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

class BaseMark {
 public:
  enum GeomType { POINTS = 0 };  // LINES, POLYS

  BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx);
  BaseMark(GeomType geomType,
           const QueryRendererContextShPtr& ctx,
           const rapidjson::Value& obj,
           const rapidjson::Pointer& objPath);
  virtual ~BaseMark();

  GeomType getType() { return _type; }

  void setShaderDirty() { _shaderDirty = true; }
  void setPropsDirty() { _propsDirty = true; }

  // virtual void _pushDomainItem(const rapidjson::Value& item) = 0;

  // virtual std::pair<std::string, std::string> buildShaderSource() = 0;
  virtual void draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) = 0;

  void setInvalidKey(const int64_t invalidKey) { _invalidKey = invalidKey; }

  virtual bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

  virtual void update() {
    _updateShader();
    _buildVertexArrayObjectFromProperties();
  }

 protected:
  GeomType _type;

  // all query-based shaders should have a "key"

  // TODO(croot): Should we use this as a "property"? Or should we
  // just always include it as an int/int64? My feeling is it should
  // always be present and should be an int/int64
  RenderProperty<int> key;
  int64_t _invalidKey;

  QueryDataTableVBOShPtr _dataPtr;

  struct PerGpuData {
    QueryRenderManager::PerGpuDataWkPtr qrmGpuData;
    ::Rendering::GL::Resources::GLShaderShPtr shaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vaoPtr;

    PerGpuData() : qrmGpuData(), shaderPtr(nullptr) {}
    explicit PerGpuData(const QueryRendererContext::PerGpuData& qrcGpuData,
                        const ::Rendering::GL::Resources::GLShaderShPtr& shaderPtr = nullptr)
        : qrmGpuData(qrcGpuData.qrmGpuData), shaderPtr(shaderPtr) {}
    PerGpuData(const PerGpuData& data) : qrmGpuData(data.qrmGpuData), shaderPtr(data.shaderPtr) {}
    PerGpuData(PerGpuData&& data) : qrmGpuData(std::move(data.qrmGpuData)), shaderPtr(std::move(data.shaderPtr)) {}

    ~PerGpuData() {
      // need to make active to properly delete all GL resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }

    QueryRenderManager::PerGpuDataShPtr getQRMGpuData() { return qrmGpuData.lock(); }

    void makeActiveOnCurrentThread() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuData = getQRMGpuData();
      if (qrmGpuData) {
        qrmGpuData->makeActiveOnCurrentThread();
      }
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;

  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _dataPtrJsonPath;
  rapidjson::Pointer _propertiesJsonPath;
  rapidjson::Pointer _jsonPath;

  bool _shaderDirty;
  bool _propsDirty;

  std::vector<BaseRenderProperty*> _vboProps;
  std::vector<BaseRenderProperty*> _uniformProps;

  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
  void _buildVertexArrayObjectFromProperties();

  virtual void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
  virtual void _updateShader() = 0;

  virtual void _addPropertiesToAttrMap(const GpuId& gpuId,
                                       ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) = 0;
  virtual void _bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader) = 0;

  void _initGpuResources(const QueryRendererContextShPtr& ctx,
                         const std::unordered_set<GpuId> unusedGpus = std::unordered_set<GpuId>()) {
    const QueryRendererContext::PerGpuDataMap& qrcPerGpuData = ctx->getGpuDataMap();
    for (auto& itr : qrcPerGpuData) {
      if (_perGpuData.find(itr.first) == _perGpuData.end()) {
        _perGpuData.emplace(itr.first, PerGpuData(itr.second));
      }
    }

    for (auto gpuId : unusedGpus) {
      _perGpuData.erase(gpuId);
    }
  }

  // protected:
  //     typedef std::unique_ptr<Shader> ShaderPtr;
  //     static std::unordered_map<int, ShaderPtr> _shaderMap;
};

typedef std::unique_ptr<BaseMark> GeomConfigUqPtr;
typedef std::shared_ptr<BaseMark> GeomConfigShPtr;

class PointMark : public BaseMark {
 public:
  PointMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx);
  ~PointMark();

  void draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) final;

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
  RenderProperty<float> x;
  rapidjson::Pointer _xJsonPath;

  RenderProperty<float> y;
  rapidjson::Pointer _yJsonPath;

  RenderProperty<float> z;
  rapidjson::Pointer _zJsonPath;

  RenderProperty<float> size;
  rapidjson::Pointer _sizeJsonPath;

  RenderProperty<unsigned int> id;
  rapidjson::Pointer _idJsonPath;

  RenderProperty<::Rendering::Objects::ColorRGBA> fillColor;
  rapidjson::Pointer _fillColorJsonPath;

  void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _updateShader();

  void _addPropertiesToAttrMap(const GpuId& gpuId, ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap);
  void _bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader);
};

BaseMark::GeomType getMarkTypeFromJSONObj(const rapidjson::Value& obj);
GeomConfigShPtr createMark(const rapidjson::Value& obj,
                           const rapidjson::Pointer& objPath,
                           const QueryRendererContextShPtr& ctx);

}  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDEREROBJECTS_H_
