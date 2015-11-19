#ifndef QUERY_RENDERER_OBJECTS_H_
#define QUERY_RENDERER_OBJECTS_H_

#include "QueryRendererError.h"
#include "rapidjson/document.h"
#include <memory>  // std::unique_ptr
#include "Shader.h"
#include <string>
#include <utility>  // std::pair
#include <initializer_list>
#include <GL/glew.h>
#include "QueryRenderer.h"
#include "VertexBuffer.h"
#include "BufferLayout.h"
#include "TypeGL.h"
#include "RapidJSONUtils.h"
#include "DataTable.h"
#include "Color.h"
#include <cstdint>
#include <typeinfo>

namespace MapD_Renderer {

class QueryRendererContext;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

class BaseScaleDomainRangeData {
 public:
  BaseScaleDomainRangeData(const std::string& name, bool useString = false) : _name(name), _useString(useString) {}
  virtual ~BaseScaleDomainRangeData() {}
  virtual int size() = 0;
  virtual const TypeGLShPtr& getTypeGL() = 0;
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

  // DataType domainType;
  // DataType rangeType;

  BaseScale(const QueryRendererContextShPtr& ctx,
            DataType domainDataType,
            DataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);
  BaseScale(const rapidjson::Value& obj,
            const rapidjson::Pointer& objPath,
            const QueryRendererContextShPtr& ctx,
            DataType domainDataType,
            DataType rangeDataType,
            const std::string& name = "",
            ScaleType type = ScaleType::UNDEFINED);

  virtual ~BaseScale();

  // std::string getName() { return name; }
  const std::string& getNameRef() { return _name; }
  ScaleType getType() { return _type; }

  DataType getDomainDataType() { return _domainDataType; }
  DataType getRangeDataType() { return _rangeDataType; }

  const TypeGLShPtr& getDomainTypeGL() {
    RUNTIME_EX_ASSERT(_domainTypeGL != nullptr, "BaseScale::getDomainTypeGL(): the domain type is uninitialized.");
    return _domainTypeGL;
  }

  const TypeGLShPtr& getRangeTypeGL() {
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

  virtual void bindUniformsToRenderer(Shader* activeShader,
                                      const std::string& extraSuffix = "",
                                      bool ignoreDomain = false,
                                      bool ignoreRange = false) = 0;

  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  virtual void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

 protected:
  std::string _name;
  ScaleType _type;

  bool _useClamp;

  // TODO(croot): somehow consolidate all the types and use typeid() or the like
  // to handle type-ness.
  DataType _domainDataType;
  TypeGLShPtr _domainTypeGL;

  DataType _rangeDataType;
  TypeGLShPtr _rangeTypeGL;

  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _jsonPath;

  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
};

typedef std::unique_ptr<BaseScale> ScaleUqPtr;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

template <typename T>
class ScaleDomainRangeData : public BaseScaleDomainRangeData {
 public:
  static const DataType dataType;
  ScaleDomainRangeData(const std::string& name, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(nullptr), _cachedTypeGL(nullptr) {}
  ScaleDomainRangeData(const std::string& name, size_t size, bool useString = false)
      : BaseScaleDomainRangeData(name, useString), _vectorPtr(new std::vector<T>(size)), _cachedTypeGL(nullptr) {}
  ~ScaleDomainRangeData() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryRendererContextShPtr& ctx,
                             BaseScale::ScaleType type);
  void updateJSONPath(const rapidjson::Pointer& objPath);

  int size() { return (_vectorPtr == nullptr ? 0 : _vectorPtr->size()); }

  std::vector<T>& getVectorData() { return *_vectorPtr; }

  inline const TypeGLShPtr& getTypeGL();
  inline const std::type_info& getTypeInfo() { return typeid(T); }

  static T getDataValueFromJSONObj(const rapidjson::Value& obj);

 private:
  std::shared_ptr<std::vector<T>> _vectorPtr;
  TypeGLShPtr _cachedTypeGL;
  rapidjson::Pointer _jsonPath;

  void _pushItem(const rapidjson::Value& obj);
  void _setItem(size_t idx, const rapidjson::Value& obj);
  void _setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);
  void _updateVectorDataByType(TDataColumn<T>* dataColumnPtr, BaseScale::ScaleType type);
};

template <>
inline const TypeGLShPtr& ScaleDomainRangeData<ColorRGBA>::getTypeGL();

template <>
inline const TypeGLShPtr& ScaleDomainRangeData<std::string>::getTypeGL();

template <>
unsigned int MapD_Renderer::ScaleDomainRangeData<unsigned int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
int MapD_Renderer::ScaleDomainRangeData<int>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
float MapD_Renderer::ScaleDomainRangeData<float>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
double MapD_Renderer::ScaleDomainRangeData<double>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
ColorRGBA MapD_Renderer::ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(const rapidjson::Value& obj);

template <>
std::string MapD_Renderer::ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(const rapidjson::Value& obj);

// template <>
// TypeGLShPtr ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
//     // TODO: combine all the different types into a utility file somewhere.
//     // i.e. this is already defined in BufferLayout.h, so let's find a
//     // good way to consolidate these definitions

//     return TypeGLShPtr(new TypeGL<uint8_t, 4>(true, true));
// }

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);

template <>
void ScaleDomainRangeData<std::string>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);

template <>
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA>* dataColumnPtr,
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
  void bindUniformsToRenderer(Shader* activeShader,
                              const std::string& extraSuffix = "",
                              bool ignoreDomain = false,
                              bool ignoreRange = false);
  BaseScaleDomainRangeData* getDomainData() { return &_domainPtr; };
  BaseScaleDomainRangeData* getRangeData() { return &_rangePtr; };

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

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

// std::unique_ptr<BaseScale> createScaleConfig(DataType domainType, DataType rangeType);

std::string getScaleNameFromJSONObj(const rapidjson::Value& obj);
BaseScale::ScaleType getScaleTypeFromJSONObj(const rapidjson::Value& obj);
DataType getScaleDomainDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
DataType getScaleRangeDataTypeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
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

  virtual const TypeGLShPtr& getDomainTypeGL() {
    _verifyScalePointer();
    return _scalePtr->getDomainTypeGL();
  }

  virtual const TypeGLShPtr& getRangeTypeGL() {
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

  virtual void bindUniformsToRenderer(Shader* activeShader, const std::string& extraSuffix = "") {
    _verifyScalePointer();
    return _scalePtr->bindUniformsToRenderer(activeShader, extraSuffix);
  }

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

  const TypeGLShPtr& getDomainTypeGL();
  const TypeGLShPtr& getRangeTypeGL();

  std::string getGLSLCode(const std::string& extraSuffix = "");

  void bindUniformsToRenderer(Shader* activeShader, const std::string& extraSuffix = "");

 private:
  std::unique_ptr<ScaleDomainRangeData<DomainType>> _coercedDomainData;
  std::unique_ptr<ScaleDomainRangeData<RangeType>> _coercedRangeData;
  std::unique_ptr<RangeType> _coercedDefaultVal;

  void _doStringToDataConversion(ScaleDomainRangeData<std::string>* domainData);
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
        _vboPtr(nullptr),
        _vboInitType(VboInitType::UNDEFINED),
        _dataPtr(nullptr),
        _ctx(ctx),
        _inType(nullptr),
        _outType(nullptr),
        _scaleConfigPtr(nullptr),
        _flexibleType(flexibleType) {}

  virtual ~BaseRenderProperty() {
    // std::cerr << "IN BaseRenderProperty DESTRUCTOR " << _name << std::endl;
  }

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const DataVBOShPtr& dataPtr);
  void initializeFromData(const std::string& columnName, const DataVBOShPtr& dataPtr);

  int size() const {
    if (_vboPtr) {
      return _vboPtr->size();
    }
    return 0;
  }

  std::string getName() const { return _name; }

  std::string getGLSLFunc() const { return "get" + _name; }

  std::string getInGLSLName() const { return "inT" + _name; }

  std::string getInGLSLType() const;

  std::string getOutGLSLName() const { return "outT" + _name; }

  std::string getOutGLSLType() const;

  bool hasVboPtr() { return (_vboPtr != nullptr); }
  BaseVertexBufferShPtr getVboPtr() const { return _vboPtr; }

  bool usesScaleConfig() { return (_scaleConfigPtr != nullptr); }

  // ScaleShPtr& getScaleConfig() { return _scaleConfigPtr; }
  const ScaleRefShPtr& getScaleReference() { return _scaleConfigPtr; }

  void bindToRenderer(Shader* activeShader) const {
    RUNTIME_EX_ASSERT(_vboPtr != nullptr,
                      "BaseRenderProperty::bindToRenderer(): A vertex buffer is not defined. Cannot bind to renderer.");

    _vboPtr->bindToRenderer(activeShader, _vboAttrName, _name);
  }

  std::string getDataColumnName() { return _vboAttrName; }
  const DataVBOShPtr& getDataTablePtr() { return _dataPtr; }

 protected:
  enum class VboInitType { FROM_VALUE = 0, FROM_DATAREF, UNDEFINED };

  BaseMark* _prntMark;
  std::string _name;
  bool _useScale;

  std::string _vboAttrName;
  BaseVertexBufferShPtr _vboPtr;
  VboInitType _vboInitType;
  DataVBOShPtr _dataPtr;

  QueryRendererContextShPtr _ctx;

  TypeGLShPtr _inType;
  TypeGLShPtr _outType;

  // ScaleShPtr _scaleConfigPtr;
  ScaleRefShPtr _scaleConfigPtr;

  bool _flexibleType;

  rapidjson::Pointer _fieldJsonPath;
  rapidjson::Pointer _valueJsonPath;
  rapidjson::Pointer _scaleJsonPath;

  virtual void _initScaleFromJSONObj(const rapidjson::Value& obj) = 0;
  virtual void _initFromJSONObj(const rapidjson::Value& obj) {}
  virtual void _initValueFromJSONObj(const rapidjson::Value& obj, int numItems = 1) = 0;
  virtual void _initTypeFromVbo() = 0;
  virtual void _verifyScale() = 0;
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
    _inType.reset(new TypeGL<T, numComponents>());
    _outType.reset(new TypeGL<T, numComponents>());
  }
  ~RenderProperty() {}

  void initializeValue(const T& val, int numItems = 1);

 private:
  T _mult;
  T _offset;

  void _initScaleFromJSONObj(const rapidjson::Value& obj);
  void _initFromJSONObj(const rapidjson::Value& obj);
  void _initValueFromJSONObj(const rapidjson::Value& obj, int numItems = 1);
  void _initTypeFromVbo();
  void _verifyScale();
};

template <>
RenderProperty<ColorRGBA, 1>::RenderProperty(BaseMark* prntMark,
                                             const std::string& name,
                                             const QueryRendererContextShPtr& ctx,
                                             bool useScale,
                                             // TODO(croot): perhaps remove flexibleType? it ultimately is saying
                                             // whether or not we can use a scale, right, which we have defined
                                             // with useScale?
                                             bool flexibleType);

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj);

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems);

// template <>
// void RenderProperty<ColorRGBA, 1>::_initTypeFromVbo();

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale();

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
  virtual void draw() = 0;

  void setInvalidKey(const int64_t invalidKey) { _invalidKey = invalidKey; }

  virtual void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;

 protected:
  GeomType _type;

  // all query-based shaders should have a "key"

  // TODO(croot): Should we use this as a "property"? Or should we
  // just always include it as an int/int64? My feeling is it should
  // always be present and should be an int/int64
  RenderProperty<int> key;
  int64_t _invalidKey;

  DataVBOShPtr _dataPtr;

  ShaderUqPtr _shaderPtr;
  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _dataPtrJsonPath;
  rapidjson::Pointer _propertiesJsonPath;
  rapidjson::Pointer _jsonPath;

  bool _shaderDirty;
  bool _propsDirty;

  void _bindToRenderer(Shader* activeShader);
  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

 private:
  GLuint _vao;  // opengl vertex array object id

  void _buildVertexArrayObjectFromProperties(Shader* activeShader);

  virtual void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
  virtual void _updateShader() = 0;

  virtual void _initPropertiesForRendering(Shader* activeShader) = 0;
  virtual void _bindPropertiesToRenderer(Shader* activeShader) = 0;

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

  void draw();

  void updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

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

  RenderProperty<ColorRGBA> fillColor;
  rapidjson::Pointer _fillColorJsonPath;

  void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _updateShader();

  void _initPropertiesForRendering(Shader* activeShader);
  void _bindPropertiesToRenderer(Shader* activeShader);
};

BaseMark::GeomType getMarkTypeFromJSONObj(const rapidjson::Value& obj);
GeomConfigShPtr createMark(const rapidjson::Value& obj,
                           const rapidjson::Pointer& objPath,
                           const QueryRendererContextShPtr& ctx);

}  // MapD_Renderer namespace

#endif  // QUERY_RENDERER_OBJECTS_H_
