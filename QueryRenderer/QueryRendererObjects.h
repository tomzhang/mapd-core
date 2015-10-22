#ifndef QUERY_RENDERER_OBJECTS_H_
#define QUERY_RENDERER_OBJECTS_H_

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

namespace MapD_Renderer {

class QueryRendererContext;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

class BaseScale {
 public:
  enum class ScaleType { LINEAR = 0, ORDINAL };
  const static std::vector<std::string> scaleVertexShaderFilenames;

  std::string name;
  ScaleType type;

  // DataType domainType;
  // DataType rangeType;

  BaseScale(const QueryRendererContextShPtr& ctx);
  BaseScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);

  virtual ~BaseScale();

  TypeGLShPtr& getDomainType() {
    assert(_domainType != nullptr);
    return _domainType;
  }

  TypeGLShPtr& getRangeType() {
    assert(_rangeType != nullptr);
    return _rangeType;
  }

  std::string getScaleGLSLFuncName();

  virtual std::string getGLSLCode() = 0;

  std::string getDomainGLSLUniformName() { return "uDomains_" + name; }

  std::string getRangeGLSLUniformName() { return "uRanges_" + name; }

  virtual void bindUniformsToRenderer(Shader* activeShader) = 0;

 protected:
  TypeGLShPtr _domainType;
  TypeGLShPtr _rangeType;
  QueryRendererContextShPtr _ctx;
  bool _useClamp;

 private:
  // virtual void _pushDomainItem(const rapidjson::Value& item) = 0;
  // virtual void _pushRangeItem(const rapidjson::Value& item) = 0;

  void _initFromJSONObj(const rapidjson::Value& obj);
};

typedef std::unique_ptr<BaseScale> ScaleUqPtr;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

template <typename T>
class ScaleDomainRangeData {
 public:
  static const DataType dataType;
  ScaleDomainRangeData(const std::string& name, bool useString = false)
      : _name(name), _useString(useString), _vectorPtr(nullptr) {}
  ~ScaleDomainRangeData() {}

  void initializeFromJSONObj(const rapidjson::Value& obj,
                             const QueryRendererContextShPtr& ctx,
                             BaseScale::ScaleType type);

  // void setVectorPtr(const std::shared_ptr<std::vector<T>>& vectorPtr) {
  //     _vectorPtr = vectorPtr;
  // }

  // void initializeNewVectorPtr() {
  //     _vectorPtr.reset(new std::vector<T>());
  // }

  // void initializeNewVectorPtr(std::initializer_list<T> l) {
  //     _vectorPtr.reset(new std::vector<T>(l));
  // }

  int size() { return (_vectorPtr == nullptr ? 0 : _vectorPtr->size()); }

  vector<T>& getVectorData() { return *_vectorPtr; }

  inline TypeGLShPtr getTypeGL();

 private:
  std::string _name;
  bool _useString;
  std::shared_ptr<std::vector<T>> _vectorPtr;

  void _pushItem(const rapidjson::Value& obj);
  void _setFromStringValue(const std::string& strVal, BaseScale::ScaleType type);
  void _updateVectorDataByType(TDataColumn<T>* dataColumnPtr, BaseScale::ScaleType type);
};

template <>
inline TypeGLShPtr ScaleDomainRangeData<ColorRGBA>::getTypeGL();

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
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA>* dataColumnPtr,
                                                              BaseScale::ScaleType type);

template <typename DomainType, typename RangeType>
class Scale : public BaseScale {
 public:
  Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
  ~Scale();

  std::string getGLSLCode();
  void bindUniformsToRenderer(Shader* activeShader);

 private:
  // std::vector<DomainType> _domain;
  // std::vector<RangeType> _range;

  // std::shared_ptr<std::vector<DomainType>> _domainPtr;
  // std::shared_ptr<std::vector<RangeType>> _rangePtr;

  ScaleDomainRangeData<DomainType> _domainPtr;
  ScaleDomainRangeData<RangeType> _rangePtr;

  void _pushDomainItem(const rapidjson::Value& obj);
  void _pushRangeItem(const rapidjson::Value& obj);

  void _initFromJSONObj(const rapidjson::Value& obj);
  void _initGLTypes();
};

// std::unique_ptr<BaseScale> createScaleConfig(DataType domainType, DataType rangeType);
ScaleShPtr createScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);

class BaseRenderProperty {
 public:
  BaseRenderProperty(const std::string& name, const QueryRendererContextShPtr& ctx, bool useScale = true)
      : _name(name),
        _useScale(useScale),
        _vboAttrName(""),
        _vboPtr(nullptr),
        _ctx(ctx),
        _inType(nullptr),
        _outType(nullptr),
        _scaleConfigPtr(nullptr){};

  virtual ~BaseRenderProperty() {
    // std::cerr << "IN BaseRenderProperty DESTRUCTOR " << _name << std::endl;
  }

  void initializeFromJSONObj(const rapidjson::Value& obj, const DataTableShPtr& dataPtr);
  void initializeFromData(const std::string& columnName, const DataTableShPtr& dataPtr);

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

  bool usesScaleConfig() { return (_scaleConfigPtr != nullptr); }

  ScaleShPtr& getScaleConfig() { return _scaleConfigPtr; }

  void bindToRenderer(Shader* activeShader) const {
    assert(_vboPtr != nullptr);

    _vboPtr->bindToRenderer(activeShader, _vboAttrName, _name);
  }

 protected:
  std::string _name;
  bool _useScale;

  std::string _vboAttrName;
  VertexBufferShPtr _vboPtr;

  QueryRendererContextShPtr _ctx;

  TypeGLShPtr _inType;
  TypeGLShPtr _outType;

  ScaleShPtr _scaleConfigPtr;

  void _initScaleFromJSONObj(const rapidjson::Value& obj);
  virtual void _initFromJSONObj(const rapidjson::Value& obj) {}
  virtual void _initValueFromJSONObj(const rapidjson::Value& obj, int numItems = 1) = 0;
  virtual void _initTypeFromVbo() = 0;
  virtual void _verifyScale() = 0;
};

template <typename T, int numComponents = 1>
class RenderProperty : public BaseRenderProperty {
 public:
  RenderProperty(const std::string& name, const QueryRendererContextShPtr& ctx, bool useScale = true)
      : BaseRenderProperty(name, ctx, useScale), _mult(), _offset() {}
  ~RenderProperty() {
    // std::cerr << "IN RenderProperty DESTRUCTOR " << _name << std::endl;
  }

  void initializeValue(const T& val, int numItems = 1);

 private:
  T _mult;
  T _offset;

  void _initFromJSONObj(const rapidjson::Value& obj);
  void _initValueFromJSONObj(const rapidjson::Value& obj, int numItems = 1);
  void _initTypeFromVbo();
  void _verifyScale();
};

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj);

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems);

template <>
void RenderProperty<ColorRGBA, 1>::_initTypeFromVbo();

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale();

class BaseMark {
 public:
  enum GeomType { POINTS = 0 };  // LINES, POLYS

  GeomType type;

  BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx);
  BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx, const rapidjson::Value& obj);
  virtual ~BaseMark();

  // virtual void _pushDomainItem(const rapidjson::Value& item) = 0;

  // virtual std::pair<std::string, std::string> buildShaderSource() = 0;
  virtual void draw() = 0;

 protected:
  DataTableShPtr _dataPtr;
  ShaderUqPtr _shaderPtr;

  void _bindToRenderer(Shader* activeShader);

 private:
  GLuint _vao;  // opengl vertex array object id
  QueryRendererContextShPtr _ctx;

  void _initFromJSONObj(const rapidjson::Value& obj);
  void _buildVertexArrayObjectFromProperties(Shader* activeShader);

  virtual void _initPropertiesFromJSONObj(const rapidjson::Value& obj) = 0;
  virtual void _initShader() = 0;

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
  PointMark(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);
  ~PointMark();

  void draw();

 private:
  RenderProperty<float> x;
  RenderProperty<float> y;
  RenderProperty<float> z;
  RenderProperty<float> size;
  RenderProperty<unsigned int> id;
  RenderProperty<ColorRGBA> fillColor;

  void _initPropertiesFromJSONObj(const rapidjson::Value& propObj);
  void _initShader();

  void _initPropertiesForRendering(Shader* activeShader);
  void _bindPropertiesToRenderer(Shader* activeShader);
};

GeomConfigShPtr createMark(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);

}  // MapD_Renderer namespace

#endif  // QUERY_RENDERER_OBJECTS_H_
