#ifndef QUERYRENDERER_MARKS_MARK_H_
#define QUERYRENDERER_MARKS_MARK_H_

#include "../Types.h"
#include "../PerGpuData.h"
#include "RenderProperty.h"

#include <Rendering/Renderer/GL/GLRenderer.h>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"

namespace QueryRenderer {

class BaseMark {
 public:
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

  virtual operator std::string() const = 0;

 protected:
  GeomType _type;

  // all query-based shaders should have a "key"

  // TODO(croot): Should we use this as a "property"? Or should we
  // just always include it as an int/int64? My feeling is it should
  // always be present and should be an int/int64
  RenderProperty<int> key;
  int64_t _invalidKey;

  QueryDataTableVBOShPtr _dataPtr;

  struct PerGpuData : BasePerGpuData {
    ::Rendering::GL::Resources::GLShaderShPtr shaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vaoPtr;

    PerGpuData() : BasePerGpuData(), shaderPtr(nullptr) {}
    explicit PerGpuData(const BasePerGpuData& data,
                        const ::Rendering::GL::Resources::GLShaderShPtr& shaderPtr = nullptr)
        : BasePerGpuData(data), shaderPtr(shaderPtr) {}
    PerGpuData(const PerGpuData& data) : BasePerGpuData(data), shaderPtr(data.shaderPtr) {}
    PerGpuData(PerGpuData&& data) : BasePerGpuData(std::move(data)), shaderPtr(std::move(data.shaderPtr)) {}

    ~PerGpuData() {
      // need to make active to properly delete all GL resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
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

  void _initGpuResources(const QueryRendererContext* ctx,
                         const std::unordered_set<GpuId> unusedGpus = std::unordered_set<GpuId>(),
                         bool initializing = true);

  virtual void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                                 const std::unordered_set<GpuId> unusedGpus) = 0;

  friend class QueryRendererContext;

  // protected:
  //     typedef std::unique_ptr<Shader> ShaderPtr;
  //     static std::unordered_map<int, ShaderPtr> _shaderMap;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_MARK_H_
