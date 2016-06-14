#ifndef QUERYRENDERER_MARKS_MARK_H_
#define QUERYRENDERER_MARKS_MARK_H_

#include "Enums.h"
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
           const rapidjson::Pointer& objPath,
           QueryDataTableBaseType baseType,
           bool mustUseDataRef);
  virtual ~BaseMark();

  GeomType getType() { return _type; }
  size_t numGpus() const { return _perGpuData.size(); }
  std::set<GpuId> getUsedGpus() const { return _initUnusedGpus(); }

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

  QueryDataTableShPtr _dataPtr;

  struct PerGpuData : BasePerGpuData {
    ::Rendering::GL::Resources::GLShaderShPtr shaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vaoPtr;
    ::Rendering::GL::Resources::GLShaderShPtr strokeShaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr strokeVaoPtr;

    PerGpuData() : BasePerGpuData() {}
    explicit PerGpuData(const BasePerGpuData& data,
                        const ::Rendering::GL::Resources::GLShaderShPtr& shaderPtr = nullptr,
                        const ::Rendering::GL::Resources::GLShaderShPtr& strokeShaderPtr = nullptr)
        : BasePerGpuData(data), shaderPtr(shaderPtr), strokeShaderPtr(strokeShaderPtr) {}
    PerGpuData(const PerGpuData& data)
        : BasePerGpuData(data),
          shaderPtr(data.shaderPtr),
          vaoPtr(data.vaoPtr),
          strokeShaderPtr(data.strokeShaderPtr),
          strokeVaoPtr(data.strokeVaoPtr) {}
    PerGpuData(PerGpuData&& data)
        : BasePerGpuData(std::move(data)),
          shaderPtr(std::move(data.shaderPtr)),
          vaoPtr(std::move(data.vaoPtr)),
          strokeShaderPtr(std::move(data.strokeShaderPtr)),
          strokeVaoPtr(std::move(data.strokeVaoPtr)) {}

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

  std::set<BaseRenderProperty*> _vboProps;
  std::set<BaseRenderProperty*> _uboProps;
  std::set<BaseRenderProperty*> _uniformProps;

  void _initFromJSONObj(const rapidjson::Value& obj,
                        const rapidjson::Pointer& objPath,
                        QueryDataTableBaseType baseType,
                        bool mustUseDataRef,
                        bool initializing);

  virtual std::set<BaseRenderProperty*> _getUsedProps() = 0;
  void _updateProps(const std::set<BaseRenderProperty*>& usedProps, bool force = false);

 private:
  void _buildVertexArrayObjectFromProperties();

  virtual void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
  virtual void _updateShader() = 0;

  virtual void _buildVAOData(const GpuId& gpuId,
                             ::Rendering::GL::Resources::GLShader* activeShader,
                             ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap,
                             ::Rendering::GL::Resources::GLIndexBufferShPtr& ibo) = 0;

  std::set<GpuId> _initUnusedGpus() const;
  void _initGpuResources(const QueryRendererContext* ctx, bool initializing = true);

  virtual void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                                 const std::set<GpuId>& usedGpus,
                                                 const std::set<GpuId>& unusedGpus) = 0;

  friend class QueryRendererContext;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_MARK_H_