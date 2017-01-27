#ifndef QUERYRENDERER_MARKS_MARK_H_
#define QUERYRENDERER_MARKS_MARK_H_

#include "Enums.h"
#include "../Types.h"
#include "../PerGpuData.h"
#include "RenderProperty.h"

#include <Rendering/Renderer/GL/GLRenderer.h>

#include <boost/multi_index_container.hpp>

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

  bool hasAccumulator() const;
  std::string getAccumulatorScaleName() const;
  ScaleShPtr getAccumulatorScale() const;

  // set/clear accumulator should only be called by render property
  // TODO(croot): make these functions private and make render properties friends?
  void setAccumulatorScale(const ScaleShPtr& scalePtr, const ScaleRefShPtr& scaleRefPtr);
  void clearAccumulatorScale();

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

  RenderProperty<int64_t> key;
  int64_t _invalidKey;

  QueryDataTableShPtr _dataPtr;

  struct PerGpuData : BasePerGpuData {
    ::Rendering::GL::Resources::GLShaderShPtr shaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr vaoPtr;
    ::Rendering::GL::Resources::GLShaderShPtr strokeShaderPtr;
    ::Rendering::GL::Resources::GLVertexArrayShPtr strokeVaoPtr;

    PerGpuData() : BasePerGpuData() {}
    explicit PerGpuData(const RootPerGpuDataShPtr& rootData,
                        const ::Rendering::GL::Resources::GLShaderShPtr& shaderPtr = nullptr,
                        const ::Rendering::GL::Resources::GLShaderShPtr& strokeShaderPtr = nullptr)
        : BasePerGpuData(rootData), shaderPtr(shaderPtr), strokeShaderPtr(strokeShaderPtr) {}
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

  // for multi-index hash-by-property-name tag
  struct PropertyName {};
  typedef boost::multi_index_container<
      BaseRenderProperty*,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<boost::multi_index::identity<BaseRenderProperty*>>,

          boost::multi_index::hashed_unique<
              boost::multi_index::tag<PropertyName>,
              boost::multi_index::const_mem_fun<BaseRenderProperty, std::string, &BaseRenderProperty::getName>>>>
      PropMap;

  typedef PropMap::index<PropertyName>::type PropMap_by_Name;

  PerGpuDataMap _perGpuData;

  QueryRendererContextShPtr _ctx;

  rapidjson::Pointer _dataPtrJsonPath;
  rapidjson::Pointer _propertiesJsonPath;
  rapidjson::Pointer _jsonPath;

  bool _shaderDirty;
  bool _propsDirty;

  PropMap _vboProps;
  PropMap _uboProps;
  PropMap _uniformProps;
  PropMap _decimalProps;

  void _initFromJSONObj(const rapidjson::Value& obj,
                        const rapidjson::Pointer& objPath,
                        QueryDataTableBaseType baseType,
                        bool mustUseDataRef);

  void _initGpuResources(const QueryRendererContext* ctx, bool initializing = true);

  virtual std::set<BaseRenderProperty*> _getUsedProps() = 0;
  void _updateProps(const std::set<BaseRenderProperty*>& usedProps, bool force = false);
  void _updateShader(std::string& vertSrc, std::string& fragSrc);
  void _setKeyInShaderSrc(std::string& shaderSrc);

 private:
  void _buildVertexArrayObjectFromProperties();

  virtual void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) = 0;
  virtual void _updateShader() = 0;

  virtual void _buildVAOData(const GpuId& gpuId,
                             ::Rendering::GL::Resources::GLShader* activeShader,
                             ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap& attrMap,
                             ::Rendering::GL::Resources::GLIndexBufferShPtr& ibo) = 0;

  std::set<GpuId> _initUnusedGpus() const;

  virtual void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                                 const std::set<GpuId>& usedGpus,
                                                 const std::set<GpuId>& unusedGpus) = 0;

  ScaleWkPtr _activeAccumulator;

  friend class QueryRendererContext;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_MARK_H_
