#ifndef QUERYRENDERER_MARKS_POINTMARK_H_
#define QUERYRENDERER_MARKS_POINTMARK_H_

#include "BaseMark.h"

namespace QueryRenderer {

class PointMark : public BaseMark {
 public:
  PointMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx);
  ~PointMark();

  void draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) final;

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;

  operator std::string() const final;

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

  std::set<BaseRenderProperty*> _getUsedProps() final;
  void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _updateShader() final;

  void _buildVAOData(const GpuId& gpuId,
                     ::Rendering::GL::Resources::GLShader* activeShader,
                     ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap& attrMap,
                     ::Rendering::GL::Resources::GLIndexBufferShPtr& ibo) final;

  void _bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader);

  void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                         const std::set<GpuId>& usedGpus,
                                         const std::set<GpuId>& unusedGpus) final;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_POINTMARK_H_
