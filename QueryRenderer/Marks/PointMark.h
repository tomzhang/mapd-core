#ifndef QUERYRENDERER_MARKS_POINTMARK_H_
#define QUERYRENDERER_MARKS_POINTMARK_H_

#include "BaseMark.h"

namespace QueryRenderer {

class PointMark : public BaseMark {
 public:
  PointMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx);
  ~PointMark();

  void draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) final;

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

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

  void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);
  void _updateShader() final;

  void _addPropertiesToAttrMap(const GpuId& gpuId, ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap);
  void _bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader);

  void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                         const std::unordered_set<GpuId> unusedGpus) final {
    // this function should only be called when not initializing.
    // so pass 'false' for the initializing parameter in the following
    x.initGpuResources(ctx, unusedGpus, false);
    y.initGpuResources(ctx, unusedGpus, false);
    z.initGpuResources(ctx, unusedGpus, false);
    size.initGpuResources(ctx, unusedGpus, false);
    id.initGpuResources(ctx, unusedGpus, false);
    fillColor.initGpuResources(ctx, unusedGpus, false);
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_POINTMARK_H_
