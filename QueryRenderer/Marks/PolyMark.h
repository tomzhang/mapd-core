#ifndef QUERYRENDERER_MARKS_POLYMARK_H_
#define QUERYRENDERER_MARKS_POLYMARK_H_

#include "BaseMark.h"

namespace QueryRenderer {

class PolyMark : public BaseMark {
 public:
  PolyMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx);
  ~PolyMark();

  void draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) final;

  bool updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;

  operator std::string() const final;

 private:
  RenderProperty<float> x;
  rapidjson::Pointer _xJsonPath;

  RenderProperty<float> y;
  rapidjson::Pointer _yJsonPath;

  // RenderProperty<float> z;
  // rapidjson::Pointer _zJsonPath;

  RenderProperty<::Rendering::Objects::ColorRGBA> fillColor;
  rapidjson::Pointer _fillColorJsonPath;

  RenderProperty<::Rendering::Objects::ColorRGBA> strokeColor;
  rapidjson::Pointer _strokeColorJsonPath;

  RenderProperty<float> strokeWidth;
  rapidjson::Pointer _strokeWidthJsonPath;

  RenderProperty<unsigned int> id;
  rapidjson::Pointer _idJsonPath;

  std::set<BaseRenderProperty*> _usedProps;
  std::set<BaseRenderProperty*> _usedStrokeProps;

  bool _doFill();
  bool _doStroke();
  std::set<BaseRenderProperty*> _getUsedProps() final;
  void _initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath);

  void _buildShaderSrc(std::string& vertSrc,
                       std::string& fragSrc,
                       std::vector<BaseRenderProperty*>& props,
                       const std::string& uniformBlockName,
                       const std::string& uniformBlockInstanceName);

  void _updateShader() final;

  void _buildVAOData(const GpuId& gpuId,
                     ::Rendering::GL::Resources::GLShader* activeShader,
                     ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap,
                     ::Rendering::GL::Resources::GLIndexBufferShPtr& ibo) final;

  void _bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader,
                              const std::set<BaseRenderProperty*>& props);

  void _updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                         const std::set<GpuId>& usedGpus,
                                         const std::set<GpuId>& unusedGpus) final;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_POLYMARK_H_
