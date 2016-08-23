#ifndef QUERYRENDERER_QUERYRENDERSMAA_H_
#define QUERYRENDERER_QUERYRENDERSMAA_H_

#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>

namespace QueryRenderer {

class QueryRenderSMAAPass {
 public:
  enum class SMAA_QUALITY_PRESET { LOW = 0, MEDIUM, HIGH, ULTRA };
  enum class SMAA_EDGE_DETECTION_TYPE { LUMA = 0, COLOR, DEPTH };

  QueryRenderSMAAPass(::Rendering::GL::GLRendererShPtr& rendererPtr,
                      const size_t width,
                      const size_t height,
                      const size_t numSamples = 1,
                      SMAA_QUALITY_PRESET qualityPreset = SMAA_QUALITY_PRESET::HIGH,
                      SMAA_EDGE_DETECTION_TYPE edgeDetectType = SMAA_EDGE_DETECTION_TYPE::COLOR);
  ~QueryRenderSMAAPass();

  size_t getWidth() const;
  size_t getHeight() const;

  ::Rendering::GL::GLRenderer* getGLRenderer() { return _rendererPtr.get(); }

  void setQuality(SMAA_QUALITY_PRESET quality);
  SMAA_QUALITY_PRESET getQuality() const { return _qualityPreset; }

  void setEdgeDetectionType(SMAA_EDGE_DETECTION_TYPE edgeDetectType) { _edgeDetectType = edgeDetectType; }
  SMAA_EDGE_DETECTION_TYPE getEdgetDetectionType() const { return _edgeDetectType; }

  void resize(size_t width, size_t height);

  void runPass(size_t width,
               size_t height,
               ::Rendering::GL::GLRenderer* renderer,
               ::Rendering::GL::Resources::GLFramebufferShPtr& inputFbo,
               ::Rendering::GL::Resources::GLFramebufferShPtr& outputFbo);

  ::Rendering::GL::Resources::GLFramebufferShPtr getEdgeDetectionFBO() const { return _edgeDetectionFbo; }
  ::Rendering::GL::Resources::GLFramebufferShPtr getBlendingWeightFBO() const { return _blendingWeightFbo; }

 private:
  ::Rendering::GL::GLRendererShPtr _rendererPtr;

  ::Rendering::GL::Resources::GLTexture2dShPtr _edgeDetectionTex;
  ::Rendering::GL::Resources::GLFramebufferShPtr _edgeDetectionFbo;
  ::Rendering::GL::Resources::GLShaderShPtr _edgeDetectionShader;

  ::Rendering::GL::Resources::GLTexture2dShPtr _blendingWeightTex;
  ::Rendering::GL::Resources::GLFramebufferShPtr _blendingWeightFbo;
  ::Rendering::GL::Resources::GLShaderShPtr _blendingWeightShader;

  ::Rendering::GL::Resources::GLShaderShPtr _neighborhoodBlendShader;

  ::Rendering::GL::Resources::GLVertexBufferShPtr _rectvbo;
  ::Rendering::GL::Resources::GLVertexArrayShPtr _vao;

  ::Rendering::GL::Resources::GLTexture2dShPtr _areaTex;
  ::Rendering::GL::Resources::GLTexture2dShPtr _searchTex;

  SMAA_QUALITY_PRESET _qualityPreset;
  SMAA_EDGE_DETECTION_TYPE _edgeDetectType;

  bool _usePredication;   // TODO(croot)
  bool _useReprojection;  // TODO(croot)

  void _initializeRsrcs(const size_t width, const size_t height, const size_t numSamples);
  void _buildShaders(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERSMAA_H_
