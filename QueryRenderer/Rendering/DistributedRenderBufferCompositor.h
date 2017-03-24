#ifndef QUERYRENDERER_DISTRIBUTEDRENDERBUFFERCOMPOSITOR_H_
#define QUERYRENDERER_DISTRIBUTEDRENDERBUFFERCOMPOSITOR_H_

#include "../Types.h"
#include "QueryFramebuffer.h"
#include <Rendering/Renderer/GL/Resources/Types.h>

namespace QueryRenderer {
class DistributedRenderBufferCompositor {
 public:
  DistributedRenderBufferCompositor(const QueryFramebufferShPtr& framebufferPtr, const bool supportsInt64);
  ~DistributedRenderBufferCompositor();

  ::Rendering::GL::GLRenderer* getGLRenderer();
  QueryFramebufferShPtr getFramebufferPtr() { return _framebufferPtr; }

  size_t getWidth() const;
  size_t getHeight() const;

  void resize(const size_t width, const size_t height);
  void render(const std::vector<RawPixelData>& pixelData);

 private:
  QueryFramebufferShPtr _framebufferPtr;

  ::Rendering::GL::Resources::GLVertexBufferShPtr _rectvbo;
  ::Rendering::GL::Resources::GLShaderShPtr _shader;
  ::Rendering::GL::Resources::GLVertexArrayShPtr _vao;
  ::Rendering::GL::Resources::GLTexture2dShPtr _rgbaTexture;
  ::Rendering::GL::Resources::GLTexture2dShPtr _id1ATexture;
  ::Rendering::GL::Resources::GLTexture2dShPtr _id1BTexture;
  ::Rendering::GL::Resources::GLTexture2dShPtr _id2Texture;

  // TODO(croot): do depth

  void _initResources(const bool supportsInt64);
};
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DISTRIBUTEDRENDERBUFFERCOMPOSITOR_H_
