#ifndef QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
#define QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support some extensions we're going to
// use for compositing

#include "Types.h"
#include <GL/glew.h>
// #include <GL/gl.h>
// #include <GL/glext.h>

#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <unordered_set>

namespace QueryRenderer {

class QueryRenderCompositor;

class QueryRenderCompositor {
 public:
  virtual ~QueryRenderCompositor();

  size_t getWidth();
  size_t getHeight();
  size_t getNumSamples();

  virtual void resize(size_t width, size_t height) = 0;

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType);

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(
      ::Rendering::GL::GLRenderer* renderer,
      FboRenderBuffer rboType = FboRenderBuffer::DEPTH_BUFFER);

  void deleteFboTexture2d(const ::Rendering::GL::Resources::GLTexture2dShPtr& texture2dPtr);
  void deleteFboRenderbuffer(const ::Rendering::GL::Resources::GLRenderbufferShPtr& renderbufferPtr);

 protected:
  QueryRenderCompositor(::Rendering::GL::GLRenderer* renderer,
                        size_t width,
                        size_t height,
                        size_t numSamples = 1,
                        bool doHitTest = false,
                        bool doDepthTest = false);

  virtual ::Rendering::GL::Resources::GLTexture2dShPtr _createFboTexture2dImpl(::Rendering::GL::GLRenderer* renderer,
                                                                               FboColorBuffer texType) = 0;

  virtual ::Rendering::GL::Resources::GLRenderbufferShPtr _createFboRenderbufferImpl(
      ::Rendering::GL::GLRenderer* renderer,
      FboRenderBuffer rboType) = 0;

  QueryFramebufferUqPtr _framebufferPtr;

  std::unordered_set<::Rendering::GL::Resources::GLTexture2dShPtr> _compositeTextures;
  std::unordered_set<::Rendering::GL::Resources::GLRenderbufferShPtr> _compositeRbos;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
