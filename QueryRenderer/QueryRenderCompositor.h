#ifndef QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
#define QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support some extensions we're going to
// use for compositing

#include "Types.h"
#include "QueryFramebuffer.h"
#include <Rendering/RenderError.h>
#include <Rendering/Types.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <GL/glew.h>
#include <unordered_set>
#include <memory>

namespace QueryRenderer {

class QueryRenderCompositor;
class QueryRenderer;

namespace Impl {

class QueryRenderCompositorImpl {
 public:
  virtual ~QueryRenderCompositorImpl() {}

  size_t getWidth() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->getWidth();
  }

  size_t getHeight() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->getHeight();
  }

  size_t getNumSamples() {
    // CHECK(_framebufferPtr);
    // return _framebufferPtr->getNumSamples();
    return 1;
  }

  bool doHitTest() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->doHitTest();
  }

  bool doDepthTest() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->doDepthTest();
  }

  ::Rendering::Renderer* getRenderer() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->getRenderer();
  }

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx = 0,
                                                 size_t starty = 0,
                                                 int width = -1,
                                                 int height = -1) {
    CHECK(_framebufferPtr);
    return _framebufferPtr->readColorBuffer(startx, starty, width, height);
  }

  virtual ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                          FboColorBuffer texType) = 0;

  virtual ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                                FboRenderBuffer rboType) = 0;

  void resize(size_t width, size_t height) {
    CHECK(_framebufferPtr);
    _framebufferPtr->resize(width, height);
    _resizeImpl(width, height);
  }

  virtual void render(QueryRenderer* queryRenderer) = 0;

 protected:
  QueryRenderCompositorImpl(::Rendering::GL::GLRenderer* renderer,
                            size_t width,
                            size_t height,
                            size_t numSamples = 1,
                            bool doHitTest = false,
                            bool doDepthTest = false)
      : _framebufferPtr(new QueryFramebuffer(renderer, width, height, doHitTest, doDepthTest)) {}

  QueryFramebufferUqPtr _framebufferPtr;

  virtual void _resizeImpl(size_t width, size_t height) = 0;

  friend class QueryRenderCompositor;
};

}  //  namespace Impl

class QueryRenderCompositor {
 public:
  ~QueryRenderCompositor();

  size_t getWidth();
  size_t getHeight();
  size_t getNumSamples();
  bool doHitTest();
  bool doDepthTest();
  ::Rendering::Renderer* getRenderer();

  void resize(size_t width, size_t height);

  void render();

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx = 0, size_t starty = 0, int width = -1, int height = -1);

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType);

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(
      ::Rendering::GL::GLRenderer* renderer,
      FboRenderBuffer rboType = FboRenderBuffer::DEPTH_BUFFER);

  void deleteFboTexture2d(const ::Rendering::GL::Resources::GLTexture2dShPtr& texture2dPtr);
  void deleteFboRenderbuffer(const ::Rendering::GL::Resources::GLRenderbufferShPtr& renderbufferPtr);

 private:
  QueryRenderCompositor(QueryRenderer* prnt,
                        ::Rendering::GL::GLRenderer* renderer,
                        size_t width,
                        size_t height,
                        size_t numSamples = 1,
                        bool doHitTest = false,
                        bool doDepthTest = false);

  std::unique_ptr<Impl::QueryRenderCompositorImpl> _implPtr;

  std::unordered_set<::Rendering::GL::Resources::GLTexture2dShPtr> _compositeTextures;
  std::unordered_set<::Rendering::GL::Resources::GLRenderbufferShPtr> _compositeRbos;

  QueryRenderer* _queryRenderer;
  bool _doHitTest, _doDepthTest;

  friend class ::QueryRenderer::QueryRenderer;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
