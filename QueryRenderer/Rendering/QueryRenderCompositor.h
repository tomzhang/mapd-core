#ifndef QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
#define QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support some extensions we're going to
// use for compositing

#include "../Types.h"
#include "Types.h"
#include "QueryFramebuffer.h"
#include <Rendering/RenderError.h>
#include <Rendering/Types.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
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
    CHECK(_framebufferPtr);
    return _framebufferPtr->getNumSamples();
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

  ::Rendering::GL::GLRenderer* getGLRenderer() {
    CHECK(_framebufferPtr);
    return _framebufferPtr->getGLRenderer();
  }

  QueryFramebufferUqPtr& getFramebuffer() { return _framebufferPtr; }

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx = 0,
                                                 size_t starty = 0,
                                                 int width = -1,
                                                 int height = -1) {
    CHECK(_framebufferPtr);
    return _framebufferPtr->readColorBuffer(startx, starty, width, height);
  }

  std::shared_ptr<unsigned int> readIdBuffer(size_t startx = 0,
                                             size_t starty = 0,
                                             int width = -1,
                                             int height = -1,
                                             const FboColorBuffer idBufferType = FboColorBuffer::ID_BUFFER) {
    CHECK(_framebufferPtr);
    return _framebufferPtr->readIdBuffer(startx, starty, width, height, idBufferType);
  }

  void readIdBuffer(size_t startx, size_t starty, int width, int height, unsigned int* idBuffer) {
    CHECK(_framebufferPtr);
    _framebufferPtr->readIdBuffer(startx, starty, width, height, idBuffer);
  }

  void copyRowIdBufferToPbo(QueryIdMapPixelBufferUIntShPtr& pbo) {
    CHECK(_framebufferPtr);
    _framebufferPtr->copyRowIdBufferToPbo(pbo);
  }

  void copyTableIdBufferToPbo(QueryIdMapPixelBufferIntShPtr& pbo) {
    CHECK(_framebufferPtr);
    _framebufferPtr->copyTableIdBufferToPbo(pbo);
  }

  virtual ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                          FboColorBuffer texType) = 0;

  virtual ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                                FboRenderBuffer rboType) = 0;

  virtual void registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                          size_t accumIdx,
                                          size_t numTexturesInArray) = 0;

  virtual void unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                            size_t accumIdx) = 0;

  void resize(size_t width, size_t height) {
    CHECK(_framebufferPtr);
    _framebufferPtr->resize(width, height);
    _resizeImpl(width, height);
  }

  virtual void render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) = 0;

 protected:
  QueryRenderCompositorImpl(QueryRenderManager* prnt,
                            ::Rendering::RendererShPtr rendererPtr,
                            size_t width,
                            size_t height,
                            size_t numSamples = 1,
                            bool doHitTest = false,
                            bool doDepthTest = false)
      : _framebufferPtr(nullptr) {
    CHECK(rendererPtr);
    ::Rendering::GL::GLRenderer* renderer = dynamic_cast<::Rendering::GL::GLRenderer*>(rendererPtr.get());
    CHECK(renderer);
    _framebufferPtr.reset(new QueryFramebuffer(renderer, width, height, doHitTest, doDepthTest, numSamples));
  }
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
  ::Rendering::GL::GLRenderer* getGLRenderer();

  void resize(size_t width, size_t height);

  QueryFramebufferUqPtr& render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus);

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx = 0, size_t starty = 0, int width = -1, int height = -1);

  std::shared_ptr<unsigned int> readIdBuffer(size_t startx = 0,
                                             size_t starty = 0,
                                             int width = -1,
                                             int height = -1,
                                             const FboColorBuffer idBufferType = FboColorBuffer::ID_BUFFER);

  void readIdBuffer(size_t startx, size_t starty, int width, int height, unsigned int* idBuffer);
  void copyRowIdBufferToPbo(QueryIdMapPixelBufferUIntShPtr& pbo);
  void copyTableIdBufferToPbo(QueryIdMapPixelBufferIntShPtr& pbo);

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType);

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(
      ::Rendering::GL::GLRenderer* renderer,
      FboRenderBuffer rboType = FboRenderBuffer::DEPTH_BUFFER);

  void registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                  size_t accumIdx,
                                  size_t numTexturesInArray);

  void unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex, size_t accumIdx);

 private:
  QueryRenderCompositor(QueryRenderManager* prnt,
                        ::Rendering::RendererShPtr& rendererPtr,
                        size_t width,
                        size_t height,
                        size_t numSamples = 1,
                        bool doHitTest = false,
                        bool doDepthTest = false);

  std::unique_ptr<Impl::QueryRenderCompositorImpl> _implPtr;

  bool _doHitTest, _doDepthTest;

  friend class ::QueryRenderer::QueryRenderManager;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERCOMPOSITOR_H_
