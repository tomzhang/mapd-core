#include "DistributedRenderBufferCompositor.h"
#include "shaders/distributedCompositor_vert.h"
#include "shaders/distributedCompositor_frag.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>

using ::Rendering::GL::Resources::GLInterleavedBufferLayout;
using ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr;
using ::Rendering::GL::Resources::GLTexture2dSampleProps;

namespace QueryRenderer {

DistributedRenderBufferCompositor::DistributedRenderBufferCompositor(const QueryFramebufferShPtr& framebufferPtr,
                                                                     const bool supportsInt64)
    : _framebufferPtr(framebufferPtr) {
  CHECK(framebufferPtr && framebufferPtr->getNumSamples() == 1);
  _initResources(supportsInt64);
}

DistributedRenderBufferCompositor::~DistributedRenderBufferCompositor() {
  // make our renderer active on the current thread to ensure
  // all resources are properly deleted
  auto renderer = _framebufferPtr->getGLRenderer();
  renderer->makeActiveOnCurrentThread();
}

void DistributedRenderBufferCompositor::_initResources(const bool supportsInt64) {
  auto renderer = _framebufferPtr->getGLRenderer();

  ::Rendering::GL::GLRenderer* currRenderer = nullptr;
  ::Rendering::Window* currWindow = nullptr;
  bool reset = false;

  if (!renderer->isActiveOnCurrentThread()) {
    currRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    currWindow = ::Rendering::GL::GLRenderer::getCurrentThreadWindow();
    renderer->makeActiveOnCurrentThread();
    reset = true;
  }

  auto rsrcMgr = renderer->getResourceManager();

  // create a rectangle vertex buffer that will cover the entire buffer
  GLInterleavedBufferLayoutShPtr bufferLayout(new GLInterleavedBufferLayout(renderer->getSupportedExtensions()));
  bufferLayout->addAttribute<float, 2>("pos");
  _rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);

  std::string fragSrc(DistributedCompositor_frag::source);
  _shader = rsrcMgr->createShader(DistributedCompositor_vert::source, fragSrc);
  renderer->bindShader(_shader);

  // TODO(croot): automate the texture image unit binding
  _shader->setSamplerTextureImageUnit("rgbaSampler", GL_TEXTURE0);
  _shader->setSamplerTextureImageUnit("id1ASampler", GL_TEXTURE1);
  _shader->setSamplerTextureImageUnit("id1BSampler", GL_TEXTURE2);
  _shader->setSamplerTextureImageUnit("id2Sampler", GL_TEXTURE3);

  _vao = rsrcMgr->createVertexArray({{_rectvbo, {}}});

  auto width = _framebufferPtr->getWidth();
  auto height = _framebufferPtr->getHeight();

  _rgbaTexture = QueryFramebuffer::createFboTexture2d(rsrcMgr, FboColorBuffer::COLOR_BUFFER, width, height, 1);
  _id1ATexture = QueryFramebuffer::createFboTexture2d(rsrcMgr, FboColorBuffer::ID1A_BUFFER, width, height, 1);
  if (supportsInt64) {
    _id1BTexture = QueryFramebuffer::createFboTexture2d(rsrcMgr, FboColorBuffer::ID1B_BUFFER, width, height, 1);
  }
  _id2Texture = QueryFramebuffer::createFboTexture2d(rsrcMgr, FboColorBuffer::ID2_BUFFER, width, height, 1);

  if (currRenderer) {
    currRenderer->makeActiveOnCurrentThread(currWindow);
  } else if (reset) {
    renderer->makeInactive();
  }
}

::Rendering::GL::GLRenderer* DistributedRenderBufferCompositor::getGLRenderer() {
  return _framebufferPtr->getGLRenderer();
}

size_t DistributedRenderBufferCompositor::getWidth() const {
  return (_rgbaTexture ? _rgbaTexture->getWidth() : 0);
}

size_t DistributedRenderBufferCompositor::getHeight() const {
  return (_rgbaTexture ? _rgbaTexture->getHeight() : 0);
}

void DistributedRenderBufferCompositor::resize(const size_t width, const size_t height) {
  _rgbaTexture->resize(width, height);
  _id1ATexture->resize(width, height);

  if (_id1BTexture) {
    _id1BTexture->resize(width, height);
  }

  _id2Texture->resize(width, height);

  // TODO(croot): do depth
}

void DistributedRenderBufferCompositor::render(const std::vector<RawPixelData>& pixelData) {
  auto renderer = _framebufferPtr->getGLRenderer();
  renderer->makeActiveOnCurrentThread();

  bool doHitTest = false;
  bool doDepthTest = false;
  size_t numBuffers = 0;
  int width = 0;
  int height = 0;

  // TODO(croot): handle depth buffering
  for (auto& bufferData : pixelData) {
    if (!bufferData.isEmpty()) {
      numBuffers += 1;

      if (numBuffers == 1) {
        width = bufferData.width;
        height = bufferData.height;
        if (bufferData.rowIdsA) {
          doHitTest = true;
        }
        _framebufferPtr->setHitTest(doHitTest);
        _framebufferPtr->setDepthTest(doDepthTest);
        _framebufferPtr->bindToRenderer(renderer);
        renderer->enable(GL_BLEND);
        renderer->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        renderer->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        renderer->setViewport(0, 0, width, height);
        renderer->setClearColor(0, 0, 0, 0);
        renderer->clearAll();

        renderer->bindShader(_shader);
        renderer->bindVertexArray(_vao);
      } else {
        CHECK((!doHitTest || (bufferData.rowIdsA && bufferData.tableIds)) && bufferData.width == width &&
              bufferData.height == height);
      }
      _rgbaTexture->uploadPixelsToTexture(
          bufferData.width, bufferData.height, GL_RGBA, GL_UNSIGNED_BYTE, bufferData.pixels.get());
      _shader->setSamplerAttribute("rgbaSampler", _rgbaTexture);

      if (doHitTest) {
        _id1ATexture->uploadPixelsToTexture(
            bufferData.width, bufferData.height, GL_RED_INTEGER, GL_UNSIGNED_INT, bufferData.rowIdsA.get());
        _shader->setSamplerAttribute("id1ASampler", _id1ATexture);
        if (_id1BTexture) {
          _id1BTexture->uploadPixelsToTexture(
              bufferData.width, bufferData.height, GL_RED_INTEGER, GL_UNSIGNED_INT, bufferData.rowIdsB.get());
          _shader->setSamplerAttribute("id1BSampler", _id1BTexture);
          _shader->setUniformAttribute("useIdBSampler", true);
        } else {
          _shader->setUniformAttribute("useIdBSampler", false);
        }
        _id2Texture->uploadPixelsToTexture(
            bufferData.width, bufferData.height, GL_RED_INTEGER, GL_INT, bufferData.tableIds.get());
        _shader->setSamplerAttribute("id2Sampler", _id2Texture);
      }
      renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);
    }
  }
  // TODO(croot): push/pop a state?
}

}  // namespace QueryRenderer
