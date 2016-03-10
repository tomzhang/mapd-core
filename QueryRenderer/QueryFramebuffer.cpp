#include "QueryFramebuffer.h"
#include "QueryRenderCompositor.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <cstring>

// #include "MapDGL.h"
// #include "QueryRendererError.h"
// #include "QueryFramebuffer.h"
// #include <iostream>
// #include <stdexcept>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::FboBind;
using ::Rendering::GL::Resources::GLFramebufferAttachmentMap;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

static bool clampWidthAndHeightToSrc(size_t startx,
                                     size_t starty,
                                     int width,
                                     int height,
                                     size_t srcWidth,
                                     size_t srcHeight,
                                     int& clampedWidth,
                                     int& clampedHeight) {
  bool rtn = false;

  clampedWidth = (width < 0 ? srcWidth - startx : width);
  clampedHeight = (height < 0 ? srcHeight - starty : height);

  bool widthOverflow = (startx + clampedWidth > srcWidth);
  bool heightOverflow = (starty + clampedHeight > srcHeight);

  if (widthOverflow || heightOverflow) {
    rtn = true;

    if (widthOverflow) {
      clampedWidth = srcWidth - startx;
    }

    if (heightOverflow) {
      clampedHeight = srcHeight - starty;
    }
  }

  return rtn;
}

QueryFramebuffer::QueryFramebuffer(GLRenderer* renderer, int width, int height, bool doHitTest, bool doDepthTest)
    : _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _rgbaTex(nullptr),
      _idTex(nullptr),
      _rbo(nullptr),
      _fbo(nullptr) {
  _init(renderer, width, height);
}

QueryFramebuffer::QueryFramebuffer(QueryRenderCompositor* compositor, ::Rendering::GL::GLRenderer* renderer)
    : _doHitTest(compositor->doHitTest()),
      _doDepthTest(compositor->doDepthTest()),
      _rgbaTex(nullptr),
      _idTex(nullptr),
      _rbo(nullptr),
      _fbo(nullptr) {
  _init(compositor, renderer);
}

QueryFramebuffer::~QueryFramebuffer() {
}

void QueryFramebuffer::_init(::Rendering::GL::GLRenderer* renderer, int width, int height) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = createFboTexture2d(rsrcMgr, FboColorBuffer::COLOR_BUFFER, width, height);

  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_doHitTest) {
    _idTex = createFboTexture2d(rsrcMgr, FboColorBuffer::ID_BUFFER, width, height);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex});
  }

  if (_doDepthTest) {
    _rbo = createFboRenderbuffer(rsrcMgr, FboRenderBuffer::DEPTH_BUFFER, width, height);
    attachments.insert({GL_DEPTH_ATTACHMENT, _rbo});
  }

  _fbo = rsrcMgr->createFramebuffer(attachments);
}

void QueryFramebuffer::_init(QueryRenderCompositor* compositor, ::Rendering::GL::GLRenderer* renderer) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = compositor->createFboTexture2d(renderer, FboColorBuffer::COLOR_BUFFER);
  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_doHitTest) {
    _idTex = compositor->createFboTexture2d(renderer, FboColorBuffer::ID_BUFFER);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex});
  }

  if (_doDepthTest) {
    _rbo = compositor->createFboRenderbuffer(renderer, FboRenderBuffer::DEPTH_BUFFER);
    attachments.insert({GL_DEPTH_ATTACHMENT, _rbo});
  }

  _fbo = rsrcMgr->createFramebuffer(attachments);
}

void QueryFramebuffer::resize(int width, int height) {
  CHECK(_fbo);
  _fbo->resize(width, height);
}

size_t QueryFramebuffer::getWidth() const {
  return _fbo->getWidth();
}

size_t QueryFramebuffer::getHeight() const {
  return _fbo->getHeight();
}

::Rendering::Renderer* QueryFramebuffer::getRenderer() {
  return _fbo->getRenderer();
}

GLTexture2dShPtr QueryFramebuffer::getColorTexture2d(FboColorBuffer texType) {
  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      return _rgbaTex;
    case FboColorBuffer::ID_BUFFER:
      return _idTex;
    default:
      CHECK(false);
  }
  return nullptr;
}

GLRenderbufferShPtr QueryFramebuffer::getRenderbuffer(FboRenderBuffer rboType) {
  switch (rboType) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return _rbo;
    default:
      CHECK(false);
  }

  return nullptr;
}

GLuint QueryFramebuffer::getId(FboColorBuffer buffer) {
  switch (buffer) {
    case FboColorBuffer::COLOR_BUFFER:
      return (_rgbaTex ? _rgbaTex->getId() : 0);
    case FboColorBuffer::ID_BUFFER:
      return (_idTex ? _idTex->getId() : 0);
    default:
      return 0;
  }
  return 0;
}

GLuint QueryFramebuffer::getId(FboRenderBuffer buffer) {
  switch (buffer) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return (_rbo ? _rbo->getId() : 0);
    default:
      return 0;
  }

  return 0;
}

void QueryFramebuffer::bindToRenderer(GLRenderer* renderer, FboBind bindType) {
  renderer->bindFramebuffer(bindType, _fbo);
}

std::shared_ptr<unsigned char> QueryFramebuffer::readColorBuffer(size_t startx, size_t starty, int width, int height) {
  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  if (width < 0) {
    width = myWidth;
  }

  if (height < 0) {
    height = myHeight;
  }

  int widthToUse, heightToUse;
  if (clampWidthAndHeightToSrc(startx, starty, width, height, myWidth, myHeight, widthToUse, heightToUse)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the pixels to read ((x, y) = (" << startx << ", " << starty
                 << "), width = " << width << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << myWidth << ", height = " << myHeight
                 << "). Only pixels within the bounds will be read. The rest will be initialized to (0,0,0,0).";
  }

  std::shared_ptr<unsigned char> pixels(new unsigned char[width * height * 4], std::default_delete<unsigned char[]>());
  unsigned char* rawPixels = pixels.get();
  std::memset(rawPixels, 0, width * height * 4 * sizeof(unsigned char));

  if (widthToUse > 0 && heightToUse > 0) {
    _fbo->readPixels(
        GL_COLOR_ATTACHMENT0, startx, starty, widthToUse, heightToUse, GL_RGBA, GL_UNSIGNED_BYTE, rawPixels);
  }

  return pixels;
}

std::shared_ptr<unsigned int> QueryFramebuffer::readIdBuffer(size_t startx, size_t starty, int width, int height) {
  RUNTIME_EX_ASSERT(_idTex != nullptr,
                    "QueryFramebuffer: The framebuffer was not setup to write an ID map. Cannot retrieve ID at pixel.");

  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  if (width < 0) {
    width = myWidth;
  }

  if (height < 0) {
    height = myHeight;
  }

  int widthToUse, heightToUse;

  if (clampWidthAndHeightToSrc(startx, starty, width, height, myWidth, myHeight, widthToUse, heightToUse)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the pixels to read ((x, y) = (" << startx << ", " << starty
                 << "), width = " << width << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << myWidth << ", height = " << myHeight
                 << "). Only pixels within the bounds will be read. The rest will be initialized to 0.";
  }

  std::shared_ptr<unsigned int> pixels(new unsigned int[width * height], std::default_delete<unsigned int[]>());
  unsigned int* rawPixels = pixels.get();
  std::memset(rawPixels, 0, width * height * sizeof(unsigned int));

  if (widthToUse > 0 && heightToUse > 0) {
    _fbo->readPixels(
        GL_COLOR_ATTACHMENT1, startx, starty, widthToUse, heightToUse, GL_RED_INTEGER, GL_UNSIGNED_INT, rawPixels);
  }

  return pixels;
}

GLTexture2dShPtr QueryFramebuffer::createFboTexture2d(GLResourceManagerShPtr& rsrcMgr,
                                                      FboColorBuffer texType,
                                                      size_t width,
                                                      size_t height) {
  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    case FboColorBuffer::ID_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    default:
      CHECK(false);
  }

  return nullptr;
}

GLRenderbufferShPtr QueryFramebuffer::createFboRenderbuffer(GLResourceManagerShPtr& rsrcMgr,
                                                            FboRenderBuffer rboType,
                                                            size_t width,
                                                            size_t height) {
  switch (rboType) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return rsrcMgr->createRenderbuffer(width, height, GL_DEPTH_COMPONENT);
    default:
      CHECK(false);
  }

  return nullptr;
}

}  // namespace QueryRenderer
