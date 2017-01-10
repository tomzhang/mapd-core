#include "QueryFramebuffer.h"
#include "QueryRenderCompositor.h"
#include "QueryIdMapPixelBuffer.h"
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

QueryFramebuffer::QueryFramebuffer(GLRenderer* renderer,
                                   int width,
                                   int height,
                                   bool doHitTest,
                                   bool doDepthTest,
                                   size_t numSamples,
                                   const bool useInt64Id)
    : _defaultDoHitTest(doHitTest), _defaultDoDepthTest(doDepthTest), _doHitTest(doHitTest), _doDepthTest(doDepthTest) {
  _init(renderer, width, height, numSamples, useInt64Id);
}

QueryFramebuffer::QueryFramebuffer(QueryRenderCompositor* compositor,
                                   ::Rendering::GL::GLRenderer* renderer,
                                   const bool useInt64Id)
    : _defaultDoHitTest(compositor->doHitTest()),
      _defaultDoDepthTest(compositor->doDepthTest()),
      _doHitTest(compositor->doHitTest()),
      _doDepthTest(compositor->doDepthTest()) {
  _init(compositor, renderer, useInt64Id);
}

QueryFramebuffer::~QueryFramebuffer() {}

void QueryFramebuffer::_init(::Rendering::GL::GLRenderer* renderer,
                             int width,
                             int height,
                             size_t numSamples,
                             const bool useInt64Id) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = createFboTexture2d(rsrcMgr, FboColorBuffer::COLOR_BUFFER, width, height, numSamples);

  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_defaultDoHitTest) {
    _idTex1A = createFboTexture2d(rsrcMgr, FboColorBuffer::ID1A_BUFFER, width, height, numSamples);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex1A});

    if (useInt64Id) {
      _idTex1B = createFboTexture2d(rsrcMgr, FboColorBuffer::ID1B_BUFFER, width, height, numSamples);
      attachments.insert({GL_COLOR_ATTACHMENT2, _idTex1B});
    }

    _idTex2 = createFboTexture2d(rsrcMgr, FboColorBuffer::ID2_BUFFER, width, height, numSamples);
    attachments.insert({GL_COLOR_ATTACHMENT3, _idTex2});
  }

  if (_defaultDoDepthTest) {
    _rbo = createFboRenderbuffer(rsrcMgr, FboRenderBuffer::DEPTH_BUFFER, width, height, numSamples);
    attachments.insert({GL_DEPTH_ATTACHMENT, _rbo});
  }

  _fbo = rsrcMgr->createFramebuffer(attachments);
}

void QueryFramebuffer::_init(QueryRenderCompositor* compositor,
                             ::Rendering::GL::GLRenderer* renderer,
                             const bool useInt64Id) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = compositor->createFboTexture2d(renderer, FboColorBuffer::COLOR_BUFFER);
  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_defaultDoHitTest) {
    _idTex1A = compositor->createFboTexture2d(renderer, FboColorBuffer::ID1A_BUFFER);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex1A});

    if (useInt64Id) {
      _idTex1B = compositor->createFboTexture2d(renderer, FboColorBuffer::ID1B_BUFFER);
      attachments.insert({GL_COLOR_ATTACHMENT2, _idTex1B});
    }

    _idTex2 = compositor->createFboTexture2d(renderer, FboColorBuffer::ID2_BUFFER);
    attachments.insert({GL_COLOR_ATTACHMENT3, _idTex2});
  }

  if (_defaultDoDepthTest) {
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

size_t QueryFramebuffer::getNumSamples() const {
  return _fbo->getNumSamples();
}

void QueryFramebuffer::setHitTest(bool doHitTest) {
  if (doHitTest) {
    RUNTIME_EX_ASSERT(_defaultDoHitTest,
                      "Cannot activate hit testing for query framebuffer that wasn't initialized for hit-testing.");
  }
  _doHitTest = doHitTest;
}

void QueryFramebuffer::setDepthTest(bool doDepthTest) {
  if (doDepthTest) {
    RUNTIME_EX_ASSERT(_defaultDoDepthTest,
                      "Cannot activate depth testing for query framebuffer that wasn't initialized for depth-testing.");
  }
  _doDepthTest = doDepthTest;
}

::Rendering::Renderer* QueryFramebuffer::getRenderer() {
  return _fbo->getRenderer();
}

::Rendering::GL::GLRenderer* QueryFramebuffer::getGLRenderer() {
  return _fbo->getGLRenderer();
}

GLTexture2dShPtr QueryFramebuffer::getGLTexture2d(FboColorBuffer texType) const {
  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      return _rgbaTex;
    case FboColorBuffer::ID1A_BUFFER:
      return _idTex1A;
    case FboColorBuffer::ID1B_BUFFER:
      return _idTex1B;
    case FboColorBuffer::ID2_BUFFER:
      return _idTex2;
    default:
      CHECK(false);
  }
  return nullptr;
}

GLRenderbufferShPtr QueryFramebuffer::getGLRenderbuffer(FboRenderBuffer rboType) const {
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
    case FboColorBuffer::ID1A_BUFFER:
      return (_idTex1A ? _idTex1A->getId() : 0);
    case FboColorBuffer::ID1B_BUFFER:
      return (_idTex1B ? _idTex1B->getId() : 0);
    case FboColorBuffer::ID2_BUFFER:
      return (_idTex2 ? _idTex2->getId() : 0);
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

std::pair<std::vector<GLenum>, std::vector<GLenum>> QueryFramebuffer::getEnabledDisabledAttachments() const {
  std::vector<GLenum> enableAttachments = {};
  std::vector<GLenum> disableAttachments = {};

  if (_doHitTest) {
    enableAttachments.push_back(GL_COLOR_ATTACHMENT1);
    if (_idTex1B) {
      enableAttachments.push_back(GL_COLOR_ATTACHMENT2);
    } else {
      disableAttachments.push_back(GL_COLOR_ATTACHMENT2);
    }
    enableAttachments.push_back(GL_COLOR_ATTACHMENT3);
  } else if (_defaultDoHitTest) {
    disableAttachments.push_back(GL_COLOR_ATTACHMENT1);
    disableAttachments.push_back(GL_COLOR_ATTACHMENT2);
    disableAttachments.push_back(GL_COLOR_ATTACHMENT3);
  }

  if (_doDepthTest) {
    enableAttachments.push_back(GL_DEPTH_ATTACHMENT);
  } else if (_defaultDoDepthTest) {
    disableAttachments.push_back(GL_DEPTH_ATTACHMENT);
  }

  return {std::move(enableAttachments), std::move(disableAttachments)};
}

void QueryFramebuffer::bindToRenderer(GLRenderer* renderer, FboBind bindType) {
  renderer->bindFramebuffer(_fbo, bindType);

  if (bindType == FboBind::DRAW || bindType == FboBind::READ_AND_DRAW) {
    auto attachments = getEnabledDisabledAttachments();

    _fbo->enableAttachments(attachments.first);
    _fbo->disableAttachments(attachments.second);

    _fbo->activateEnabledAttachmentsForDrawing();
  }
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

void QueryFramebuffer::readIdBuffer(size_t startx,
                                    size_t starty,
                                    int width,
                                    int height,
                                    unsigned int* idBuffer,
                                    const FboColorBuffer idBufferType) {
  ::Rendering::GL::Resources::GLTexture2d* idTexToUse;
  GLenum attachmentToUse;

  switch (idBufferType) {
    case FboColorBuffer::ID1A_BUFFER:
      idTexToUse = _idTex1A.get();
      attachmentToUse = GL_COLOR_ATTACHMENT1;
      break;
    case FboColorBuffer::ID1B_BUFFER:
      idTexToUse = _idTex1B.get();
      attachmentToUse = GL_COLOR_ATTACHMENT2;
      break;
    case FboColorBuffer::ID2_BUFFER:
      idTexToUse = _idTex2.get();
      attachmentToUse = GL_COLOR_ATTACHMENT3;
      break;
    default:
      THROW_RUNTIME_EX("QueryFramebuffer:readIdBuffer: Unsupported id buffer type " +
                       std::to_string(static_cast<int>(idBufferType)));
      break;
  }

  RUNTIME_EX_ASSERT(idTexToUse != nullptr,
                    "QueryFramebuffer: The framebuffer was not setup to write an ID map. Cannot retrieve ID at pixel.");

  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  int widthToUse, heightToUse;

  if (clampWidthAndHeightToSrc(startx, starty, width, height, myWidth, myHeight, widthToUse, heightToUse)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the pixels to read ((x, y) = (" << startx << ", " << starty
                 << "), width = " << width << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << myWidth << ", height = " << myHeight
                 << "). Only pixels within the bounds will be read. The rest will be initialized to 0.";
  }

  std::memset(idBuffer, 0, width * height * sizeof(unsigned int));
  if (widthToUse > 0 && heightToUse > 0) {
    _fbo->readPixels(
        attachmentToUse, startx, starty, widthToUse, heightToUse, GL_RED_INTEGER, GL_UNSIGNED_INT, idBuffer);
  }
}

std::shared_ptr<unsigned int> QueryFramebuffer::readIdBuffer(size_t startx,
                                                             size_t starty,
                                                             int width,
                                                             int height,
                                                             const FboColorBuffer idBufferType) {
  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  if (width < 0) {
    width = myWidth;
  }

  if (height < 0) {
    height = myHeight;
  }

  std::shared_ptr<unsigned int> pixels(new unsigned int[width * height], std::default_delete<unsigned int[]>());
  unsigned int* rawPixels = pixels.get();

  readIdBuffer(startx, starty, width, height, rawPixels, idBufferType);

  return pixels;
}

void QueryFramebuffer::blitToFramebuffer(QueryFramebuffer& dstFboPtr,
                                         size_t startx,
                                         size_t starty,
                                         size_t width,
                                         size_t height) {
  CHECK(dstFboPtr.getGLRenderer() == getGLRenderer());

  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  CHECK(myWidth == dstFboPtr.getWidth() && myHeight == dstFboPtr.getHeight());

  _fbo->blitToFramebuffer(*(dstFboPtr.getGLFramebuffer()),
                          GL_COLOR_ATTACHMENT0,
                          startx,
                          starty,
                          width,
                          height,
                          GL_COLOR_ATTACHMENT0,
                          startx,
                          starty,
                          width,
                          height,
                          GL_NEAREST);

  if (_doHitTest) {
    _fbo->blitToFramebuffer(*(dstFboPtr.getGLFramebuffer()),
                            GL_COLOR_ATTACHMENT1,
                            startx,
                            starty,
                            width,
                            height,
                            GL_COLOR_ATTACHMENT1,
                            startx,
                            starty,
                            width,
                            height,
                            GL_NEAREST);

    _fbo->blitToFramebuffer(*(dstFboPtr.getGLFramebuffer()),
                            GL_COLOR_ATTACHMENT2,
                            startx,
                            starty,
                            width,
                            height,
                            GL_COLOR_ATTACHMENT2,
                            startx,
                            starty,
                            width,
                            height,
                            GL_NEAREST);
  }

  // if (_doDepthTest) {
  //   _fbo->blitToFramebuffer(GL_COLOR_ATTACHMENT0, dstFboPtr->getGLFramebuffer());
  // }
}

void QueryFramebuffer::copyRowIdBufferToPbo(QueryIdMapPixelBufferUIntShPtr& pbo, const bool leastSignificantBits) {
  RUNTIME_EX_ASSERT(pbo != nullptr, "Pbo is empty. Cannot copy pixels to an undefined pbo.");

  RUNTIME_EX_ASSERT(
      (leastSignificantBits && _idTex1A != nullptr) || (!leastSignificantBits && _idTex1B),
      "QueryFramebuffer: The framebuffer was not setup for an ID map. Cannot copy ID map to pixel buffer object.");

  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  size_t pboWidth = pbo->getWidth();
  size_t pboHeight = pbo->getHeight();

  RUNTIME_EX_ASSERT(pboWidth <= myWidth && pboHeight <= myHeight,
                    "The pbo for the idmap is too big for the framebuffer. It is " + std::to_string(pboWidth) + "x" +
                        std::to_string(pboHeight) + " and the fbo is " + std::to_string(myWidth) + "x" +
                        std::to_string(myHeight) + ". The pbo size needs to be <= the fbo size.");

  // TODO(croot): should we initialize the buffer to something? Like all 0s beforehand?
  // If so, see glClearBufferData()/glClearBufferSubData()
  CHECK(myWidth > 0 && myHeight > 0);

  GLRenderer* renderer = _fbo->getGLRenderer();
  renderer->bindWritePixelBuffer(pbo->getPixelBuffer2d());
  _fbo->copyPixelsToBoundPixelBuffer((leastSignificantBits ? GL_COLOR_ATTACHMENT1 : GL_COLOR_ATTACHMENT2),
                                     0,
                                     0,
                                     pboWidth,
                                     pboHeight,
                                     0,
                                     GL_RED_INTEGER,
                                     GL_UNSIGNED_INT);
  renderer->bindWritePixelBuffer(nullptr);
}

void QueryFramebuffer::copyTableIdBufferToPbo(QueryIdMapPixelBufferIntShPtr& pbo) {
  RUNTIME_EX_ASSERT(pbo != nullptr, "Pbo is empty. Cannot copy pixels to an undefined pbo.");

  RUNTIME_EX_ASSERT(
      _idTex2 != nullptr,
      "QueryFramebuffer: The framebuffer was not setup for an ID map. Cannot copy ID map to pixel buffer object.");

  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  size_t pboWidth = pbo->getWidth();
  size_t pboHeight = pbo->getHeight();

  RUNTIME_EX_ASSERT(pboWidth <= myWidth && pboHeight <= myHeight,
                    "The pbo for the idmap is too big for the framebuffer. It is " + std::to_string(pboWidth) + "x" +
                        std::to_string(pboHeight) + " and the fbo is " + std::to_string(myWidth) + "x" +
                        std::to_string(myHeight) + ". The pbo size needs to be <= the fbo size.");

  // TODO(croot): should we initialize the buffer to something? Like all 0s beforehand?
  // If so, see glClearBufferData()/glClearBufferSubData()
  CHECK(myWidth > 0 && myHeight > 0);

  GLRenderer* renderer = _fbo->getGLRenderer();
  renderer->bindWritePixelBuffer(pbo->getPixelBuffer2d());
  _fbo->copyPixelsToBoundPixelBuffer(GL_COLOR_ATTACHMENT3, 0, 0, pboWidth, pboHeight, 0, GL_RED_INTEGER, GL_INT);
  renderer->bindWritePixelBuffer(nullptr);
}

GLTexture2dShPtr QueryFramebuffer::createFboTexture2d(GLResourceManagerShPtr& rsrcMgr,
                                                      FboColorBuffer texType,
                                                      size_t width,
                                                      size_t height,
                                                      size_t numSamples) {
  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, numSamples);
    case FboColorBuffer::ID1A_BUFFER:
    case FboColorBuffer::ID1B_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, numSamples);
    case FboColorBuffer::ID2_BUFFER:
      // The ID2_BUFFER is used to store the table-id of the id stored in ID_BUFFER.
      // We need the table id hit-testing in multi-layer rendering.

      // TODO(croot): there is an issue where the ids are currently stored in a 32-bit unsigned integer
      // texture, but the id could certainly exceed the max number for a 32-bit uint.
      // A couple things we could do - 1) use a 64-bit int texture when fully supported (currently only
      // in an extension, I believe).
      // 2) use a second texture - since we already need a 2nd texture for the table id, and that probably only
      //    needs 8-16 bits, we could create a full 32-bit texture and use the remaining bits to pack
      //    more bits for the main id buffer.
      return rsrcMgr->createTexture2d(width, height, GL_R32I, GL_RED_INTEGER, GL_INT, numSamples);
    default:
      CHECK(false);
  }

  return nullptr;
}

GLRenderbufferShPtr QueryFramebuffer::createFboRenderbuffer(GLResourceManagerShPtr& rsrcMgr,
                                                            FboRenderBuffer rboType,
                                                            size_t width,
                                                            size_t height,
                                                            size_t numSamples) {
  switch (rboType) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return rsrcMgr->createRenderbuffer(width, height, GL_DEPTH_COMPONENT, numSamples);
    default:
      CHECK(false);
  }

  return nullptr;
}

}  // namespace QueryRenderer
