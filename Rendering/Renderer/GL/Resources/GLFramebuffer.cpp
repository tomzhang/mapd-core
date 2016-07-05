#include "../MapDGL.h"
#include "../../../RenderError.h"
#include "GLFramebuffer.h"
#include "GLTexture2d.h"
#include "GLRenderbuffer.h"
#include "GLPixelBuffer2d.h"

namespace Rendering {
namespace GL {
namespace Resources {

namespace detail {

AttachmentContainer::AttachmentContainer() {
  // need to initialize the number of color attachments for the system
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &_maxColorAttachments));
}

AttachmentContainer::~AttachmentContainer() {
}

inline bool AttachmentContainer::isColorAttachment(GLenum attachment, int maxColorAttachments) {
  return static_cast<int>(attachment - GL_COLOR_ATTACHMENT0) < maxColorAttachments;
}

bool AttachmentContainer::hasAttachment(GLenum attachment) {
  return (_attachmentMap.find(attachment) != _attachmentMap.end());
}

void AttachmentContainer::addTexture2dAttachment(GLenum attachment, GLuint tex, GLenum target) {
  MAPD_CHECK_GL_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, target, tex, 0));

  AttachmentData data = {attachment, tex, true};
  _attachmentMap.insert(data);

  if (isColorAttachment(attachment, _maxColorAttachments)) {
    _dirty = true;
  }
}

void AttachmentContainer::addRenderbufferAttachment(GLenum attachment, GLuint rbo) {
  MAPD_CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rbo));

  AttachmentData data = {attachment, rbo, true};
  _attachmentMap.insert(data);

  if (isColorAttachment(attachment, _maxColorAttachments)) {
    _dirty = true;
  }
}

void AttachmentContainer::removeAttachment(GLenum attachment) {
  _attachmentMap.erase(attachment);
  if (isColorAttachment(attachment, _maxColorAttachments)) {
    _dirty = true;
  }
}

void AttachmentContainer::enableAllAttachments() {
  ChangeActive enabled(true);
  for (auto itr = _attachmentMap.begin(); itr != _attachmentMap.end(); ++itr) {
    if (!itr->active) {
      _attachmentMap.modify(itr, enabled);
      if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _dirty = true;
      }
    }
  }
}

void AttachmentContainer::disableAllAttachments() {
  ChangeActive disabled(false);
  for (auto itr = _attachmentMap.begin(); itr != _attachmentMap.end(); ++itr) {
    if (itr->active) {
      _attachmentMap.modify(itr, disabled);
      if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _dirty = true;
      }
    }
  }
}

void AttachmentContainer::enableAttachments(const std::vector<GLenum>& attachments) {
  ChangeActive enabled(true);
  for (auto attachment : attachments) {
    auto itr = _attachmentMap.find(attachment);
    if (itr != _attachmentMap.end() && !itr->active) {
      _attachmentMap.modify(itr, enabled);

      if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _dirty = true;
      }
    }
  }
}

void AttachmentContainer::disableAttachments(const std::vector<GLenum>& attachments) {
  ChangeActive disabled(false);
  for (auto attachment : attachments) {
    auto itr = _attachmentMap.find(attachment);
    if (itr != _attachmentMap.end() && itr->active) {
      _attachmentMap.modify(itr, disabled);

      if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _dirty = true;
      }
    }
  }
}

void AttachmentContainer::enableAttachment(GLenum attachment) {
  ChangeActive enabled(true);
  auto itr = _attachmentMap.find(attachment);
  if (itr != _attachmentMap.end() && !itr->active) {
    _attachmentMap.modify(itr, enabled);

    if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
      _dirty = true;
    }
  }
}

void AttachmentContainer::disableAttachment(GLenum attachment) {
  ChangeActive disabled(false);
  auto itr = _attachmentMap.find(attachment);
  if (itr != _attachmentMap.end() && itr->active) {
    _attachmentMap.modify(itr, disabled);

    if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
      _dirty = true;
    }
  }
}

void AttachmentContainer::enableGLAttachments() {
  // TODO(croot): what to do if the active attachments are empty?

  if (_dirty) {
    _activeAttachments.clear();

    AttachmentMap_in_order& inOrder = _attachmentMap.get<inorder>();
    AttachmentMap_in_order::iterator itr;
    for (itr = inOrder.begin(); itr != inOrder.end(); ++itr) {
      if (itr->active && isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _activeAttachments.push_back(itr->attachmentType);
      }
    }

    if (!_activeAttachments.size()) {
      _activeAttachments.push_back(GL_NONE);
    }

    _dirty = false;
  }

  MAPD_CHECK_GL_ERROR(glDrawBuffers(_activeAttachments.size(), &_activeAttachments[0]));
}

void AttachmentContainer::clear() {
  _activeAttachments.clear();
  _attachmentMap.clear();
  _dirty = true;
}

}  // namespace detail

GLFramebuffer::GLFramebuffer(const RendererWkPtr& rendererPtr, const GLFramebufferAttachmentMap& attachments)
    : GLResource(rendererPtr, GLResourceType::FRAMEBUFFER), _fbo(0), _textureBuffers(), _renderBuffers() {
  _initResource(attachments);
}

GLFramebuffer::~GLFramebuffer() {
  cleanupResource();
}

void GLFramebuffer::_initResource(const GLFramebufferAttachmentMap& attachments) {
  validateRenderer(__FILE__, __LINE__);

  GLint currReadFbo;
  GLint currDrawFbo;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &currReadFbo));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currDrawFbo));

  MAPD_CHECK_GL_ERROR(glGenFramebuffers(1, &_fbo));
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, _fbo));

  for (auto& itr : attachments) {
    GLenum atchmnt = itr.first;
    GLResourceType rsrcType = itr.second->getResourceType();
    switch (rsrcType) {
      case GLResourceType::TEXTURE_2D:
        _attachmentManager.addTexture2dAttachment(atchmnt, itr.second->getId(), itr.second->getTarget());
        _textureBuffers.push_back(std::static_pointer_cast<GLTexture2d>(itr.second));
        break;
      case GLResourceType::RENDERBUFFER:
        _attachmentManager.addRenderbufferAttachment(atchmnt, itr.second->getId());
        _renderBuffers.push_back(std::static_pointer_cast<GLRenderbuffer>(itr.second));
        break;
      default:
        LOG(WARNING) << "Invalid resource type " << rsrcType << " to bind to framebuffer at attachmet " << atchmnt
                     << ". Skipping attachment.";
        break;
    }
  }

  _attachmentManager.enableGLAttachments();

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

  if (status != GL_FRAMEBUFFER_COMPLETE) {
    std::stringstream ss;
    ss << "Framebuffer error: 0x" << std::hex << status << ".";

    switch (status) {
      case GL_FRAMEBUFFER_UNDEFINED:
        ss << " The default framebuffer does not exist.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        ss << " One or more of the framebuffer attachments are incomplete.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        ss << " The framebuffer does not have any images attached to it.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        ss << " The framebuffer has an undefined type for one or more of the color attachments for drawing.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        ss << " The framebuffer has an undefined type for one or more of the color attachments for reading.";
        break;
      case GL_FRAMEBUFFER_UNSUPPORTED:
        ss << " The combination of internal formats for the attachments is unsupported.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        ss << " The value of GL_RENDERBUFFER_SAMPLES, GL_TEXTURE_SAMPLES, GL_TEXTURE_FIXED_SAMPLE_LOCATIONS, and/or "
              "GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached "
              "renderbuffers and textures.";
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        ss << " A framebuffer attachment is layered, or all color attachments are not from textures of the same target";
        break;
      default:
        ss << " Encountered an undefined framebuffer error.";
        break;
    }

    THROW_RUNTIME_EX(ss.str());
  }

  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, currReadFbo));
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currDrawFbo));

  setUsable();
}

void GLFramebuffer::_cleanupResource() {
  if (_fbo) {
    MAPD_CHECK_GL_ERROR(glDeleteFramebuffers(1, &_fbo));
  }
  _makeEmpty();
}

void GLFramebuffer::_makeEmpty() {
  _fbo = 0;
  _textureBuffers.clear();
  _renderBuffers.clear();
  _attachmentManager.clear();
}

size_t GLFramebuffer::getWidth() const {
  if (_textureBuffers.size()) {
    return _textureBuffers[0]->getWidth();
  } else if (_renderBuffers.size()) {
    return _renderBuffers[0]->getWidth();
  }
  return 0;
}

size_t GLFramebuffer::getHeight() const {
  if (_textureBuffers.size()) {
    return _textureBuffers[0]->getHeight();
  } else if (_renderBuffers.size()) {
    return _renderBuffers[0]->getHeight();
  }
  return 0;
}

size_t GLFramebuffer::getNumSamples() const {
  if (_textureBuffers.size()) {
    return _textureBuffers[0]->getNumSamples();
  } else if (_renderBuffers.size()) {
    return _renderBuffers[0]->getNumSamples();
  }
  return 0;
}

void GLFramebuffer::readPixels(GLenum attachment,
                               size_t startx,
                               size_t starty,
                               size_t width,
                               size_t height,
                               GLenum format,
                               GLenum type,
                               GLvoid* data) {
  // TODO(croot): should we worry about which renderer here? Or is always defaulting to
  // the parent renderer ok? If it isn't, we either need to pass the renderer as a default
  // argument, or supply an api to the renderer directly, which will then call this function
  // which would be private.

  RUNTIME_EX_ASSERT(
      _attachmentManager.hasAttachment(attachment),
      "Error trying to read pixels from fbo attachment " + std::to_string(attachment) + ". Attachment doesn't exist.");

  validateUsability(__FILE__, __LINE__);

  GLint currReadBuffer, currReadFbo;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &currReadFbo));

  bool bind = (currReadFbo != static_cast<GLint>(_fbo));
  if (bind) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo));
  }

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_BUFFER, &currReadBuffer));
  MAPD_CHECK_GL_ERROR(glReadBuffer(attachment));
  MAPD_CHECK_GL_ERROR(glReadPixels(startx, starty, width, height, format, type, data));

  if (currReadBuffer != static_cast<GLint>(attachment)) {
    MAPD_CHECK_GL_ERROR(glReadBuffer(currReadBuffer));
  }

  if (bind) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, currReadFbo));
  }
}

void GLFramebuffer::blitToFramebuffer(GLFramebuffer& dstFbo,
                                      GLenum srcAttachment,
                                      size_t srcX,
                                      size_t srcY,
                                      size_t srcWidth,
                                      size_t srcHeight,
                                      GLenum dstAttachment,
                                      size_t dstX,
                                      size_t dstY,
                                      size_t dstWidth,
                                      size_t dstHeight,
                                      GLenum filter) {
  RUNTIME_EX_ASSERT(hasAttachment(srcAttachment),
                    "Error trying to blit fbo attachment " + std::to_string(srcAttachment) +
                        ". Attachment doesn't exist in src fbo.");

  RUNTIME_EX_ASSERT(dstFbo.hasAttachment(dstAttachment),
                    "Error trying to blit fbo attachment " + std::to_string(dstAttachment) +
                        ". Attachment doesn't exist in dst fbo.");

  bool isSrcColor =
      detail::AttachmentContainer::isColorAttachment(srcAttachment, _attachmentManager._maxColorAttachments);
  bool isDstColor =
      detail::AttachmentContainer::isColorAttachment(dstAttachment, dstFbo._attachmentManager._maxColorAttachments);

  RUNTIME_EX_ASSERT(isSrcColor == isDstColor,
                    "Error trying to blit an fbo to another fbo. The attachments " + std::to_string(srcAttachment) +
                        " and " + std::to_string(dstAttachment) + " are not compatible types.");

  if (!isSrcColor) {
    RUNTIME_EX_ASSERT(srcAttachment == dstAttachment,
                      "Error trying to blit an fbo to another fbo. The attachments " + std::to_string(srcAttachment) +
                          " and " + std::to_string(dstAttachment) + " are not compatible types.");
  }

  RUNTIME_EX_ASSERT(srcWidth > 0 && srcHeight > 0 && dstWidth > 0 && dstHeight > 0,
                    "Error trying to blit fbos: Invalid dimensions: " + std::to_string(srcWidth) + "x" +
                        std::to_string(srcHeight) + " blit to " + std::to_string(dstWidth) + "x" +
                        std::to_string(dstHeight));

  auto srcX0 = srcX + srcWidth;
  auto srcY0 = srcY + srcHeight;

  auto dstX0 = dstX + dstWidth;
  auto dstY0 = dstY + dstHeight;

  auto srcFullWidth = getWidth();
  auto srcFullHeight = getHeight();
  auto dstFullWidth = dstFbo.getWidth();
  auto dstFullHeight = dstFbo.getHeight();

  RUNTIME_EX_ASSERT(srcX <= srcFullWidth && srcX0 <= srcFullWidth && srcY <= srcFullHeight && srcY0 <= srcFullHeight,
                    "Error trying to blit fbos: src bounds extend beyond the bounds of the fbo - startPt: [" +
                        std::to_string(srcX) + ", " + std::to_string(srcY) + "], endPt: [" + std::to_string(srcX0) +
                        ", " + std::to_string(srcY0) + "], dims: " + std::to_string(srcFullWidth) + "x" +
                        std::to_string(srcFullHeight));

  RUNTIME_EX_ASSERT(dstX <= dstFullWidth && dstX0 <= dstFullWidth && dstY <= dstFullHeight && dstY0 <= dstFullHeight,
                    "Error trying to blit fbos: dst bounds extend beyond the bounds of the fbo - startPt: [" +
                        std::to_string(dstX) + ", " + std::to_string(dstY) + "], endPt: [" + std::to_string(dstX0) +
                        ", " + std::to_string(dstY0) + "], dims: " + std::to_string(dstFullWidth) + "x" +
                        std::to_string(dstFullHeight));

  GLbitfield mask;

  if (isSrcColor) {
    mask = GL_COLOR_BUFFER_BIT;
  } else {
    switch (srcAttachment) {
      case GL_DEPTH_ATTACHMENT:
        mask = GL_DEPTH_BUFFER_BIT;
        break;
      case GL_STENCIL_ATTACHMENT:
        mask = GL_STENCIL_BUFFER_BIT;
        break;
      case GL_DEPTH_STENCIL_ATTACHMENT:
        mask = GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
        break;
      default:
        THROW_RUNTIME_EX("Error trying to blit fbos: unsupported attachment " + std::to_string(srcAttachment));
    }
  }

  validateUsability(__FILE__, __LINE__);
  dstFbo.validateUsability(__FILE__, __LINE__);

  GLint currReadBuffer, currReadFbo;
  GLint currDrawBuffer, currDrawFbo;
  GLint srcFboId = static_cast<GLint>(_fbo);
  GLint dstFboId = static_cast<GLint>(dstFbo.getId());
  bool bindSrc, bindDst;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &currReadFbo));
  bindSrc = (currReadFbo != srcFboId);
  if (bindSrc) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, srcFboId));
  }
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_BUFFER, &currReadBuffer));

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currDrawFbo));
  bindDst = (currDrawFbo != dstFboId);
  if (bindDst) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstFboId));
  }
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_DRAW_BUFFER, &currDrawBuffer));

  MAPD_CHECK_GL_ERROR(glReadBuffer(srcAttachment));
  MAPD_CHECK_GL_ERROR(glDrawBuffer(dstAttachment));

  MAPD_CHECK_GL_ERROR(glBlitFramebuffer(srcX, srcY, srcX0, srcY0, dstX, dstY, dstX0, dstY0, mask, filter));

  GLenum err = glGetError();

  if (currReadBuffer != static_cast<GLint>(srcAttachment)) {
    MAPD_CHECK_GL_ERROR(glReadBuffer(currReadBuffer));
  }

  if (bindSrc) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, currReadFbo));
  }

  if (currDrawBuffer != static_cast<GLint>(dstAttachment)) {
    MAPD_CHECK_GL_ERROR(glDrawBuffer(currDrawBuffer));
  }

  if (bindDst) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currDrawFbo));
  }

  RUNTIME_EX_ASSERT(err == GL_NO_ERROR, "Error trying to blit framebuffers: " + getGLErrorStr(err));
}

void GLFramebuffer::copyPixelsToBoundPixelBuffer(GLenum attachment,
                                                 size_t startx,
                                                 size_t starty,
                                                 size_t width,
                                                 size_t height,
                                                 size_t offsetBytes,
                                                 GLenum format,
                                                 GLenum type) {
  // TODO(croot): should we worry about which renderer here? Or is always defaulting to
  // the parent renderer ok? If it isn't, we either need to pass the renderer as a default
  // argument, or supply an api to the renderer directly, which will then call this function
  // which would be private.

  RUNTIME_EX_ASSERT(
      _attachmentManager.hasAttachment(attachment),
      "Error trying to read pixels from fbo attachment " + std::to_string(attachment) + ". Attachment doesn't exist.");

  validateUsability(__FILE__, __LINE__);

  GLRenderer* renderer = getGLRenderer();
  GLPixelBuffer2dShPtr pbo = renderer->getBoundWritePixelBuffer();
  RUNTIME_EX_ASSERT(pbo != nullptr, "No bound write pixel buffer. Cannot copy pixels to pixel buffer.");

  // size_t myWidth = getWidth();
  // size_t myHeight = getHeight();

  // size_t pboWidth = pbo->getWidth();
  // size_t pboHeight = pbo->getHeight();
  // RUNTIME_EX_ASSERT(pboWidth == myWidth && pboHeight == myHeight,
  //                   "GLFramebuffer dimensions: " + std::to_string(myWidth) + "x" + std::to_string(myHeight) +
  //                       " does not match the dimensions of the bound pixel buffer: " + std::to_string(pboWidth) + "x"
  //                       +
  //                       std::to_string(pboHeight));

  GLint currReadBuffer, currReadFbo;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &currReadFbo));
  bool fboBind = (currReadFbo != static_cast<GLint>(_fbo));
  if (fboBind) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo));
  }

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_BUFFER, &currReadBuffer));
  MAPD_CHECK_GL_ERROR(glReadBuffer(attachment));

  MAPD_CHECK_GL_ERROR(glReadPixels(startx, starty, width, height, format, type, BUFFER_OFFSET(offsetBytes)));

  if (currReadBuffer != static_cast<GLint>(attachment)) {
    MAPD_CHECK_GL_ERROR(glReadBuffer(currReadBuffer));
  }

  if (fboBind) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, currReadFbo));
  }
}

void GLFramebuffer::copyPixelsToPixelBuffer(GLenum attachment,
                                            size_t startx,
                                            size_t starty,
                                            size_t width,
                                            size_t height,
                                            size_t offsetBytes,
                                            GLenum format,
                                            GLenum type,
                                            GLPixelBuffer2dShPtr& pbo) {
  GLRenderer* renderer = getGLRenderer();
  GLPixelBuffer2dShPtr currPbo = renderer->getBoundReadPixelBuffer();

  renderer->bindWritePixelBuffer(pbo);
  copyPixelsToBoundPixelBuffer(attachment, startx, starty, width, height, offsetBytes, format, type);
  renderer->bindWritePixelBuffer(currPbo);
}

void GLFramebuffer::resize(size_t width, size_t height) {
  // TODO(croot): should we worry about which renderer here? Or is always defaulting to
  // the parent renderer ok?

  for (auto& tex : _textureBuffers) {
    tex->resize(width, height);
  }

  for (auto& rb : _renderBuffers) {
    rb->resize(width, height);
  }
}

bool GLFramebuffer::hasAttachment(GLenum attachment) {
  return _attachmentManager.hasAttachment(attachment);
}

void GLFramebuffer::enableAllAttachments() {
  _attachmentManager.enableAllAttachments();
}

void GLFramebuffer::disableAllAttachments() {
  _attachmentManager.disableAllAttachments();
}

void GLFramebuffer::enableAttachments(const std::vector<GLenum>& activeAttachments) {
  _attachmentManager.enableAttachments(activeAttachments);
}

void GLFramebuffer::disableAttachments(const std::vector<GLenum>& attachmentsToDisable) {
  _attachmentManager.disableAttachments(attachmentsToDisable);
}

void GLFramebuffer::enableAttachment(GLenum attachment) {
  _attachmentManager.enableAttachment(attachment);
}

void GLFramebuffer::disableAttachment(GLenum attachment) {
  _attachmentManager.disableAttachment(attachment);
}

void GLFramebuffer::activateEnabledAttachmentsForDrawing() {
  _attachmentManager.enableGLAttachments();
}

// void GLFramebuffer::bindToRenderer(GLRenderer* renderer, FboBind bindType) {
//   // TODO(croot): perhaps write a "don't check renderer" version?
//   validateUsability(renderer);

//   MAPD_CHECK_GL_ERROR(glBindFramebuffer(static_cast<int>(bindType), _fbo));
//   // _attachmentManager.enableGLAttachments();

//   // TODO(croot): How do we properly deal with the state machine
//   // if (_attachmentManager.hasAttachment(GL_DEPTH_ATTACHMENT)) {
//   //   MAPD_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
//   // }
// }

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
