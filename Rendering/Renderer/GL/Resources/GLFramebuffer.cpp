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

void AttachmentContainer::addTexture2dAttachment(GLenum attachment, GLuint tex) {
  MAPD_CHECK_GL_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, tex, 0));

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
        _attachmentManager.addTexture2dAttachment(atchmnt, itr.second->getId());
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
  GLPixelBuffer2dShPtr pbo = renderer->getBoundReadPixelBuffer();
  RUNTIME_EX_ASSERT(pbo != nullptr, "No bound read pixel buffer. Cannot read pixels.");

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

  renderer->bindReadPixelBuffer(pbo);
  copyPixelsToBoundPixelBuffer(attachment, startx, starty, width, height, offsetBytes, format, type);
  renderer->bindReadPixelBuffer(currPbo);
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
