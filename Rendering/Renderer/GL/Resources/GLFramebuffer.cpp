#include "../MapDGL.h"
#include "../../../RenderError.h"
#include "GLFramebuffer.h"
#include "GLTexture2d.h"
#include "GLRenderbuffer.h"

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

  AttachmentData data = {attachment, tex};
  _attachmentMap.insert(data);

  if (isColorAttachment(attachment, _maxColorAttachments)) {
    _dirty = true;
  }
}

void AttachmentContainer::addRenderbufferAttachment(GLenum attachment, GLuint rbo) {
  MAPD_CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rbo));

  AttachmentData data = {attachment, rbo};
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

void AttachmentContainer::enableAttachments() {
  // TODO(croot): what to do if the active attachments are empty?

  if (_dirty) {
    _activeAttachments.clear();

    AttachmentMap_in_order& inOrder = _attachmentMap.get<inorder>();
    AttachmentMap_in_order::iterator itr;
    for (itr = inOrder.begin(); itr != inOrder.end(); ++itr) {
      if (isColorAttachment(itr->attachmentType, _maxColorAttachments)) {
        _activeAttachments.push_back(itr->attachmentType);
      }
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
    : GLResource(rendererPtr), _fbo(0), _textureBuffers(), _renderBuffers() {
  _initResource(attachments);
}

GLFramebuffer::~GLFramebuffer() {
  cleanupResource();
}

void GLFramebuffer::_initResource(const GLFramebufferAttachmentMap& attachments) {
  validateRenderer();

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

  _attachmentManager.enableAttachments();

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

  validateUsability();

  // size_t myWidth = getWidth();
  // size_t myHeight = getHeight();
  // RUNTIME_EX_ASSERT(startx + width <= myWidth && starty + height <= myHeight,
  //                   "GLFramebuffer: bounds of the pixels to read ((x, y) = (" + std::to_string(startx) + ", " +
  //                       std::to_string(starty) + "), width = " + std::to_string(width) + ", height = " +
  //                       std::to_string(height) + ") extends beyond the bounds of the framebuffer (width = " +
  //                       std::to_string(myWidth) + ", height = " + std::to_string(myHeight) + "). Cannot read
  //                       pixels.");

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

// void GLFramebuffer::bindToRenderer(GLRenderer* renderer, FboBind bindType) {
//   // TODO(croot): perhaps write a "don't check renderer" version?
//   validateUsability(renderer);

//   MAPD_CHECK_GL_ERROR(glBindFramebuffer(static_cast<int>(bindType), _fbo));
//   // _attachmentManager.enableAttachments();

//   // TODO(croot): How do we properly deal with the state machine
//   // if (_attachmentManager.hasAttachment(GL_DEPTH_ATTACHMENT)) {
//   //   MAPD_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
//   // }
// }

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
