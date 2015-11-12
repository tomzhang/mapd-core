#include "MapDGL.h"
#include "QueryRendererError.h"
#include "QueryFramebuffer.h"
#include <iostream>
#include <stdexcept>

using namespace MapD_Renderer;

struct Texture2DSamplingProperties {
  GLint minFilter;
  GLint magFilter;
  GLint wrapS;
  GLint wrapT;
};

void resizeTexture2D(GLuint texture,
                     GLint width,
                     GLint height,
                     GLint internalFormat,
                     GLenum pixelFormat,
                     GLenum pixelType) {
  MAPD_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, texture));
  MAPD_CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, pixelType, NULL));
}

GLuint generateTexture2D(GLint width, GLint height, GLint internalFormat, GLenum pixelFormat, GLenum pixelType) {
  GLuint tex;
  MAPD_CHECK_GL_ERROR(glGenTextures(1, &tex));
  resizeTexture2D(tex, width, height, internalFormat, pixelFormat, pixelType);
  return tex;
}

GLuint generateTexture2D(GLint width,
                         GLint height,
                         GLint internalFormat,
                         GLenum pixelFormat,
                         GLenum pixelType,
                         const Texture2DSamplingProperties& samplingProps) {
  GLuint tex = generateTexture2D(width, height, internalFormat, pixelFormat, pixelType);

  // TODO - use a sampler object instead
  MAPD_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, samplingProps.magFilter));
  MAPD_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, samplingProps.minFilter));
  MAPD_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, samplingProps.wrapS));
  MAPD_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, samplingProps.wrapT));

  return tex;
}

void resizeRenderbuffer(GLuint rbo, GLint width, GLint height, GLint internalFormat) {
  MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, rbo));
  MAPD_CHECK_GL_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height));
}

GLuint generateRenderBuffer(GLint width, GLint height, GLint internalFormat) {
  GLuint rbo;

  MAPD_CHECK_GL_ERROR(glGenRenderbuffers(1, &rbo));
  resizeRenderbuffer(rbo, width, height, internalFormat);

  return rbo;
}

int AttachmentContainer::numColorAttachments = -1;

AttachmentContainer::AttachmentContainer() {
  // need to initialize the number of color attachments for the system
  if (numColorAttachments < 0) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &numColorAttachments));
  }
}

AttachmentContainer::~AttachmentContainer() {
  // std::cerr << "IN AttachmentContainer DESTRUCTOR" << std::endl;
}

inline bool AttachmentContainer::isColorAttachment(GLenum attachment) {
  return static_cast<int>(attachment - GL_COLOR_ATTACHMENT0) < numColorAttachments;
}

void AttachmentContainer::addTexture2dAttachment(GLenum attachment, GLuint tex) {
  MAPD_CHECK_GL_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, tex, 0));

  AttachmentData data = {attachment, tex};
  _attachmentMap.insert(data);

  if (isColorAttachment(attachment)) {
    _dirty = true;
  }
}

void AttachmentContainer::addRenderbufferAttachment(GLenum attachment, GLuint rbo) {
  MAPD_CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rbo));

  AttachmentData data = {attachment, rbo};
  _attachmentMap.insert(data);

  if (isColorAttachment(attachment)) {
    _dirty = true;
  }
}

void AttachmentContainer::removeAttachment(GLenum attachment) {
  _attachmentMap.erase(attachment);
  if (isColorAttachment(attachment)) {
    _dirty = true;
  }
}

void AttachmentContainer::enableAttachments() {
  if (_dirty) {
    _activeAttachments.clear();

    AttachmentMap_in_order& inOrder = _attachmentMap.get<inorder>();
    AttachmentMap_in_order::iterator itr;
    for (itr = inOrder.begin(); itr != inOrder.end(); ++itr) {
      if (isColorAttachment(itr->attachmentType)) {
        _activeAttachments.push_back(itr->attachmentType);
      }
    }

    _dirty = false;
  }

  MAPD_CHECK_GL_ERROR(glDrawBuffers(_activeAttachments.size(), &_activeAttachments[0]));
}

QueryFramebuffer::QueryFramebuffer(int width, int height, bool doHitTest, bool doDepthTest)
    : _width(width),
      _height(height),
      _fbo(0),
      _textureBuffers(MAX_TEXTURE_BUFFERS + 1, 0),
      _renderBuffers(MAX_RENDER_BUFFERS + 1, 0) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0, "Invalid dimensions for the framebuffer. Dimensions must be > 0");
  _init(doHitTest, doDepthTest);
}

QueryFramebuffer::~QueryFramebuffer() {
  MAPD_CHECK_GL_ERROR(glDeleteFramebuffers(1, &_fbo));
  MAPD_CHECK_GL_ERROR(glDeleteTextures(MAX_TEXTURE_BUFFERS + 1, &_textureBuffers[0]));
  MAPD_CHECK_GL_ERROR(glDeleteRenderbuffers(MAX_RENDER_BUFFERS + 1, &_renderBuffers[0]));
  // std::cerr << "IN QueryFramebuffer DESTRUCTOR" << std::endl;
}

void QueryFramebuffer::_init(bool doHitTest, bool doDepthTest) {
  GLint currFramebuffer, currTexture, currRbo;
  GLuint tex, rbo;

  _doHitTest = doHitTest;
  _doDepthTest = doDepthTest;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currFramebuffer));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D, &currTexture));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_RENDERBUFFER_BINDING, &currRbo));

  MAPD_CHECK_GL_ERROR(glGenFramebuffers(1, &_fbo));
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, _fbo));

  // TODO: Make a texture wrapper
  tex = generateTexture2D(_width, _height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
  _attachmentManager.addTexture2dAttachment(GL_COLOR_ATTACHMENT0, tex);
  _textureBuffers[COLOR_BUFFER] = tex;

  if (_doHitTest) {
    tex = generateTexture2D(_width, _height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    _attachmentManager.addTexture2dAttachment(GL_COLOR_ATTACHMENT1, tex);
    _textureBuffers[ID_BUFFER] = tex;
  }

  if (_doDepthTest) {
    rbo = generateRenderBuffer(_width, _height, GL_DEPTH_COMPONENT);
    _attachmentManager.addRenderbufferAttachment(GL_DEPTH_ATTACHMENT, rbo);
    _renderBuffers[DEPTH_BUFFER] = rbo;
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

  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, currFramebuffer));
  MAPD_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, currTexture));
  MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, currRbo));
}

void QueryFramebuffer::resize(int width, int height) {
  if (width != _width || height != _height) {
    resizeTexture2D(_textureBuffers[COLOR_BUFFER], width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);

    if (_doHitTest) {
      resizeTexture2D(_textureBuffers[ID_BUFFER], width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    }

    if (_doDepthTest) {
      resizeRenderbuffer(_renderBuffers[DEPTH_BUFFER], width, height, GL_DEPTH_COMPONENT);
    }

    _width = width;
    _height = height;
  }
}

void QueryFramebuffer::bindToRenderer(BindType bindType) {
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(static_cast<int>(bindType), _fbo));
  // _attachmentManager.enableAttachments();

  // TODO: How do we properly deal with the state machine
  if (_renderBuffers[DEPTH_BUFFER]) {
    MAPD_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
  }
}
