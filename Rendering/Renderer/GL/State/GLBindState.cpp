#include "GLBindState.h"
#include "../MapDGL.h"
#include "../Resources/GLTexture2d.h"
#include "../Resources/GLVertexBuffer.h"
#include "../Resources/GLResource.h"
#include "../Resources/GLFramebuffer.h"
#include "../Resources/GLShader.h"
#include "../Resources/GLVertexArray.h"
#include "../Resources/GLRenderbuffer.h"
#include "../Resources/GLPixelBuffer2d.h"

namespace Rendering {
namespace GL {
namespace State {

using Resources::BufferAccessType;
using Resources::BufferAccessFreq;

GLBindState::GLBindState(GLRenderer* prntRenderer) : _prntRenderer(prntRenderer) {
}

GLBindState::~GLBindState() {
}

void GLBindState::bindResource(const Resources::GLResourceShPtr& rsrc) {
  Resources::GLResourceType type = rsrc->getResourceType();
  switch (type) {
    case Resources::GLResourceType::FRAMEBUFFER:
      bindFramebuffer(Resources::FboBind::READ_AND_DRAW, std::static_pointer_cast<Resources::GLFramebuffer>(rsrc));
      break;
    case Resources::GLResourceType::SHADER:
      bindShader(std::static_pointer_cast<Resources::GLShader>(rsrc));
      break;
    default:
      THROW_RUNTIME_EX("Cannot bind unsupported resource of type " + Rendering::GL::Resources::to_string(type) +
                       " to renderer.");
      break;
  }
}

void GLBindState::bindTexture2d(const Resources::GLTexture2dShPtr& texRsrc) {
  // TODO(croot): Currently binding a texture2d will handle both multi-sampled
  // and non-multi-sampled textures. But do we want to separate the two? Is
  // there ever a case where we'd want to bind a non multi-sampled and a mult-
  // sampled texture at the same time?

  GLuint texId = 0;
  GLuint multiSampleTexId = 0;

  if (texRsrc) {
    switch (texRsrc->getTarget()) {
      case GL_TEXTURE_2D_MULTISAMPLE:
        multiSampleTexId = texRsrc->getId();
        break;
      case GL_TEXTURE_2D:
        texId = texRsrc->getId();
        break;
      default:
        CHECK(false);
    }
  }

  bool bind = (boundTex2d.owner_before(texRsrc) || texRsrc.owner_before(boundTex2d));

  if (bind) {
    if (texRsrc) {
      texRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, multiSampleTexId));
    MAPD_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, texId));

    boundTex2d = texRsrc;
  }
}

void GLBindState::bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc) {
  GLuint vboId = (vboRsrc ? vboRsrc->getId() : 0);

  bool bind = (boundVbo.owner_before(vboRsrc) || vboRsrc.owner_before(boundVbo));

  if (bind) {
    if (vboRsrc) {
      vboRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vboId));

    boundVbo = vboRsrc;
  }
}

void GLBindState::bindFramebuffer(Resources::FboBind bindType, const Resources::GLFramebufferShPtr& fbRsrc) {
  GLuint fboId = (fbRsrc ? fbRsrc->getId() : 0);

  bool bindRead = false;
  bool bindDraw = false;

  switch (bindType) {
    case Resources::FboBind::READ:
      bindRead = (boundReadFbo.owner_before(fbRsrc) || fbRsrc.owner_before(boundReadFbo));
      break;
    case Resources::FboBind::DRAW:
      bindDraw = (boundDrawFbo.owner_before(fbRsrc) || fbRsrc.owner_before(boundDrawFbo));
      break;
    case Resources::FboBind::READ_AND_DRAW:
      bindRead = (boundReadFbo.owner_before(fbRsrc) || fbRsrc.owner_before(boundReadFbo));
      bindDraw = (boundDrawFbo.owner_before(fbRsrc) || fbRsrc.owner_before(boundDrawFbo));
      break;
  }

  if (bindRead || bindDraw) {
    if (fbRsrc) {
      fbRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindFramebuffer(static_cast<int>(bindType), fboId));

    if (bindRead) {
      boundReadFbo = fbRsrc;
    }

    if (bindDraw) {
      boundDrawFbo = fbRsrc;
    }
  }
}

void GLBindState::bindShader(const Resources::GLShaderShPtr& shaderRsrc) {
  GLuint programId = (shaderRsrc ? shaderRsrc->getId() : 0);

  bool bind = (boundShader.owner_before(shaderRsrc) || shaderRsrc.owner_before(boundShader));

  if (bind) {
    if (shaderRsrc) {
      shaderRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glUseProgram(programId));

    boundShader = shaderRsrc;
  }
}

void GLBindState::bindVertexArray(const Resources::GLVertexArrayShPtr& vaoRsrc) {
  GLuint vao = 0;

  if (vaoRsrc) {
    vao = vaoRsrc->getId();

    // verify that the attached vbos are still valid.
    vaoRsrc->validateVBOs();
  }

  bool bind = (boundVao.owner_before(vaoRsrc) || vaoRsrc.owner_before(boundVao));

  if (bind) {
    if (vaoRsrc) {
      vaoRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindVertexArray(vao));

    boundVao = vaoRsrc;
  }
}

void GLBindState::bindRenderbuffer(const Resources::GLRenderbufferShPtr& rboRsrc) {
  GLuint rbo = (rboRsrc ? rboRsrc->getId() : 0);

  bool bind = (boundRbo.owner_before(rboRsrc) || rboRsrc.owner_before(boundRbo));

  if (bind) {
    if (rboRsrc) {
      rboRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, rbo));

    boundRbo = rboRsrc;
  }
}

void GLBindState::bindReadPixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc) {
  GLuint pbo = (pboRsrc ? pboRsrc->getId() : 0);

  bool bind = (boundReadPbo.owner_before(pboRsrc) || pboRsrc.owner_before(boundReadPbo));

  if (bind) {
    if (pboRsrc) {
      pboRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);

      BufferAccessType accessType = pboRsrc->getAccessType();
      RUNTIME_EX_ASSERT(accessType == BufferAccessType::READ || accessType == BufferAccessType::READ_AND_WRITE ||
                            accessType == BufferAccessType::COPY,
                        "Cannot bind pbo resource for reading. It was not initialized with READ/COPY access type.");
    }

    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo));

    boundReadPbo = pboRsrc;
  }
}

void GLBindState::bindWritePixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc) {
  GLuint pbo = (pboRsrc ? pboRsrc->getId() : 0);

  bool bind = (boundWritePbo.owner_before(pboRsrc) || pboRsrc.owner_before(boundWritePbo));

  if (bind) {
    if (pboRsrc) {
      pboRsrc->validateRenderer(__FILE__, __LINE__, _prntRenderer);

      BufferAccessType accessType = pboRsrc->getAccessType();
      RUNTIME_EX_ASSERT(accessType == BufferAccessType::WRITE || accessType == BufferAccessType::READ_AND_WRITE ||
                            accessType == BufferAccessType::COPY,
                        "Cannot bind pbo resource for reading. It was not initialized with WRITE/COPY access type.");
    }

    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    boundWritePbo = pboRsrc;
  }
}

Resources::GLTexture2dShPtr GLBindState::getBoundTexture2d() const {
  return boundTex2d.lock();
}

bool GLBindState::hasBoundTexture2d() const {
  return boundTex2d.lock() != nullptr;
}

Resources::GLVertexBufferShPtr GLBindState::getBoundVertexBuffer() const {
  return boundVbo.lock();
}

bool GLBindState::hasBoundVertexBuffer() const {
  return boundVbo.lock() != nullptr;
}

Resources::GLFramebufferShPtr GLBindState::getBoundFramebuffer(const Resources::FboBind bindType) const {
  switch (bindType) {
    case Resources::FboBind::READ:
      return boundReadFbo.lock();
    case Resources::FboBind::DRAW:
      return boundDrawFbo.lock();
    case Resources::FboBind::READ_AND_DRAW:
      if (!boundReadFbo.owner_before(boundDrawFbo) && !boundDrawFbo.owner_before(boundReadFbo)) {
        return boundReadFbo.lock();
      }
      THROW_RUNTIME_EX(
          "The bound read and draw framebuffers are different. Please specify which framebuffer (READ or DRAW) you "
          "wish to accees.");
  }
  return nullptr;
}

bool GLBindState::hasBoundFramebuffer(const Resources::FboBind bindType) const {
  switch (bindType) {
    case Resources::FboBind::READ:
      return boundReadFbo.lock() != nullptr;
    case Resources::FboBind::DRAW:
      return boundDrawFbo.lock() != nullptr;
    case Resources::FboBind::READ_AND_DRAW:
      return (boundReadFbo.lock() != nullptr && boundDrawFbo.lock() != nullptr);
  }

  return false;
}

Resources::GLShaderShPtr GLBindState::getBoundShader() const {
  return boundShader.lock();
}

bool GLBindState::hasBoundShader() const {
  return boundShader.lock() != nullptr;
}

Resources::GLVertexArrayShPtr GLBindState::getBoundVertexArray() const {
  return boundVao.lock();
}

bool GLBindState::hasBoundVertexArray() const {
  return boundVao.lock() != nullptr;
}

Resources::GLRenderbufferShPtr GLBindState::getBoundRenderbuffer() const {
  return boundRbo.lock();
}

bool GLBindState::hasBoundRenderbuffer() const {
  return boundRbo.lock() != nullptr;
}

Resources::GLPixelBuffer2dShPtr GLBindState::getBoundReadPixelBuffer() const {
  return boundReadPbo.lock();
}

bool GLBindState::hasBoundReadPixelBuffer() const {
  return !boundReadPbo.expired();
}

Resources::GLPixelBuffer2dShPtr GLBindState::getBoundWritePixelBuffer() const {
  return boundWritePbo.lock();
}

bool GLBindState::hasBoundWritePixelBuffer() const {
  return !boundWritePbo.expired();
}

}  // namespace State
}  // namespace GL
}  // namespace Rendering
