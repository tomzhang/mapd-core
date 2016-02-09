#include "GLBindState.h"
#include "../MapDGL.h"
#include "../Resources/GLVertexBuffer.h"
#include "../Resources/GLResource.h"
#include "../Resources/GLFramebuffer.h"
#include "../Resources/GLShader.h"
#include "../Resources/GLVertexArray.h"

namespace Rendering {
namespace GL {
namespace State {

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

void GLBindState::bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc) {
  GLuint vboId = (vboRsrc ? vboRsrc->getId() : 0);

  bool bind = (boundVbo.owner_before(vboRsrc) || vboRsrc.owner_before(boundVbo));

  if (bind) {
    if (vboRsrc) {
      vboRsrc->validateRenderer(_prntRenderer);
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
      fbRsrc->validateRenderer(_prntRenderer);
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
      shaderRsrc->validateRenderer(_prntRenderer);
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
      vaoRsrc->validateRenderer(_prntRenderer);
    }

    MAPD_CHECK_GL_ERROR(glBindVertexArray(vao));

    boundVao = vaoRsrc;
  }
}

Resources::GLVertexBufferShPtr GLBindState::getBoundVbo() const {
  return boundVbo.lock();
}

bool GLBindState::hasBoundVbo() const {
  return boundVbo.lock() != nullptr;
}

Resources::GLFramebufferShPtr GLBindState::getBoundFbo(const Resources::FboBind bindType) const {
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

bool GLBindState::hasBoundFbo(const Resources::FboBind bindType) const {
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

}  // namespace State
}  // namespace GL
}  // namespace Rendering
