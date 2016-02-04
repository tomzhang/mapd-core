#include "GLTexture2d.h"
#include "../MapDGL.h"
#include <iostream>

namespace Rendering {
namespace GL {
namespace Resources {

static void resizeTexture2d(GLenum& target,
                            GLuint tex,
                            GLint width,
                            GLint height,
                            GLenum internalFormat,
                            GLenum pixelFormat,
                            GLenum pixelType,
                            int numSamples = 1,
                            bool keepBound = false) {
  int maxSize = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize));

  RUNTIME_EX_ASSERT(width <= maxSize,
                    "Cannot resize texture with width " + std::to_string(width) + ". It exceeds the max width of " +
                        std::to_string(maxSize));
  RUNTIME_EX_ASSERT(height <= maxSize,
                    "Cannot resize texture with height " + std::to_string(height) + ". It exceeds the max height of " +
                        std::to_string(maxSize));

  int maxSamples = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_SAMPLES, &maxSamples));
  RUNTIME_EX_ASSERT(numSamples <= maxSamples,
                    "Cannot resize texture with " + std::to_string(numSamples) + ". The render context only supports " +
                        std::to_string(maxSamples) + " samples.");

  GLenum bindingTarget = GL_TEXTURE_BINDING_2D;

  CHECK(numSamples > 0);
  if (numSamples > 1) {
    if (target != GL_TEXTURE_2D_MULTISAMPLE) {
      LOG(WARNING) << "resizeTexture2d(): target " << target
                   << " is the wrong type for a 2D texture with multisampling. Changing the target to "
                   << GL_TEXTURE_2D_MULTISAMPLE << ".";
      target = GL_TEXTURE_2D_MULTISAMPLE;
    }

    bindingTarget = GL_TEXTURE_BINDING_2D_MULTISAMPLE;
  } else if (target != GL_TEXTURE_2D) {
    LOG(WARNING) << "resizeTexture2d(): target " << target
                 << " is the wrong type for a 2D texture without multisampling. Changing the target to "
                 << GL_TEXTURE_2D << ".";
    target = GL_TEXTURE_2D;
  }

  GLint currTex = tex;
  if (!keepBound) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(bindingTarget, &currTex));
  }

  MAPD_CHECK_GL_ERROR(glBindTexture(target, tex));

  if (numSamples > 1) {
    MAPD_CHECK_GL_ERROR(glTexImage2DMultisample(target, numSamples, internalFormat, width, height, GL_TRUE));
  } else {
    MAPD_CHECK_GL_ERROR(glTexImage2D(target, 0, internalFormat, width, height, 0, pixelFormat, pixelType, NULL));
  }

  if (currTex != static_cast<GLint>(tex)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(target, currTex));
  }
}

GLTexture2d::GLTexture2d(const RendererWkPtr& rendererPtr,
                         int width,
                         int height,
                         GLenum internalFormat,
                         GLenum pixelFormat,
                         GLenum pixelType,
                         int numSamples)
    : GLResource(rendererPtr),
      _width(width),
      _height(height),
      _internalFormat(internalFormat),
      _pixelFormat(pixelFormat),
      _pixelType(pixelType),
      _numSamples(numSamples),
      _textureId(0) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0, "Invalid dimensions for the texture. Dimensions must be > 0");
  _initResource();
}

GLTexture2d::~GLTexture2d() {
  std::cerr << "CROOT - GLTexture2d destructor" << std::endl;
  cleanupResource();
}

void GLTexture2d::_initResource() {
  validateRenderer();

  RUNTIME_EX_ASSERT(_numSamples > 0, "Invalid number of samples " + std::to_string(_numSamples));

  if (_numSamples > 1) {
    _target = GL_TEXTURE_2D_MULTISAMPLE;
  } else {
    _target = GL_TEXTURE_2D;
  }

  MAPD_CHECK_GL_ERROR(glGenTextures(1, &_textureId));

  resizeTexture2d(_target, _textureId, _width, _height, _internalFormat, _pixelFormat, _pixelType, _numSamples);

  setUsable();
}

void GLTexture2d::_cleanupResource() {
  if (_textureId) {
    MAPD_CHECK_GL_ERROR(glDeleteTextures(1, &_textureId));
    _textureId = 0;
  }

  _makeEmpty();
}

void GLTexture2d::_makeEmpty() {
  _width = _height = 0;
  _textureId = 0;
}

void GLTexture2d::resize(int width, int height) {
  if (width != _width || height != _height) {
    RUNTIME_EX_ASSERT(width > 0 && height > 0, "Invalid dimensions for the texture. Dimensions must be > 0");

    validateUsability();

    resizeTexture2d(_target, _textureId, _width, _height, _internalFormat, _pixelFormat, _pixelType, _numSamples);

    _width = width;
    _height = height;
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
