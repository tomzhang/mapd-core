#include "GLRenderbuffer.h"
#include "../MapDGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

static void resizeRenderbuffer(GLuint rbo,
                               GLint width,
                               GLint height,
                               GLint internalFormat,
                               int numSamples = 1,
                               bool keepBound = false) {
  int maxSize = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxSize));

  RUNTIME_EX_ASSERT(width <= maxSize,
                    "Cannot resize renderbuffer with width " + std::to_string(width) +
                        ". It exceeds the max width of " + std::to_string(maxSize));
  RUNTIME_EX_ASSERT(height <= maxSize,
                    "Cannot resize renderbuffer with height " + std::to_string(height) +
                        ". It exceeds the max height of " + std::to_string(maxSize));

  int maxSamples = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_SAMPLES, &maxSamples));
  RUNTIME_EX_ASSERT(numSamples <= maxSamples,
                    "Cannot create renderbuffer with " + std::to_string(numSamples) +
                        ". The render context only supports " + std::to_string(maxSamples) + " samples.");

  GLint currRbo = rbo;
  if (!keepBound) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_RENDERBUFFER_BINDING, &currRbo));
  }

  MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, rbo));

  if (numSamples > 1) {
    MAPD_CHECK_GL_ERROR(glRenderbufferStorageMultisample(GL_RENDERBUFFER, numSamples, internalFormat, width, height));
  } else {
    MAPD_CHECK_GL_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height));
  }
  MAPD_CHECK_GL_MEMORY_ERROR();

  if (currRbo != static_cast<GLint>(rbo)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, currRbo));
  }
}

GLRenderbuffer::GLRenderbuffer(const RendererWkPtr& rendererPtr,
                               size_t width,
                               size_t height,
                               GLenum internalFormat,
                               size_t numSamples)
    : GLResource(rendererPtr, GLResourceType::RENDERBUFFER),
      _width(width),
      _height(height),
      _internalFormat(internalFormat),
      _numSamples(numSamples),
      _bufferId(0) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0, "Invalid dimensions for the renderbuffer. Dimensions must be > 0");
  _initResource();
}

GLRenderbuffer::~GLRenderbuffer() {
  cleanupResource();
}

void GLRenderbuffer::_initResource() {
  validateRenderer(__FILE__, __LINE__);

  MAPD_CHECK_GL_ERROR(glGenRenderbuffers(1, &_bufferId));

  resizeRenderbuffer(_bufferId, _width, _height, _internalFormat, _numSamples);

  setUsable();
}

void GLRenderbuffer::_cleanupResource() {
  if (_bufferId) {
    MAPD_CHECK_GL_ERROR(glDeleteRenderbuffers(1, &_bufferId));
  }
  _makeEmpty();
}

void GLRenderbuffer::_makeEmpty() {
  _width = _height = 0;
  _bufferId = 0;
}

void GLRenderbuffer::resize(size_t width, size_t height) {
  if (width != _width || height != _height) {
    RUNTIME_EX_ASSERT(width > 0 && height > 0,
                      "Invalid dimensions " + std::to_string(width) + "x" + std::to_string(height) +
                          " for the renderbuffer. Dimensions must be > 0");

    validateUsability(__FILE__, __LINE__);

    resizeRenderbuffer(_bufferId, width, height, _internalFormat, _numSamples);

    _width = width;
    _height = height;
  }
}

void GLRenderbuffer::bindToRenderer(GLRenderer* renderer) {
  // TODO(croot): perhaps write a "don't check renderer" version?
  validateUsability(__FILE__, __LINE__);

  MAPD_CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, _bufferId));
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
