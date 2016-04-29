#include "GLTexture2d.h"
#include "../MapDGL.h"

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
                            GLTexture2dSampleProps* sampleProps = nullptr,
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
                    "Cannot resize texture with " + std::to_string(numSamples) +
                        " samples. The render context only supports " + std::to_string(maxSamples) + " samples.");

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

  if (sampleProps && numSamples == 1) {
    sampleProps->apply(target);
  }

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

GLTexture2dSampleProps::GLTexture2dSampleProps()
    : minFilter(GL_NEAREST), magFilter(GL_LINEAR), wrapS(GL_CLAMP_TO_EDGE), wrapT(GL_CLAMP_TO_EDGE) {
}

GLTexture2dSampleProps::GLTexture2dSampleProps(GLint minFilter, GLint magFilter, GLint wrapS, GLint wrapT)
    : minFilter(minFilter), magFilter(magFilter), wrapS(wrapS), wrapT(wrapT) {
}

GLTexture2dSampleProps::GLTexture2dSampleProps(const GLTexture2dSampleProps& props)
    : minFilter(props.minFilter), magFilter(props.magFilter), wrapS(props.wrapS), wrapT(props.wrapT) {
}

GLTexture2dSampleProps::GLTexture2dSampleProps(GLTexture2dSampleProps&& props)
    : minFilter(std::move(props.minFilter)),
      magFilter(std::move(props.magFilter)),
      wrapS(std::move(props.wrapS)),
      wrapT(std::move(props.wrapT)) {
}

bool GLTexture2dSampleProps::operator==(const GLTexture2dSampleProps& rhs) const {
  return (minFilter == rhs.minFilter && magFilter == rhs.magFilter && wrapS == rhs.wrapS && wrapT == rhs.wrapT);
}

bool GLTexture2dSampleProps::operator!=(const GLTexture2dSampleProps& rhs) const {
  return !(operator==(rhs));
}

GLTexture2dSampleProps& GLTexture2dSampleProps::operator=(const GLTexture2dSampleProps& rhs) {
  minFilter = rhs.minFilter;
  magFilter = rhs.magFilter;
  wrapS = rhs.wrapS;
  wrapT = rhs.wrapT;

  return *this;
}

void GLTexture2dSampleProps::apply(GLenum target) {
  MAPD_CHECK_GL_ERROR(glTexParameterf(target, GL_TEXTURE_MIN_FILTER, minFilter));
  MAPD_CHECK_GL_ERROR(glTexParameterf(target, GL_TEXTURE_MAG_FILTER, magFilter));
  MAPD_CHECK_GL_ERROR(glTexParameterf(target, GL_TEXTURE_WRAP_S, wrapS));
  MAPD_CHECK_GL_ERROR(glTexParameterf(target, GL_TEXTURE_WRAP_T, wrapT));
}

GLTexture2d::GLTexture2d(const RendererWkPtr& rendererPtr,
                         size_t width,
                         size_t height,
                         GLenum internalFormat,
                         GLenum pixelFormat,
                         GLenum pixelType,
                         size_t numSamples,
                         const GLTexture2dSampleProps& sampleProps)
    : GLResource(rendererPtr, GLResourceType::TEXTURE_2D),
      _width(width),
      _height(height),
      _internalFormat(internalFormat),
      _pixelFormat(pixelFormat),
      _pixelType(pixelType),
      _sampleProps(sampleProps),
      _numSamples(numSamples),
      _textureId(0) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0, "Invalid dimensions for the texture. Dimensions must be > 0");
  _initResource();
}

GLTexture2d::~GLTexture2d() {
  cleanupResource();
}

void GLTexture2d::_initResource() {
  validateRenderer(__FILE__, __LINE__);

  RUNTIME_EX_ASSERT(_numSamples > 0, "Invalid number of samples " + std::to_string(_numSamples));

  if (_numSamples > 1) {
    _target = GL_TEXTURE_2D_MULTISAMPLE;
  } else {
    _target = GL_TEXTURE_2D;
  }

  MAPD_CHECK_GL_ERROR(glGenTextures(1, &_textureId));

  resizeTexture2d(
      _target, _textureId, _width, _height, _internalFormat, _pixelFormat, _pixelType, _numSamples, &_sampleProps);

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

void GLTexture2d::resize(size_t width, size_t height) {
  if (width != _width || height != _height) {
    RUNTIME_EX_ASSERT(width > 0 && height > 0,
                      "Invalid dimensions " + std::to_string(width) + "x" + std::to_string(height) +
                          " for the texture. Dimensions must be > 0");

    validateUsability(__FILE__, __LINE__);

    _width = width;
    _height = height;

    // TODO(croot): do we need to supply the sample props down on a resize or will
    // the stay intact?
    resizeTexture2d(
        _target, _textureId, _width, _height, _internalFormat, _pixelFormat, _pixelType, _numSamples, &_sampleProps);
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
