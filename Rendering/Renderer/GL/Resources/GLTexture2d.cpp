#include "GLTexture2d.h"
#include "../MapDGL.h"
#include "GLPixelBuffer2d.h"

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
                            const GLvoid* pixelData = nullptr,
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

  RUNTIME_EX_ASSERT(!pixelData || numSamples == 1,
                    "Initializing a multisampled 2D texture with pixel data is unsupported.");

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
    MAPD_CHECK_GL_ERROR(glTexImage2D(target, 0, internalFormat, width, height, 0, pixelFormat, pixelType, pixelData));
  }

  if (currTex != static_cast<GLint>(tex)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(target, currTex));
  }
}

GLTexture2dSampleProps::GLTexture2dSampleProps()
    : minFilter(GL_NEAREST), magFilter(GL_LINEAR), wrapS(GL_CLAMP_TO_EDGE), wrapT(GL_CLAMP_TO_EDGE) {}

GLTexture2dSampleProps::GLTexture2dSampleProps(GLint minFilter, GLint magFilter, GLint wrapS, GLint wrapT)
    : minFilter(minFilter), magFilter(magFilter), wrapS(wrapS), wrapT(wrapT) {}

GLTexture2dSampleProps::GLTexture2dSampleProps(const GLTexture2dSampleProps& props)
    : minFilter(props.minFilter), magFilter(props.magFilter), wrapS(props.wrapS), wrapT(props.wrapT) {}

GLTexture2dSampleProps::GLTexture2dSampleProps(GLTexture2dSampleProps&& props)
    : minFilter(std::move(props.minFilter)),
      magFilter(std::move(props.magFilter)),
      wrapS(std::move(props.wrapS)),
      wrapT(std::move(props.wrapT)) {}

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
                         const GLTexture2dSampleProps& sampleProps,
                         const GLvoid* pixelData)
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
  _initResource(pixelData);
}

GLTexture2d::~GLTexture2d() {
  cleanupResource();
}

void GLTexture2d::_initResource(const GLvoid* pixelData) {
  validateRenderer(__FILE__, __LINE__);

  RUNTIME_EX_ASSERT(_numSamples > 0, "Invalid number of samples " + std::to_string(_numSamples));

  if (_numSamples > 1) {
    _target = GL_TEXTURE_2D_MULTISAMPLE;
  } else {
    _target = GL_TEXTURE_2D;
  }

  MAPD_CHECK_GL_ERROR(glGenTextures(1, &_textureId));

  resizeTexture2d(_target,
                  _textureId,
                  _width,
                  _height,
                  _internalFormat,
                  _pixelFormat,
                  _pixelType,
                  _numSamples,
                  &_sampleProps,
                  pixelData);

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

void GLTexture2d::copyPixelsFromPixelBuffer(const GLPixelBuffer2dShPtr& pbo,
                                            int startx,
                                            int starty,
                                            int width,
                                            int height) {
  RUNTIME_EX_ASSERT(_numSamples == 1, "Cannot copy pixels from a pixel buffer to a multi-sampled 2d texture");

  RUNTIME_EX_ASSERT(pbo->getPixelFormat() == _pixelFormat && pbo->getPixelType() == _pixelType,
                    "Cannot copy contents of the pixel buffer to the texture. The pixel formats/types do not match - "
                    "pixel buffer: format: " +
                        std::to_string(pbo->getPixelFormat()) + ", type: " + std::to_string(pbo->getPixelType()) +
                        ", texture: format: " + std::to_string(_pixelFormat) + ", type: " + std::to_string(_pixelType));

  size_t widthToUse = (width < 0 ? pbo->getWidth() : static_cast<size_t>(width));
  size_t heightToUse = (height < 0 ? pbo->getHeight() : static_cast<size_t>(height));
  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  RUNTIME_EX_ASSERT(static_cast<size_t>(startx) < myWidth && startx + widthToUse <= myWidth &&
                        static_cast<size_t>(starty) < myHeight && starty + heightToUse <= myHeight,
                    "The copy area: " + std::to_string(widthToUse) + "x" + std::to_string(heightToUse) +
                        " starting at [" + std::to_string(startx) + ", " + std::to_string(starty) +
                        "] extends beyond the bounds of the GLTexture2D: " + std::to_string(myWidth) + "x" +
                        std::to_string(myHeight) + ".");

  // NOTE: Renderer needs to be properly activated before calling this function
  GLRenderer* glRenderer = getGLRenderer();

  glRenderer->bindReadPixelBuffer(pbo);

  CHECK(_target == GL_TEXTURE_2D);
  GLint currTex;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D, &currTex));
  MAPD_CHECK_GL_ERROR(glBindTexture(_target, _textureId));

  // TODO(croot): support mipmaps
  // TODO(croot): support byte offset into pbo data?
  MAPD_CHECK_GL_ERROR(
      glTexSubImage2D(_target, 0, startx, starty, widthToUse, heightToUse, _pixelFormat, _pixelType, nullptr));

  if (currTex != static_cast<GLint>(_textureId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(_target, currTex));
  }

  glRenderer->bindReadPixelBuffer(nullptr);
}

void GLTexture2d::uploadPixelsToTexture(const size_t width,
                                        const size_t height,
                                        const GLenum pixelFormat,
                                        const GLenum pixelType,
                                        const GLvoid* pixelData,
                                        const int arrayIdx,
                                        const size_t startx,
                                        const size_t starty) {
  // TODO(croot): check that the pixel format/type of the pbo is compatible with
  // the internal format of the GLTexture2dArray
  CHECK(pixelFormat == _pixelFormat && pixelType == _pixelType);

  RUNTIME_EX_ASSERT(_numSamples == 1, "Cannot copy pixels from to a multi-sampled 2d texture");

  auto myWidth = getWidth();
  auto myHeight = getHeight();
  RUNTIME_EX_ASSERT(startx < myWidth && startx + width <= myWidth && starty < myHeight && starty + height <= myHeight,
                    "The area of pixels to set: " + std::to_string(width) + "x" + std::to_string(height) +
                        " starting at [" + std::to_string(startx) + ", " + std::to_string(starty) +
                        "] extends beyond the bounds of the textures in the GLTexture2d: " + std::to_string(myWidth) +
                        "x" + std::to_string(myHeight) + ".");

  // TODO(croot): do something to verify that the pixel format uploaded matches that of the internal texture?

  // TODO(croot): support mipmaps
  // TODO(croot): support byte offset into pbo data?

  CHECK(_target == GL_TEXTURE_2D);
  GLint currTex;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D, &currTex));
  MAPD_CHECK_GL_ERROR(glBindTexture(_target, _textureId));
  MAPD_CHECK_GL_ERROR(glTexSubImage2D(_target, 0, startx, starty, width, height, _pixelFormat, _pixelType, pixelData));

  if (currTex != static_cast<GLint>(_textureId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(_target, currTex));
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
