#include "GLTexture2dArray.h"
#include "../MapDGL.h"
#include "GLPixelBuffer2d.h"

namespace Rendering {
namespace GL {
namespace Resources {

static GLuint createTexture2dArray(GLenum& target,
                                   GLint width,
                                   GLint height,
                                   GLint depth,
                                   GLenum internalFormat,
                                   int numSamples = 1,
                                   GLTexture2dSampleProps* sampleProps = nullptr,
                                   bool keepBound = false) {
  int maxSize = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize));

  RUNTIME_EX_ASSERT(width <= maxSize,
                    "Cannot create texture array with width " + std::to_string(width) +
                        ". It exceeds the max width of " + std::to_string(maxSize));
  RUNTIME_EX_ASSERT(height <= maxSize,
                    "Cannot create texture array with height " + std::to_string(height) +
                        ". It exceeds the max height of " + std::to_string(maxSize));
  RUNTIME_EX_ASSERT(depth <= maxSize,
                    "Cannot create texture array with depth " + std::to_string(depth) +
                        ". It exceeds the max depth of " + std::to_string(maxSize));

  int maxSamples = -1;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_SAMPLES, &maxSamples));
  RUNTIME_EX_ASSERT(numSamples <= maxSamples,
                    "Cannot create texture array with " + std::to_string(numSamples) +
                        " samples. The render context only supports " + std::to_string(maxSamples) + " samples.");

  GLenum bindingTarget = GL_TEXTURE_BINDING_2D_ARRAY;

  CHECK(numSamples > 0);
  if (numSamples > 1) {
    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
      LOG(WARNING) << "createTexture2dArray(): target " << target
                   << " is the wrong type for a 2D texture array with multisampling. Changing the target to "
                   << GL_TEXTURE_2D_MULTISAMPLE_ARRAY << ".";
      target = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    }

    bindingTarget = GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
  } else if (target != GL_TEXTURE_2D_ARRAY) {
    LOG(WARNING) << "createTexture2dArray(): target " << target
                 << " is the wrong type for a 2D texture without multisampling. Changing the target to "
                 << GL_TEXTURE_2D_ARRAY << ".";
    target = GL_TEXTURE_2D_ARRAY;
  }

  GLuint textureArrayId;
  MAPD_CHECK_GL_ERROR(glGenTextures(1, &textureArrayId));

  GLint currTex = textureArrayId;
  if (!keepBound) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(bindingTarget, &currTex));
  }

  MAPD_CHECK_GL_ERROR(glBindTexture(target, textureArrayId));

  if (sampleProps) {
    sampleProps->apply(target);
  }

  if (numSamples > 1) {
    MAPD_CHECK_GL_ERROR(glTexStorage3DMultisample(target, numSamples, internalFormat, width, height, depth, GL_TRUE));
  } else {
    MAPD_CHECK_GL_ERROR(glTexStorage3D(target, 1, internalFormat, width, height, depth));
  }

  if (currTex != static_cast<GLint>(textureArrayId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(target, currTex));
  }

  return textureArrayId;
}

GLTexture2dArray::GLTexture2dArray(const RendererWkPtr& rendererPtr,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   GLenum internalFormat,
                                   size_t numSamples,
                                   const GLTexture2dSampleProps& sampleProps)
    : GLResource(rendererPtr, GLResourceType::TEXTURE_2D_ARRAY),
      _width(width),
      _height(height),
      _depth(depth),
      _internalFormat(internalFormat),
      _sampleProps(sampleProps),
      _numSamples(numSamples),
      _textureArrayId(0),
      _textureReferences() {
  RUNTIME_EX_ASSERT(width > 0 && height > 0 && depth > 0,
                    "Invalid dimensions for the texture array. Dimensions must be > 0");
  _initResource();
}

GLTexture2dArray::GLTexture2dArray(const RendererWkPtr& rendererPtr, const std::vector<GLTexture2dShPtr>& initTextures)
    : GLResource(rendererPtr, GLResourceType::TEXTURE_2D_ARRAY),
      _width(0),
      _height(0),
      _depth(0),
      _internalFormat(GL_RGBA8),
      _sampleProps(),
      _numSamples(0),
      _textureArrayId(0),
      _textureReferences() {
  RUNTIME_EX_ASSERT(initTextures.size() > 0,
                    "Input array of textures to initialize the GLTexture2dArray with must have at least 1 texture.");

  _width = initTextures[0]->getWidth();
  _height = initTextures[0]->getHeight();
  _depth = initTextures.size();
  _internalFormat = initTextures[0]->getInternalFormat();
  _sampleProps = initTextures[0]->getSamplingProps();
  _numSamples = initTextures[0]->getNumSamples();

  _initResource();

  _textureReferences = initTextures;

  updateFromTextures();
}

GLTexture2dArray::~GLTexture2dArray() {
  cleanupResource();
}

void GLTexture2dArray::_initResource() {
  validateRenderer(__FILE__, __LINE__);

  RUNTIME_EX_ASSERT(_numSamples > 0, "Invalid number of samples " + std::to_string(_numSamples));

  if (_numSamples > 1) {
    _target = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
  } else {
    _target = GL_TEXTURE_2D_ARRAY;
  }

  _textureArrayId = createTexture2dArray(_target, _width, _height, _depth, _internalFormat, _numSamples);

  setUsable();
}

void GLTexture2dArray::_cleanupResource() {
  if (_textureArrayId) {
    MAPD_CHECK_GL_ERROR(glDeleteTextures(1, &_textureArrayId));
    _textureArrayId = 0;
  }

  _makeEmpty();
}

void GLTexture2dArray::_validateTexture(const GLTexture2dShPtr& texture) {
  // TODO(croot): verify that the texture is either created from the same
  // renderer or a sharable renderer
  RUNTIME_EX_ASSERT(getGLRenderer() == texture->getGLRenderer() && texture->getWidth() == _width &&
                        texture->getHeight() == _height && texture->getInternalFormat() == _internalFormat &&
                        texture->getSamplingProps() == _sampleProps && texture->getNumSamples() == _numSamples,
                    "Cannot add texture to GLTexture2dArray. It does not meet the validation requirements. It must "
                    "have the same width, height, internal format, number of samples, sampling properties, and be from "
                    "the same renderer.");
}

void GLTexture2dArray::_makeEmpty() {
  _width = _height = 0, _depth = 0;
  _textureArrayId = 0;
}

void GLTexture2dArray::_rebuild(size_t width, size_t height, size_t depth) {
  if (width != _width || height != _height || depth != _depth) {
    validateUsability(__FILE__, __LINE__);

    _cleanupResource();
    _textureArrayId = createTexture2dArray(_target, width, height, depth, _internalFormat, _numSamples);
    _width = width;
    _height = height;
    _depth = depth;

    updateFromTextures();
  }
}

void GLTexture2dArray::resize(size_t depth) {
  if (depth != _depth && _textureReferences.size()) {
    // TODO(croot): create a base TextureArray class and create two derived classes with the different
    // apis to support instead of this, which isn't ideal.
    THROW_RUNTIME_EX("Cannot resize a texture array with texture references. Please use the push/pop/insert routines");
  }

  _rebuild(_width, _height, depth);
}

void GLTexture2dArray::resize(size_t width, size_t height) {
  _rebuild(width, height, _depth);
}

void GLTexture2dArray::resize(size_t width, size_t height, size_t depth) {
  if (depth != _depth && _textureReferences.size()) {
    // TODO(croot): create a base TextureArray class and create two derived classes with the different
    // apis to support instead of this, which isn't ideal.
    THROW_RUNTIME_EX("Cannot resize a texture array with texture references. Please use the push/pop/insert routines");
  }

  _rebuild(width, height, depth);
}

void GLTexture2dArray::copyPixelsFromPixelBuffer(const GLPixelBuffer2dShPtr& pbo,
                                                 int arrayIdx,
                                                 int startx,
                                                 int starty,
                                                 int width,
                                                 int height) {
  // TODO(croot): check that the pixel format/type of the pbo is compatible with
  // the internal format of the GLTexture2dArray

  RUNTIME_EX_ASSERT(arrayIdx < static_cast<int>(_depth),
                    "Invalid index " + std::to_string(arrayIdx) + ". The texture array only has " +
                        std::to_string(_depth) + " textures.");

  RUNTIME_EX_ASSERT(_numSamples == 1, "Cannot copy pixels from a pixel buffer to a multi-sampled 2d texture array");

  size_t widthToUse = (width < 0 ? pbo->getWidth() : static_cast<size_t>(width));
  size_t heightToUse = (height < 0 ? pbo->getHeight() : static_cast<size_t>(height));
  size_t myWidth = getWidth();
  size_t myHeight = getHeight();

  RUNTIME_EX_ASSERT(startx < static_cast<int>(myWidth) && startx + widthToUse <= myWidth &&
                        starty < static_cast<int>(myHeight) && starty + heightToUse <= myHeight,
                    "The copy area: " + std::to_string(widthToUse) + "x" + std::to_string(heightToUse) +
                        " starting at [" + std::to_string(startx) + ", " + std::to_string(starty) +
                        "] extends beyond the bounds of the textures in the GLTexture2dArray: " +
                        std::to_string(myWidth) + "x" + std::to_string(myHeight) + ".");

  // NOTE: Renderer needs to be properly activated before calling this function
  GLRenderer* glRenderer = getGLRenderer();

  glRenderer->bindReadPixelBuffer(pbo);

  CHECK(_target == GL_TEXTURE_2D_ARRAY);
  GLint currTex;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &currTex));
  MAPD_CHECK_GL_ERROR(glBindTexture(_target, _textureArrayId));

  // TODO(croot): support mipmaps
  // TODO(croot): support byte offset into pbo data?
  if (arrayIdx < 0) {
    for (int i = 0; i < static_cast<int>(_depth); ++i) {
      MAPD_CHECK_GL_ERROR(glTexSubImage3D(_target,
                                          0,
                                          startx,
                                          starty,
                                          i,
                                          widthToUse,
                                          heightToUse,
                                          1,
                                          pbo->getPixelFormat(),
                                          pbo->getPixelType(),
                                          nullptr));
    }
  } else {
    MAPD_CHECK_GL_ERROR(glTexSubImage3D(_target,
                                        0,
                                        startx,
                                        starty,
                                        arrayIdx,
                                        widthToUse,
                                        heightToUse,
                                        1,
                                        pbo->getPixelFormat(),
                                        pbo->getPixelType(),
                                        nullptr));
  }

  if (currTex != static_cast<GLint>(_textureArrayId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(_target, currTex));
  }

  glRenderer->bindReadPixelBuffer(nullptr);
}

void GLTexture2dArray::updateFromTextures() {
  if (!_textureReferences.size()) {
    return;
  }

  CHECK(_textureReferences.size() == _depth);

  for (const auto& texture : _textureReferences) {
    // make sure the texture references still align properly
    _validateTexture(texture);
  }

  for (size_t i = 0; i < _depth; ++i) {
    MAPD_CHECK_GL_ERROR(glCopyImageSubData(_textureReferences[i]->getId(),
                                           _textureReferences[i]->getTarget(),
                                           0,
                                           0,
                                           0,
                                           0,
                                           _textureArrayId,
                                           _target,
                                           0,
                                           0,
                                           0,
                                           i,
                                           _width,
                                           _height,
                                           1));
  }
}

void GLTexture2dArray::uploadPixelsToBufferIndex(const size_t width,
                                                 const size_t height,
                                                 const GLenum pixelFormat,
                                                 const GLenum pixelType,
                                                 const GLvoid* pixelData,
                                                 const int arrayIdx,
                                                 const size_t startx,
                                                 const size_t starty) {
  // TODO(croot): check that the pixel format/type of the pbo is compatible with
  // the internal format of the GLTexture2dArray

  RUNTIME_EX_ASSERT(arrayIdx < static_cast<int>(_depth),
                    "Invalid index " + std::to_string(arrayIdx) + ". The texture array only has " +
                        std::to_string(_depth) + " textures.");
  RUNTIME_EX_ASSERT(_textureReferences.size() == 0,
                    "Cannot copy pixels to a 2d texture array that is referencing existing 2d textures.");
  RUNTIME_EX_ASSERT(_numSamples == 1, "Cannot copy pixels from to a multi-sampled 2d texture array");

  auto myWidth = getWidth();
  auto myHeight = getHeight();
  RUNTIME_EX_ASSERT(startx < myWidth && startx + width <= myWidth && starty < myHeight && starty + height <= myHeight,
                    "The area of pixels to set: " + std::to_string(width) + "x" + std::to_string(height) +
                        " starting at [" + std::to_string(startx) + ", " + std::to_string(starty) +
                        "] extends beyond the bounds of the textures in the GLTexture2dArray: " +
                        std::to_string(myWidth) + "x" + std::to_string(myHeight) + ".");

  // TODO(croot): do something to verify that the pixel format uploaded matches that of the internal texture?

  // TODO(croot): support mipmaps
  // TODO(croot): support byte offset into pbo data?

  CHECK(_target == GL_TEXTURE_2D_ARRAY);
  GLint currTex;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &currTex));
  MAPD_CHECK_GL_ERROR(glBindTexture(_target, _textureArrayId));

  if (arrayIdx < 0) {
    for (int i = 0; i < static_cast<int>(_depth); ++i) {
      MAPD_CHECK_GL_ERROR(
          glTexSubImage3D(_target, 0, startx, starty, i, width, height, 1, pixelFormat, pixelType, pixelData));
    }
  } else {
    MAPD_CHECK_GL_ERROR(
        glTexSubImage3D(_target, 0, startx, starty, arrayIdx, width, height, 1, pixelFormat, pixelType, pixelData));
  }

  if (currTex != static_cast<GLint>(_textureArrayId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(_target, currTex));
  }
}

void GLTexture2dArray::getPixels(const GLenum pixelFormat, const GLenum pixelType, GLvoid* pixelData) {
  // TODO(croot): check that the pixel format/type compatible with
  // the internal format of the GLTexture2dArray

  // TODO(croot): should we templatize this function so that the pixelFormat/type are
  // implied? Doing so could also indicate size. Otherwise it's up to the user
  // here to make sure pixelData is properly allocated to handle the pixel transfer.
  // If not, a crash will likely ensue.

  // TODO(croot): Should we take into account GL_PACK_ALIGNMENT parameters?
  RUNTIME_EX_ASSERT(_numSamples == 1, "Cannot get pixels from a multi-sampled 2d texture");

  CHECK(_target == GL_TEXTURE_2D_ARRAY);
  GLint currTex;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &currTex));
  MAPD_CHECK_GL_ERROR(glBindTexture(_target, _textureArrayId));
  MAPD_CHECK_GL_ERROR(glGetTexImage(_target, 0, pixelFormat, pixelType, pixelData));

  if (currTex != static_cast<GLint>(_textureArrayId)) {
    // now reset the context bound state back
    MAPD_CHECK_GL_ERROR(glBindTexture(_target, currTex));
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
