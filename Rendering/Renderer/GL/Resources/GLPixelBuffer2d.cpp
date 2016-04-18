#include "GLPixelBuffer2d.h"
#include "../GLUtility.h"
#include "../MapDGL.h"
#include <sstream>
#include <cstring>

namespace Rendering {
namespace GL {
namespace Resources {

static GLenum getBufferTarget(BufferAccessType accessType) {
  switch (accessType) {
    case BufferAccessType::READ:
      return GL_PIXEL_PACK_BUFFER;
    case BufferAccessType::WRITE:
      return GL_PIXEL_UNPACK_BUFFER;
    case BufferAccessType::READ_AND_WRITE:
      return GL_PIXEL_UNPACK_BUFFER;
    case BufferAccessType::COPY:
      return GL_PIXEL_PACK_BUFFER;
  }

  return 0;
}

GLPixelBuffer2d::GLPixelBuffer2d(const RendererWkPtr& rendererPtr,
                                 size_t width,
                                 size_t height,
                                 // GLenum internalFormat,
                                 GLenum pixelFormat,
                                 GLenum pixelType,
                                 BufferAccessType accessType,
                                 BufferAccessFreq accessFreq)
    : GLBaseBuffer(rendererPtr,
                   GLResourceType::PIXELBUFFER_2D,
                   GLBufferType::PIXEL_BUFFER_2D,
                   getBufferTarget(accessType),
                   accessType,
                   accessFreq),
      _width(0),
      _height(0),
      // _internalFormat(internalFormat),
      _pixelFormat(pixelFormat),
      _pixelType(pixelType) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0,
                    "Invalid dimensions for the 2d pixel buffer:" + std::to_string(width) + "x" +
                        std::to_string(height) + ". Dimensions must be > 0");

  resize(width, height);
}

GLPixelBuffer2d::~GLPixelBuffer2d() {
  cleanupResource();
}

void GLPixelBuffer2d::_makeEmpty() {
  _width = 0;
  _height = 0;
}

void GLPixelBuffer2d::resize(size_t width, size_t height) {
  if (width != _width || height != _height) {
    RUNTIME_EX_ASSERT(width > 0 && height > 0,
                      "Invalid dimensions " + std::to_string(width) + "x" + std::to_string(height) +
                          " for the texture. Dimensions must be > 0");

    bufferData(nullptr, width * height * pixelTypeSize(_pixelType));

    _width = width;
    _height = height;
  }
}

void GLPixelBuffer2d::readPixels(size_t width, size_t height, GLenum dataType, GLvoid* data) {
  validateUsability(__FILE__, __LINE__);

  RUNTIME_EX_ASSERT(isReadable(), "The pbo is set to be writable, not readable. Cannot read pixels.");

  if (dataType != _pixelType) {
    std::stringstream ss1, ss2;
    ss1 << "0x" << std::hex << dataType;
    ss2 << "0x" << std::hex << _pixelType;

    THROW_RUNTIME_EX("Invalid data type for pixels: " + ss1.str() + ". The pbo is of type " + ss2.str() +
                     " and can only read pixels of that type.")
  }

  RUNTIME_EX_ASSERT(width == _width && height == _height,
                    "Invalid dimensions of data buffer to read pixels into. The data buffer is " +
                        std::to_string(width) + "x" + std::to_string(height) + ", but the pixel buffer is " +
                        std::to_string(_width) + std::to_string(_height));

  GLint currReadPbo;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &currReadPbo));

  bool bind = (currReadPbo != static_cast<GLint>(_bufferId));
  if (bind) {
    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_PIXEL_PACK_BUFFER, _bufferId));
  }

  void* ptr = MAPD_CHECK_GL_ERROR(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));

  RUNTIME_EX_ASSERT(ptr != nullptr, "Cannot map pixel buffer due to error: " + getGLErrorStr());

  std::memcpy(data, ptr, width * height * pixelTypeSize(dataType));

  MAPD_CHECK_GL_ERROR(glUnmapBuffer(GL_PIXEL_PACK_BUFFER));

  if (bind) {
    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_PIXEL_PACK_BUFFER, currReadPbo));
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
