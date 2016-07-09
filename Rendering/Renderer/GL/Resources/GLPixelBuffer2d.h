#ifndef RENDERING_GL_RESOURCES_GLPIXELBUFFER2D_H_
#define RENDERING_GL_RESOURCES_GLPIXELBUFFER2D_H_

#include "GLBaseBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLPixelBuffer2d : public GLBaseBuffer {
 public:
  ~GLPixelBuffer2d();

  size_t getWidth() const { return _width; }
  size_t getHeight() const { return _height; }
  GLenum getPixelFormat() const { return _pixelFormat; }
  GLenum getPixelType() const { return _pixelType; }

  void resize(size_t width, size_t height, GLvoid* data = nullptr);

  void readPixels(size_t width, size_t height, GLenum dataType, GLvoid* data);

 private:
  explicit GLPixelBuffer2d(const RendererWkPtr& rendererPtr,
                           size_t width,
                           size_t height,
                           // GLenum internalFormat,
                           GLenum pixelFormat,
                           GLenum pixelType,
                           BufferAccessType accessType = BufferAccessType::READ,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC);

  explicit GLPixelBuffer2d(const RendererWkPtr& rendererPtr,
                           size_t width,
                           size_t height,
                           GLenum pixelFormat,
                           GLenum pixelType,
                           GLvoid* data,
                           BufferAccessType accessType = BufferAccessType::READ,
                           BufferAccessFreq accessFreq = BufferAccessFreq::DYNAMIC);

  void _makeEmpty() final;

  size_t _width;
  size_t _height;
  // GLenum _internalFormat;
  GLenum _pixelFormat;
  GLenum _pixelType;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLPIXELBUFFER2D_H_
