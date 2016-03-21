#ifndef RENDERING_GL_RESOURCES_GLPIXELBUFFER2D_H_
#define RENDERING_GL_RESOURCES_GLPIXELBUFFER2D_H_

#include "GLBaseBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLPixelBuffer2d : public GLBaseBuffer {
 public:
  ~GLPixelBuffer2d();

  GLResourceType getResourceType() const final { return GLResourceType::PIXELBUFFER_2D; }

  size_t getWidth() const { return _width; }
  size_t getHeight() const { return _height; }

  void resize(size_t width, size_t height);

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
