#ifndef RENDERING_GL_RESOURCES_GLRENDERBUFFER_H_
#define RENDERING_GL_RESOURCES_GLRENDERBUFFER_H_

#include "../Types.h"
#include "Types.h"
#include "GLResource.h"
#include <GL/glew.h>

namespace Rendering {
namespace GL {

namespace Resources {

class GLRenderbuffer : public GLResource {
 public:
  ~GLRenderbuffer();

  GLResourceType getResourceType() const final { return GLResourceType::RENDERBUFFER; }
  GLuint getId() const final { return _bufferId; }
  GLenum getTarget() const final { return GL_RENDERBUFFER; }

  size_t getWidth() { return _width; }
  size_t getHeight() { return _height; }
  GLenum getInternalFormat() { return _internalFormat; }
  size_t getNumSamples() { return _numSamples; }

  void resize(size_t width, size_t height);

  void bindToRenderer(GLRenderer* renderer) final;

 private:
  explicit GLRenderbuffer(const RendererWkPtr& rendererPtr,
                          size_t width,
                          size_t height,
                          GLenum internalFormat,
                          size_t numSamples = 1);

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;

  size_t _width;
  size_t _height;
  GLenum _internalFormat;
  size_t _numSamples;

  GLuint _bufferId;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLRENDERBUFFER_H_
