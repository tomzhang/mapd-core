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

  int getWidth() { return _width; }
  int getHeight() { return _height; }
  GLenum getInternalFormat() { return _internalFormat; }
  int getNumSamples() { return _numSamples; }

  void resize(int width, int height);

  void bindToRenderer(GLRenderer* renderer) final;

 private:
  GLRenderbuffer(const RendererWkPtr& rendererPtr, int width, int height, GLenum internalFormat, int numSamples = 1);

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;

  int _width;
  int _height;
  GLenum _internalFormat;
  int _numSamples;

  GLuint _bufferId;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLRENDERBUFFER_H_
