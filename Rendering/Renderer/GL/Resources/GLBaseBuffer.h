#ifndef RENDERING_GL_RESOURCES_GLBASEBUFFER_H_
#define RENDERING_GL_RESOURCES_GLBASEBUFFER_H_

#include "Types.h"
#include "GLResource.h"
#include <GL/glew.h>

namespace Rendering {
namespace GL {
namespace Resources {

class GLBaseBuffer : public GLResource {
 public:
  virtual ~GLBaseBuffer();

  GLuint getId() const final { return _bufferId; };

  size_t numBytes() const { return _numBytes; }
  GLBufferType type() const { return _type; }

 protected:
  explicit GLBaseBuffer(const RendererWkPtr& rendererPtr,
                        GLBufferType type,
                        GLenum target = GL_ARRAY_BUFFER,
                        GLenum usage = GL_STATIC_DRAW);

  void _initResource();

  virtual void _cleanupResource() override;
  virtual void _makeEmpty() override;
  static GLenum _getBufferBinding(GLenum target);
  void bufferData(void* data, size_t numBytes);

  GLBufferType _type;
  GLuint _bufferId;
  GLenum _target;
  GLenum _usage;

  size_t _numBytes;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLBASEBUFFER_H_
