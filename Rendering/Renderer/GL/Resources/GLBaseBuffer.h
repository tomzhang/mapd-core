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

  GLuint getId() const final { return _bufferId; }
  GLenum getTarget() const final { return _target; }

  BufferAccessType getAccessType() const { return _accessType; }
  BufferAccessFreq getAccessFreq() const { return _accessFreq; }
  GLenum getGLUsage() const { return _usage; }

  size_t getNumBytes() const { return _numBytes; }
  GLBufferType getType() const { return _type; }

  bool isReadable() const {
    return (_accessType == BufferAccessType::READ || _accessType == BufferAccessType::READ_AND_WRITE ||
            _accessType == BufferAccessType::COPY);
  }

  bool isWritable() const {
    return (_accessType == BufferAccessType::WRITE || _accessType == BufferAccessType::READ_AND_WRITE ||
            _accessType == BufferAccessType::COPY);
  }

  virtual void rebuild(const void* data, size_t numBytes);
  virtual void bufferData(const void* data, size_t numBytes, GLenum altTarget = 0);
  virtual void bufferSubData(const void* data, size_t numBytes, size_t byteOffset, GLenum altTarget = 0);
  virtual void getBufferData(void* data, size_t dataSzBytes, size_t byteOffset = 0);

 protected:
  explicit GLBaseBuffer(const RendererWkPtr& rendererPtr,
                        GLResourceType rsrcType,
                        GLBufferType type,
                        GLenum target = GL_ARRAY_BUFFER,
                        BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                        BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  void _initResource();

  virtual void _cleanupResource() override;
  virtual void _makeEmpty() override;
  static GLenum _getBufferBinding(GLenum target);

  GLBufferType _type;
  GLuint _bufferId;
  GLenum _target;
  GLenum _usage;

  BufferAccessType _accessType;
  BufferAccessFreq _accessFreq;

  size_t _numBytes;

  // TODO(croot): should we add a data structure that manages intervals
  // of the data that has been initialized with data?
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLBASEBUFFER_H_
