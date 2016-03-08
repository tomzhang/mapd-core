#include "GLBaseBuffer.h"
#include "../MapDGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLBaseBuffer::GLBaseBuffer(const RendererWkPtr& rendererPtr, GLBufferType type, GLenum target, GLenum usage)
    : GLResource(rendererPtr), _type(type), _bufferId(0), _target(target), _usage(usage), _numBytes(0) {
  _initResource();
}

GLBaseBuffer::~GLBaseBuffer() {
  cleanupResource();
}

void GLBaseBuffer::_initResource() {
  validateRenderer();

  if (!_bufferId) {
    MAPD_CHECK_GL_ERROR(glGenBuffers(1, &_bufferId));
  }
}

void GLBaseBuffer::_cleanupResource() {
  if (_bufferId) {
    MAPD_CHECK_GL_ERROR(glDeleteBuffers(1, &_bufferId));
  }

  _makeEmpty();
}

void GLBaseBuffer::_makeEmpty() {
  _bufferId = 0;
  _numBytes = 0;
}

void GLBaseBuffer::bufferData(void* data, size_t numBytes) {
  // TODO(croot): this could be a performance hit when buffering data
  // frequently. Should perhaps look into buffer object streaming techniques:
  // https://www.opengl.org/wiki/Buffer_Object_Streaming
  // Gonna start with an orphaning technique/re-specification

  validateRenderer();

  // don't mess with the current state
  // TODO(croot): Apply some kind of push-pop state system
  GLint currArrayBuf;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(_getBufferBinding(_target), &currArrayBuf));

  MAPD_CHECK_GL_ERROR(glBindBuffer(_target, _bufferId));

  // passing "NULL" to glBufferData first "may" induce orphaning.
  // See: https://www.opengl.org/wiki/Buffer_Object_Streaming#Buffer_re-specification
  // This is implementation dependent, and not guaranteed to be implemented
  // properly. If not, then this will force a synchronization, which
  // can be slow. If implemented properly, this should allocate a new
  // buffer without clashing with the old one. The old one will be
  // freed once all GL commands in the command queue referencing the old
  // one are executed.
  // TODO(croot): the docs indicate that to orphan with glBufferData(...), the exact
  // same size and usage hints from before should be used. I'm not ensuring the
  // same size here. Is that a problem? Should I employ a glMapBufferRange(...) technique
  // instead?
  MAPD_CHECK_GL_ERROR(glBufferData(_target, numBytes, NULL, _usage));

  // Did the orphaning above, now actually buffer the data to the orphaned
  // buffer.
  // TODO(croot): use a pbuffer streaming technique instead perhaps?
  // That could be a faster, asynchronous way.
  if (data) {
    MAPD_CHECK_GL_ERROR(glBufferData(_target, numBytes, data, _usage));
  }

  // restore the state
  MAPD_CHECK_GL_ERROR(glBindBuffer(_target, currArrayBuf));

  _numBytes = numBytes;

  setUsable();
}

GLenum GLBaseBuffer::_getBufferBinding(GLenum target) {
  typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
  static const BufferBindingMap bufferBindings = {{GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
                                                  {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}};

  BufferBindingMap::const_iterator itr;

  RUNTIME_EX_ASSERT((itr = bufferBindings.find(target)) != bufferBindings.end(),
                    std::to_string(static_cast<int>(target)) + " is not a valid opengl buffer target.");

  return itr->second;
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
