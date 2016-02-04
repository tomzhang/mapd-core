#ifndef VERTEX_BUFFER_H_
#define VERTEX_BUFFER_H_

#include "MapDGL.h"
#include "QueryRendererError.h"
#include "BufferLayout.h"
#include "Shader.h"

#include <GL/glew.h>

#include <glog/logging.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace QueryRenderer {

class BaseVertexBuffer {
 public:
  enum class VertexBufferType { VERTEX_BUFFER = 0, QUERY_RESULT_VERTEX_BUFFER };

  BaseVertexBuffer(VertexBufferType type, GLenum target = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
      : _type(type), _bufferId(0), _target(target), _usage(usage), _size(0), _layoutPtr(nullptr) {}
  explicit BaseVertexBuffer(VertexBufferType type,
                            const BufferLayoutShPtr& layoutPtr,
                            GLenum target = GL_ARRAY_BUFFER,
                            GLenum usage = GL_STATIC_DRAW)
      : _type(type), _bufferId(0), _target(target), _usage(usage), _size(0), _layoutPtr(layoutPtr) {}
  virtual ~BaseVertexBuffer() {
    if (_bufferId) {
      MAPD_CHECK_GL_ERROR(glDeleteBuffers(1, &_bufferId));
    }
  }

  size_t size() const { return _size; }
  VertexBufferType type() const { return _type; }

  bool hasAttribute(const std::string& attrName) const {
    if (!_layoutPtr) {
      return false;
    }
    return _layoutPtr->hasAttribute(attrName);
  }

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const {
    RUNTIME_EX_ASSERT(_layoutPtr != nullptr,
                      "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute GL Type of " +
                          attrName + ".");

    return _layoutPtr->getAttributeTypeGL(attrName);
  }

  BufferAttrType getAttributeType(const std::string& attrName) const {
    RUNTIME_EX_ASSERT(
        _layoutPtr != nullptr,
        "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute Type of " + attrName + ".");

    return _layoutPtr->getAttributeType(attrName);
  }

  BufferLayoutShPtr getBufferLayout() const { return BufferLayoutShPtr(_layoutPtr); }

  void bindToRenderer(Shader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
    RUNTIME_EX_ASSERT(_bufferId != 0, "Cannot bind vertex buffer. It has not been initialized with data.");
    RUNTIME_EX_ASSERT(_layoutPtr != nullptr, "Cannot bind vertex buffer. It does not have a defined layout.");
    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, _bufferId));
    _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
  }

 protected:
  static GLenum _getBufferBinding(GLenum target) {
    typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
    static const BufferBindingMap bufferBindings = {{GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
                                                    {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}};

    BufferBindingMap::const_iterator itr;

    RUNTIME_EX_ASSERT((itr = bufferBindings.find(target)) != bufferBindings.end(),
                      std::to_string(static_cast<int>(target)) + " is not a valid opengl buffer target.");

    return itr->second;
  }

  VertexBufferType _type;

  GLuint _bufferId;
  GLenum _target;
  GLenum _usage;

  size_t _size;
  BufferLayoutShPtr _layoutPtr;

  void _initBuffer() {
    if (!_bufferId) {
      MAPD_CHECK_GL_ERROR(glGenBuffers(1, &_bufferId));
    }
  }
};

typedef std::shared_ptr<BaseVertexBuffer> BaseVertexBufferShPtr;
typedef std::unique_ptr<BaseVertexBuffer> BaseVertexBufferUqPtr;

class VertexBuffer : public BaseVertexBuffer {
 public:
  typedef std::shared_ptr<BaseBufferLayout> BufferLayoutShPtr;

  explicit VertexBuffer(const BufferLayoutShPtr& layoutPtr,
                        GLenum target = GL_ARRAY_BUFFER,
                        GLenum usage = GL_STATIC_DRAW)
      : BaseVertexBuffer(BaseVertexBuffer::VertexBufferType::VERTEX_BUFFER, layoutPtr, target, usage) {}

  template <typename T>
  VertexBuffer(const std::vector<T>& data,
               const BufferLayoutShPtr& layoutPtr,
               GLenum target = GL_ARRAY_BUFFER,
               GLenum usage = GL_STATIC_DRAW)
      : BaseVertexBuffer(BaseVertexBuffer::VertexBufferType::VERTEX_BUFFER, layoutPtr, target, usage) {
    // TODO(croot): validate that the data and the layout align

    // _size will be set in the bufferData() call
    bufferData((void*)&data[0], data.size(), sizeof(T));
  }

  ~VertexBuffer() {
    // if (_bufferId) {
    //   MAPD_CHECK_GL_ERROR(glDeleteBuffers(1, &_bufferId));
    // }
  }

  // BufferLayoutShPtr getBufferLayout() const { return BufferLayoutShPtr(_layoutPtr); }

  // TypeGLShPtr getAttributeTypeGL(const std::string& attrName) { return _layoutPtr->getAttributeTypeGL(attrName); }

  void bufferData(void* data, int numItems, int numBytesPerItem) {
    // TODO(croot): this could be a performance hit when buffering data
    // frequently. Should perhaps look into buffer object streaming techniques:
    // https://www.opengl.org/wiki/Buffer_Object_Streaming
    // Gonna start with an orphaning technique/re-specification

    _initBuffer();
    // MAPD_CHECK_GL_ERROR(glGenBuffers(1, &_bufferId));

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
    MAPD_CHECK_GL_ERROR(glBufferData(_target, numItems * numBytesPerItem, NULL, _usage));

    // Did the orphaning above, now actually buffer the data to the orphaned
    // buffer.
    // TODO(croot): use a pbuffer streaming technique instead perhaps?
    // That could be a faster, asynchronous way.
    MAPD_CHECK_GL_ERROR(glBufferData(_target, numItems * numBytesPerItem, data, _usage));

    // restore the state
    MAPD_CHECK_GL_ERROR(glBindBuffer(_target, currArrayBuf));

    _size = numItems;
  }

 private:
};

typedef std::unique_ptr<VertexBuffer> VertexBufferUqPtr;
typedef std::shared_ptr<VertexBuffer> VertexBufferShPtr;

}  // namespace QueryRenderer

#endif  // VERTEX_BUFFER_H_
