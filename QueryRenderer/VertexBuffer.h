#ifndef VERTEX_BUFFER_H_
#define VERTEX_BUFFER_H_

// #include "Buffer.h"
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

namespace MapD_Renderer {

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
      glDeleteBuffers(1, &_bufferId);
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
    if (!_layoutPtr) {
      std::runtime_error err(
          "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute GL Type of " + attrName +
          ".");

      LOG(ERROR) << err.what();
      throw err;
    }
    return _layoutPtr->getAttributeTypeGL(attrName);
  }

  BufferAttrType getAttributeType(const std::string& attrName) const {
    if (!_layoutPtr) {
      std::runtime_error err(
          "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute Type of " + attrName +
          ".");

      LOG(ERROR) << err.what();
      throw err;
    }

    return _layoutPtr->getAttributeType(attrName);
  }

  BufferLayoutShPtr getBufferLayout() const { return BufferLayoutShPtr(_layoutPtr); }

  virtual void setBufferLayout() {
    std::runtime_error err("Cannot set the buffer layout of a vertex buffer.");
    LOG(ERROR) << err.what();
    throw err;
  }

  void bindToRenderer(Shader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
    if (!_bufferId) {
      throw std::runtime_error("Cannot bind vertex buffer. It has not been initialized with data.");
    } else if (!_layoutPtr) {
      throw std::runtime_error("Cannot bind vertex buffer. It does not have a defined layout.");
    }
    glBindBuffer(GL_ARRAY_BUFFER, _bufferId);
    _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
  }

 protected:
  static GLenum _getBufferBinding(GLenum target) {
    typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
    static const BufferBindingMap bufferBindings = {{GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
                                                    {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}};

    BufferBindingMap::const_iterator itr;

    if ((itr = bufferBindings.find(target)) == bufferBindings.end()) {
      throw std::runtime_error(std::to_string(target) + " is not a valid opengl buffer target");
    }

    return itr->first;
  }

  VertexBufferType _type;

  GLuint _bufferId;
  GLenum _target;
  GLenum _usage;

  size_t _size;
  BufferLayoutShPtr _layoutPtr;

  void _initBuffer() {
    if (!_bufferId) {
      glGenBuffers(1, &_bufferId);
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
    //   glDeleteBuffers(1, &_bufferId);
    // }
  }

  // BufferLayoutShPtr getBufferLayout() const { return BufferLayoutShPtr(_layoutPtr); }

  // TypeGLShPtr getAttributeTypeGL(const std::string& attrName) { return _layoutPtr->getAttributeTypeGL(attrName); }

  // void setBufferLayout(const std::shared_ptr<BaseBufferLayout>& layoutPtr) {
  //     _layoutPtr = layoutPtr;
  // }

  void bufferData(void* data, int numItems, int numBytesPerItem) {
    _initBuffer();
    // glGenBuffers(1, &_bufferId);

    // don't mess with the current state
    // TODO(croot): Apply some kind of push-pop state system
    GLint currArrayBuf;
    glGetIntegerv(_getBufferBinding(_target), &currArrayBuf);

    glBindBuffer(_target, _bufferId);
    glBufferData(_target, numItems * numBytesPerItem, data, _usage);

    // restore the state
    glBindBuffer(_target, currArrayBuf);

    _size = numItems;
  }

  // int size() const { return _size; }

  // void bindToRenderer(Shader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
  //   if (!_bufferId) {
  //     throw std::runtime_error("Cannot bind vertex buffer. It has not been initialized with data.");
  //   } else if (!_layoutPtr) {
  //     throw std::runtime_error("Cannot bind vertex buffer. It does not have a defined layout.");
  //   }
  //   glBindBuffer(GL_ARRAY_BUFFER, _bufferId);
  //   _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
  // }

 private:
  // static GLenum _getBufferBinding(GLenum target) {
  //   typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
  //   static const BufferBindingMap bufferBindings = {{GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
  //                                                   {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}};

  //   BufferBindingMap::const_iterator itr;

  //   if ((itr = bufferBindings.find(target)) == bufferBindings.end()) {
  //     throw std::runtime_error(std::to_string(target) + " is not a valid opengl buffer target");
  //   }

  //   return itr->first;
  // }

  // int _size;
  // GLuint _bufferId;
  // GLenum _target;
  // GLenum _usage;

  // BufferLayoutShPtr _layoutPtr;

  // void _initBuffer() {
  //   if (!_bufferId) {
  //     glGenBuffers(1, &_bufferId);
  //   }
  // }
};

typedef std::unique_ptr<VertexBuffer> VertexBufferUqPtr;
typedef std::shared_ptr<VertexBuffer> VertexBufferShPtr;

}  // namespace MapD_Renderer

#endif  // VERTEX_BUFFER_H_
