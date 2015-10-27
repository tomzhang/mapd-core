#ifndef VERTEX_BUFFER_H_
#define VERTEX_BUFFER_H_

// #include "Buffer.h"
#include "BufferLayout.h"
#include "Shader.h"
#include <GL/glew.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace MapD_Renderer {

class VertexBuffer {
 public:
  typedef std::shared_ptr<BaseBufferLayout> BufferLayoutShPtr;

  explicit VertexBuffer(const BufferLayoutShPtr& layoutPtr,
                        GLenum target = GL_ARRAY_BUFFER,
                        GLenum usage = GL_STATIC_DRAW)
      : _size(0), _bufferId(0), _target(target), _usage(usage), _layoutPtr(layoutPtr) {}

  template <typename T>
  VertexBuffer(const std::vector<T>& data,
               const BufferLayoutShPtr& layoutPtr,
               GLenum target = GL_ARRAY_BUFFER,
               GLenum usage = GL_STATIC_DRAW)
      : _size(0), _bufferId(0), _target(target), _usage(usage), _layoutPtr(layoutPtr) {
    // TODO(croot): validate that the data and the layout align

    // _size will be set in the bufferData() call
    bufferData((void*)&data[0], data.size(), sizeof(T));
  }

  ~VertexBuffer() {
    // std::cout << "IN VertexBuffer DESTRUCTOR" << std::endl;
    if (_bufferId) {
      glDeleteBuffers(1, &_bufferId);
    }
  }

  BufferLayoutShPtr getBufferLayout() const { return BufferLayoutShPtr(_layoutPtr); }

  bool hasAttribute(const std::string& attrName) { return _layoutPtr->hasAttribute(attrName); }

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) { return _layoutPtr->getAttributeTypeGL(attrName); }

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

  int size() const { return _size; }

  void bindToRenderer(Shader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
    if (!_bufferId) {
      throw std::runtime_error("Cannot bind vertex buffer. It has not been initialized with data.");
    } else if (!_layoutPtr) {
      throw std::runtime_error("Cannot bind vertex buffer. It does not have a defined layout.");
    }
    glBindBuffer(GL_ARRAY_BUFFER, _bufferId);
    _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
  }

 private:
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

  int _size;
  GLuint _bufferId;
  GLenum _target;
  GLenum _usage;

  std::shared_ptr<BaseBufferLayout> _layoutPtr;

  void _initBuffer() {
    if (!_bufferId) {
      glGenBuffers(1, &_bufferId);
    }
  }
};

typedef std::unique_ptr<VertexBuffer> VertexBufferUqPtr;
typedef std::shared_ptr<VertexBuffer> VertexBufferShPtr;

}  // namespace MapD_Renderer

#endif  // VERTEX_BUFFER_H_
