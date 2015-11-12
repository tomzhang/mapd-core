#ifndef BUFFER_H_
#define BUFFER_H_

#include "MapDGL.h"
#include <vector>
#include <GL/glew.h>
#include <iostream>

namespace MapD_Renderer {

class Buffer {
 public:
  Buffer() : _bufferId(0) { std::cout << "IN buffer CONSTRUCTOR " << _bufferId << std::endl; }

  // template <typename T>
  // Buffer(std::vector<T>);

  // TODO: Should we make bufferId a shared resource?
  Buffer(GLuint bufferId) : _bufferId(bufferId) { std::cout << "IN buffer CONSTRUCTOR " << _bufferId << std::endl; }

  virtual ~Buffer() {
    // std::cout << "IN Buffer DESTRUCTOR" << std::endl;
    if (_bufferId) {
      MAPD_CHECK_GL_ERROR(glDeleteBuffers(1, &_bufferId));
    }
  }

  virtual void bufferData(void* data, int numItems, int numBytesPerItem) = 0;
  virtual int size() const = 0;

 private:
 protected:
  GLuint _bufferId;
  void _initBuffer() {
    if (!_bufferId) {
      MAPD_CHECK_GL_ERROR(glGenBuffers(1, &_bufferId));
    }
  }
};

}  // namespace MapD_Renderer

#endif  // BUFFER_H_
