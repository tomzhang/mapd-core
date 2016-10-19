#ifndef QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_
#define QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_

#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/GLPixelBuffer2d.h>

#include <memory>

namespace QueryRenderer {

template <typename T>
class QueryIdMapPixelBuffer {
 public:
  QueryIdMapPixelBuffer(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height) : _pbo(nullptr) {
    _init(renderer, width, height);
  }

  ~QueryIdMapPixelBuffer() {}

  size_t getWidth() const {
    if (_pbo) {
      return _pbo->getWidth();
    }
    return 0;
  }

  size_t getHeight() const {
    if (_pbo) {
      return _pbo->getHeight();
    }
    return 0;
  }

  void resize(size_t width, size_t height) {
    CHECK(_pbo);

    _pbo->resize(width, height);
  }

  void readIdBuffer(int width, int height, T* idBuffer);

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr getPixelBuffer2d() const { return _pbo; }

 private:
  void _init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height);

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr _pbo;
};

template <>
void QueryIdMapPixelBuffer<unsigned int>::_init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height);

template <>
void QueryIdMapPixelBuffer<unsigned int>::readIdBuffer(int width, int height, unsigned int* idBuffer);

template <>
void QueryIdMapPixelBuffer<int>::_init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height);

template <>
void QueryIdMapPixelBuffer<int>::readIdBuffer(int width, int height, int* idBuffer);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_
